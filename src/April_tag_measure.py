#!/usr/bin/env python3
"""
apriltag_ruler.py

Interactive ruler using an AprilTag.

Input:
  - image.jpg

Interactive output:
  - Detect AprilTag
  - Click + drag to draw a line
  - Measure line length in meters (assuming the line lies on the same plane as the tag)

Save output:
  - annotated image
  - results JSON

Measurement modes:
  A) Quick scale mode (default): no intrinsics required
     length_m = pixel_length / (px_per_meter_from_tag)

  B) Accurate planar mode: requires camera intrinsics (fx, fy, cx, cy)
     Uses solvePnP to estimate tag plane in camera frame, then intersects pixel rays with that plane.

Keys:
  - Left mouse drag: draw measurement line
  - r: reset all lines
  - s: save annotated image + JSON
  - q or ESC: quit
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import cv2


# -------------------------
# AprilTag detector backends
# -------------------------

def detect_apriltags(gray: np.ndarray, family: str) -> List[Dict[str, Any]]:
    """
    Returns a list of detections.
    Each detection has:
      - id (int, if available else -1)
      - corners: (4,2) float32 in image pixels, order roughly [tl,tr,br,bl]
      - center: (2,) float32
    """
    # 1) Try pupil_apriltags (fast + good)
    try:
        from pupil_apriltags import Detector  # type: ignore

        det = Detector(families=family)
        dets = det.detect(gray)
        out = []
        for d in dets:
            corners = np.array(d.corners, dtype=np.float32)  # (4,2)
            center = np.array(d.center, dtype=np.float32)    # (2,)
            out.append({"id": int(getattr(d, "tag_id", -1)), "corners": corners, "center": center})
        return out
    except Exception:
        pass

    # 2) Fallback: OpenCV aruco AprilTag dictionaries
    try:
        aruco = cv2.aruco  # type: ignore

        # Map common AprilTag families to OpenCV dict constants
        fam_map = {
            "tag16h5": getattr(aruco, "DICT_APRILTAG_16h5", None),
            "tag25h9": getattr(aruco, "DICT_APRILTAG_25h9", None),
            "tag36h10": getattr(aruco, "DICT_APRILTAG_36h10", None),
            "tag36h11": getattr(aruco, "DICT_APRILTAG_36h11", None),
        }
        dict_id = fam_map.get(family, fam_map["tag36h11"])
        if dict_id is None:
            raise RuntimeError("OpenCV build missing AprilTag dictionaries")

        dictionary = aruco.getPredefinedDictionary(dict_id)
        params = aruco.DetectorParameters()

        # OpenCV 4.7+ has ArucoDetector
        if hasattr(aruco, "ArucoDetector"):
            detector = aruco.ArucoDetector(dictionary, params)
            corners_list, ids, _ = detector.detectMarkers(gray)
        else:
            corners_list, ids, _ = aruco.detectMarkers(gray, dictionary, parameters=params)

        out = []
        if ids is None or len(ids) == 0:
            return out

        for corners, tag_id in zip(corners_list, ids.flatten()):
            c = np.array(corners, dtype=np.float32).reshape(-1, 2)  # (4,2)
            center = c.mean(axis=0)
            out.append({"id": int(tag_id), "corners": c, "center": center})
        return out
    except Exception:
        return []


# -------------------------
# Measurement logic
# -------------------------

def avg_tag_edge_px(corners: np.ndarray) -> float:
    """Average edge length in pixels from 4 tag corners."""
    c = corners.reshape(4, 2)
    edges = [
        np.linalg.norm(c[1] - c[0]),
        np.linalg.norm(c[2] - c[1]),
        np.linalg.norm(c[3] - c[2]),
        np.linalg.norm(c[0] - c[3]),
    ]
    return float(np.mean(edges))


def build_camera_matrix(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    return np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0,  0,  1]], dtype=np.float64)


def solve_tag_pose(corners_px: np.ndarray,
                   tag_size_m: float,
                   K: np.ndarray,
                   dist: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate pose of tag using solvePnP.

    Returns:
      R (3,3) rotation matrix (object->camera)
      t (3,1) translation vector (object origin in camera frame)
    """
    # Define tag coordinate frame with origin at tag center on plane Z=0.
    # Corner ordering expected: [tl, tr, br, bl] (approx).
    s = tag_size_m / 2.0
    obj_pts = np.array([
        [-s,  s, 0],  # tl
        [ s,  s, 0],  # tr
        [ s, -s, 0],  # br
        [-s, -s, 0],  # bl
    ], dtype=np.float64)

    img_pts = corners_px.astype(np.float64).reshape(4, 2)

    if dist is None:
        dist = np.zeros((5, 1), dtype=np.float64)

    ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        # Try a more general fallback
        ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise RuntimeError("solvePnP failed")

    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3, 1)
    return R, t


def intersect_ray_with_tag_plane(u: float, v: float,
                                 K: np.ndarray,
                                 dist: Optional[np.ndarray],
                                 R: np.ndarray,
                                 t: np.ndarray) -> np.ndarray:
    """
    Intersect pixel ray with the tag plane (Z=0 in tag frame), expressed in camera frame via R,t.

    We use plane in camera frame:
      - plane normal n_c = R[:,2]   (tag +Z axis in camera frame)
      - point on plane p0_c = t     (tag origin in camera frame)

    Pixel ray:
      x = inv(K) [u v 1]^T  (optionally undistorted)
      X(s) = s * x
    Solve s = (n·p0)/(n·x)
    """
    if dist is not None and np.linalg.norm(dist) > 0:
        # Undistort single point to normalized image coords
        pts = np.array([[[u, v]]], dtype=np.float64)
        und = cv2.undistortPoints(pts, K, dist, P=K)  # back to pixel coords under P=K
        u, v = float(und[0, 0, 0]), float(und[0, 0, 1])

    x = np.linalg.inv(K) @ np.array([[u], [v], [1.0]], dtype=np.float64)  # direction (not normalized)
    d = x.reshape(3, 1)

    n = R[:, 2].reshape(3, 1)  # plane normal in camera frame
    p0 = t                      # point on plane (tag origin in camera frame)

    denom = float(n.T @ d)
    if abs(denom) < 1e-9:
        raise RuntimeError("Ray is parallel to tag plane")

    s = float(n.T @ p0) / denom
    Xc = s * d  # intersection in camera frame (3,1)

    # Convert to tag/object frame: Xo = R^T (Xc - t)
    Xo = R.T @ (Xc - t)
    return Xo.reshape(3)


# -------------------------
# UI / interaction
# -------------------------

@dataclass
class LineMeasurement:
    p0: Tuple[int, int]
    p1: Tuple[int, int]
    length_m: float
    mode: str  # "quick" or "pose"


@dataclass
class Results:
    image: str
    tag_id: int
    tag_family: str
    tag_size_m: float
    px_per_meter_quick: Optional[float]
    pose_available: bool
    lines: List[LineMeasurement]


class InteractiveRuler:
    def __init__(self,
                 img_bgr: np.ndarray,
                 tag_det: Dict[str, Any],
                 tag_family: str,
                 tag_size_m: float,
                 K: Optional[np.ndarray],
                 dist: Optional[np.ndarray]):
        self.base = img_bgr.copy()
        self.img = img_bgr.copy()
        self.tag_det = tag_det
        self.tag_family = tag_family
        self.tag_size_m = tag_size_m

        self.K = K
        self.dist = dist
        self.pose_available = False
        self.R = None
        self.t = None

        corners = self.tag_det["corners"]
        self.px_per_meter_quick = avg_tag_edge_px(corners) / tag_size_m if tag_size_m > 0 else None

        if self.K is not None:
            try:
                R, t = solve_tag_pose(corners, tag_size_m, self.K, self.dist)
                self.R, self.t = R, t
                self.pose_available = True
            except Exception:
                self.pose_available = False

        self.dragging = False
        self.p0 = (0, 0)
        self.p1 = (0, 0)
        self.lines: List[LineMeasurement] = []

    def measure_line_m(self, p0: Tuple[int, int], p1: Tuple[int, int]) -> LineMeasurement:
        x0, y0 = p0
        x1, y1 = p1
        pix_len = math.hypot(x1 - x0, y1 - y0)

        # Prefer pose mode if available
        if self.pose_available and self.K is not None and self.R is not None and self.t is not None:
            X0 = intersect_ray_with_tag_plane(x0, y0, self.K, self.dist, self.R, self.t)
            X1 = intersect_ray_with_tag_plane(x1, y1, self.K, self.dist, self.R, self.t)
            length_m = float(np.linalg.norm(X1[:2] - X0[:2]))  # planar distance on tag plane
            return LineMeasurement(p0=p0, p1=p1, length_m=length_m, mode="pose")

        # Fallback quick scale
        if self.px_per_meter_quick is None or self.px_per_meter_quick <= 0:
            raise RuntimeError("No valid scale for quick mode")
        length_m = pix_len / self.px_per_meter_quick
        return LineMeasurement(p0=p0, p1=p1, length_m=length_m, mode="quick")

    def redraw(self, preview_line: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None) -> None:
        self.img = self.base.copy()

        # Draw tag corners
        corners = self.tag_det["corners"].astype(int)
        cv2.polylines(self.img, [corners.reshape(-1, 1, 2)], True, (0, 255, 0), 2)
        c = tuple(self.tag_det["center"].astype(int))
        cv2.circle(self.img, c, 4, (0, 255, 0), -1)

        # Status text
        mode = "pose" if self.pose_available else "quick"
        status = f"Tag {self.tag_det['id']} | mode={mode} | tag_size={self.tag_size_m:.4f} m"
        cv2.putText(self.img, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 3, cv2.LINE_AA)
        cv2.putText(self.img, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 1, cv2.LINE_AA)

        if not self.pose_available and self.px_per_meter_quick is not None:
            s2 = f"quick scale: {self.px_per_meter_quick:.1f} px/m (best when tag faces camera)"
            cv2.putText(self.img, s2, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 3, cv2.LINE_AA)
            cv2.putText(self.img, s2, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)

        # Draw existing measurements
        for i, lm in enumerate(self.lines, start=1):
            cv2.line(self.img, lm.p0, lm.p1, (255, 0, 255), 2)
            mid = ((lm.p0[0] + lm.p1[0]) // 2, (lm.p0[1] + lm.p1[1]) // 2)
            text = f"{i}: {lm.length_m:.4f} m ({lm.mode})"
            cv2.putText(self.img, text, (mid[0] + 6, mid[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 3, cv2.LINE_AA)
            cv2.putText(self.img, text, (mid[0] + 6, mid[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        # Preview line while dragging
        if preview_line is not None:
            a, b = preview_line
            cv2.line(self.img, a, b, (0, 255, 255), 2)

            try:
                lm = self.measure_line_m(a, b)
                text = f"{lm.length_m:.4f} m ({lm.mode})"
                mid = ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)
                cv2.putText(self.img, text, (mid[0] + 6, mid[1] - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(self.img, text, (mid[0] + 6, mid[1] - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
            except Exception:
                pass

    def on_mouse(self, event, x, y, flags, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.p0 = (x, y)
            self.p1 = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.p1 = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.dragging:
            self.dragging = False
            self.p1 = (x, y)
            try:
                lm = self.measure_line_m(self.p0, self.p1)
                self.lines.append(lm)
            except Exception as e:
                print("Measurement failed:", e)

    def run(self, win_name: str = "AprilTag Ruler") -> Results:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(win_name, self.on_mouse)

        while True:
            preview = (self.p0, self.p1) if self.dragging else None
            self.redraw(preview_line=preview)
            cv2.imshow(win_name, self.img)

            key = cv2.waitKey(15) & 0xFF
            if key in (27, ord('q')):  # ESC or q
                break
            if key == ord('r'):
                self.lines.clear()
            if key == ord('s'):
                # handled outside, but keep key reserved
                return self.results()

        cv2.destroyWindow(win_name)
        return self.results()

    def results(self) -> Results:
        return Results(
            image="",
            tag_id=int(self.tag_det["id"]),
            tag_family=self.tag_family,
            tag_size_m=self.tag_size_m,
            px_per_meter_quick=float(self.px_per_meter_quick) if self.px_per_meter_quick is not None else None,
            pose_available=self.pose_available,
            lines=self.lines.copy()
        )


# -------------------------
# CLI
# -------------------------

def load_intrinsics_json(path: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Accepts a simple JSON with:
      { "fx":..., "fy":..., "cx":..., "cy":..., "dist":[k1,k2,p1,p2,k3] }
    """
    data = json.loads(path.read_text())
    fx, fy, cx, cy = float(data["fx"]), float(data["fy"]), float(data["cx"]), float(data["cy"])
    K = build_camera_matrix(fx, fy, cx, cy)

    dist = None
    if "dist" in data and data["dist"] is not None:
        arr = np.array(data["dist"], dtype=np.float64).reshape(-1, 1)
        dist = arr
    return K, dist


def main() -> None:
    ap = argparse.ArgumentParser(description="AprilTag-based interactive ruler.")
    ap.add_argument("image", type=str, help="Path to image.jpg")
    ap.add_argument("--tag-size-m", type=float, required=True, help="Physical AprilTag size (outer black square) in meters")
    ap.add_argument("--family", type=str, default="tag25h9", help="AprilTag family (e.g. tag36h11)")

    # Intrinsics (optional)
    ap.add_argument("--intrinsics", type=str, default="", help="Path to intrinsics JSON (fx,fy,cx,cy, optional dist)")
    ap.add_argument("--fx", type=float, default=0.0)
    ap.add_argument("--fy", type=float, default=0.0)
    ap.add_argument("--cx", type=float, default=0.0)
    ap.add_argument("--cy", type=float, default=0.0)

    # Save outputs
    ap.add_argument("--outdir", type=str, default=".", help="Output directory")
    ap.add_argument("--name", type=str, default="apriltag_ruler", help="Basename for outputs")

    args = ap.parse_args()

    img_path = Path(args.image)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"Failed to read image: {img_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    dets = detect_apriltags(gray, args.family)
    if not dets:
        raise SystemExit("No AprilTags detected. Try better lighting / sharper image / correct family.")

    # Choose the largest detected tag (most reliable)
    dets_sorted = sorted(dets, key=lambda d: avg_tag_edge_px(d["corners"]), reverse=True)
    tag = dets_sorted[0]

    # Load intrinsics (optional)
    K = None
    dist = None
    if args.intrinsics:
        K, dist = load_intrinsics_json(Path(args.intrinsics))
    elif args.fx > 0 and args.fy > 0:
        K = build_camera_matrix(args.fx, args.fy, args.cx, args.cy)
        dist = np.zeros((5, 1), dtype=np.float64)

    ruler = InteractiveRuler(img, tag, args.family, args.tag_size_m, K, dist)
    res = ruler.run()

    # If user pressed 's' we returned early; either way we save at end for convenience
    res.image = str(img_path)

    annotated_path = outdir / f"{args.name}_annotated.jpg"
    json_path = outdir / f"{args.name}_results.json"

    cv2.imwrite(str(annotated_path), ruler.img)
    json_path.write_text(json.dumps(asdict(res), indent=2))

    print("Saved:", annotated_path)
    print("Saved:", json_path)


if __name__ == "__main__":
    main()