#!/usr/bin/env python3
"""
yolo_measure_3d.py
1) Capture a still image (reuse camera_capture.py, best-effort autofocus)
2) Detect objects with YOLO
3) Measure width/height in meters using:
    3a) A distance input (e.g. from a sensor or user prompt) OR
    3b) An AprilTag in the scene with known size (best effort, optional)
4) Draw a projected 3D bounding box
5) Save annotated image + JSON results

Notes:
- W/H are metric (after calibration) given correct distance to object plane.
- Depth is an estimate; improves later with a reference marker or multi-view.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import cv2
from ultralytics import YOLO

from camera_capture import CameraCapture, CaptureConfig


# ----------------------------
# Distance provider (pluggable)
# ----------------------------

class DistanceProvider:
    def get_distance_m(self) -> float:
        raise NotImplementedError


class ConstantDistance(DistanceProvider):
    def __init__(self, distance_m: float):
        self.distance_m = float(distance_m)

    def get_distance_m(self) -> float:
        return self.distance_m


class FileDistance(DistanceProvider):
    """
    Reads a float (meters) from a text file each time get_distance_m() is called.
    Useful as a simple bridge for external sensor scripts.
    """
    def __init__(self, path: Path):
        self.path = Path(path)

    def get_distance_m(self) -> float:
        txt = self.path.read_text().strip()
        return float(txt)


def build_distance_provider(distance_source: str, distance_m: float, distance_file: Optional[str]) -> DistanceProvider:
    """
    distance_source:
      - "constant" (default)
      - "file" (reads meters from --distance_file)
    """
    if distance_source == "constant":
        return ConstantDistance(distance_m)

    if distance_source == "file":
        if not distance_file:
            raise ValueError("--distance_file is required when --distance_source=file")
        return FileDistance(Path(distance_file))

    raise ValueError(f"Unsupported distance_source: {distance_source}")

# ----------------------------
# Intrinsics
# ----------------------------

@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    dist: np.ndarray  # distortion coefficients (k1,k2,p1,p2,k3,...)
    width: int
    height: int


def load_intrinsics(path: Path) -> Intrinsics:
    data = json.loads(Path(path).read_text())
    K = np.array(data["camera_matrix"], dtype=np.float64)
    dist = np.array(data["dist_coeffs"], dtype=np.float64).reshape(-1)
    w = int(data["image_width"])
    h = int(data["image_height"])
    return Intrinsics(
        fx=float(K[0, 0]),
        fy=float(K[1, 1]),
        cx=float(K[0, 2]),
        cy=float(K[1, 2]),
        dist=dist,
        width=w,
        height=h
    )


def undistort_rgb(rgb: np.ndarray, intr: Intrinsics) -> np.ndarray:
    K = np.array([[intr.fx, 0, intr.cx],
                  [0, intr.fy, intr.cy],
                  [0, 0, 1]], dtype=np.float64)
    dist = intr.dist.astype(np.float64)
    return cv2.undistort(rgb, K, dist)

# ----------------------------
# Apriltag-basted distance measurements
# ----------------------------
def apriltag_distance_z_m(rgb: np.ndarray, intr: Intrinsics, tag_size_m: float,
                          families: str = "tag36h11") -> Optional[dict]:
    """
    Returns dict with {id, z_m, tvec_m, decision_margin} or None if no tag found.
    z_m is forward distance along camera Z axis.
    """
    try:
        from pupil_apriltags import Detector
    except ImportError:
        return None

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    det = Detector(
        families=families,
        nthreads=2,
        quad_decimate=2.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0,
    )

    results = det.detect(
        gray,
        estimate_tag_pose=True,
        camera_params=(intr.fx, intr.fy, intr.cx, intr.cy),
        tag_size=float(tag_size_m),
    )

    if not results:
        return None

    best = max(results, key=lambda r: r.decision_margin)
    t = best.pose_t.reshape(3)  # meters
    z_m = float(t[2])

    return {
        "id": int(best.tag_id),
        "z_m": z_m,
        "tvec_m": [float(t[0]), float(t[1]), float(t[2])],
        "decision_margin": float(best.decision_margin),
    }

def detect_apriltags(
    rgb: np.ndarray,
    intr: Intrinsics,
    tag_size_m: float,
    families: str = "tag36h11",
) -> List[Dict[str, Any]]:
    """
    Detect AprilTags and (optionally) estimate pose.

    Returns list of dicts, each containing:
      - id
      - corners_px: 4 corners (list of [x,y]) in image pixels
      - center_px: [x,y]
      - decision_margin
      - pose_t_m: [tx,ty,tz] in meters (if pose estimation available)
    """
    try:
        from pupil_apriltags import Detector
    except ImportError:
        return []

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    det = Detector(
        families=families,
        nthreads=2,
        quad_decimate=2.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0,
    )

    results = det.detect(
        gray,
        estimate_tag_pose=True,
        camera_params=(intr.fx, intr.fy, intr.cx, intr.cy),
        tag_size=float(tag_size_m),
    )

    out: List[Dict[str, Any]] = []
    for r in results:
        # r.corners: (4,2) float px, r.center: (2,)
        corners = [[float(x), float(y)] for (x, y) in r.corners]
        center = [float(r.center[0]), float(r.center[1])]
        tvec = None
        if getattr(r, "pose_t", None) is not None:
            t = r.pose_t.reshape(3)
            tvec = [float(t[0]), float(t[1]), float(t[2])]

        out.append(
            {
                "id": int(r.tag_id),
                "corners_px": corners,
                "center_px": center,
                "decision_margin": float(r.decision_margin),
                "pose_t_m": tvec,
            }
        )

    return out

def draw_apriltag_boxes(
    rgb: np.ndarray,
    tags: List[Dict[str, Any]],
    thickness: int = 2,
) -> np.ndarray:
    img = rgb.copy()

    for t in tags:
        corners = np.array(t["corners_px"], dtype=np.float32)  # (4,2)
        pts = corners.reshape((-1, 1, 2)).astype(np.int32)
        cv2.polylines(img, [pts], isClosed=True, color=(255, 255, 0), thickness=thickness)

        # Label near first corner
        x0, y0 = int(round(corners[0, 0])), int(round(corners[0, 1]))
        cv2.putText(
            img,
            f"AprilTag {t['id']}",
            (x0, max(0, y0 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )

    return img

# ----------------------------
# 3D box math + drawing
# ----------------------------

_BOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # bottom rectangle
    (4, 5), (5, 6), (6, 7), (7, 4),  # top rectangle
    (0, 4), (1, 5), (2, 6), (3, 7)   # vertical edges
]


def project_points(pts_cam: np.ndarray, intr: Intrinsics) -> np.ndarray:
    """
    pts_cam: (N,3) in camera frame.
    returns: (N,2) pixel coordinates
    """
    X = pts_cam[:, 0]
    Y = pts_cam[:, 1]
    Z = np.clip(pts_cam[:, 2], 1e-6, None)
    u = intr.fx * (X / Z) + intr.cx
    v = intr.fy * (Y / Z) + intr.cy
    return np.stack([u, v], axis=1)


def make_cuboid_corners(W: float, H: float, D: float) -> np.ndarray:
    """
    Returns 8 corners (8,3) centered at origin in object frame.
    Convention:
      0-3 bottom (y=-H/2), 4-7 top (y=+H/2)
    """
    x = W / 2.0
    y = H / 2.0
    z = D / 2.0
    corners = np.array([
        [-x, -y, -z],
        [ x, -y, -z],
        [ x, -y,  z],
        [-x, -y,  z],
        [-x,  y, -z],
        [ x,  y, -z],
        [ x,  y,  z],
        [-x,  y,  z],
    ], dtype=np.float64)
    return corners


def rotate_yaw_y(corners: np.ndarray, yaw_rad: float) -> np.ndarray:
    c = math.cos(yaw_rad)
    s = math.sin(yaw_rad)
    R = np.array([
        [ c, 0, s],
        [ 0, 1, 0],
        [-s, 0, c],
    ], dtype=np.float64)
    return corners @ R.T


def draw_3d_box(rgb: np.ndarray, intr: Intrinsics, corners_cam: np.ndarray, thickness: int = 2) -> np.ndarray:
    img = rgb.copy()
    pts2 = project_points(corners_cam, intr)

    # Clip / skip lines with invalid projection (optional simplification)
    for a, b in _BOX_EDGES:
        p1 = pts2[a]
        p2 = pts2[b]
        cv2.line(
            img,
            (int(round(p1[0])), int(round(p1[1]))),
            (int(round(p2[0])), int(round(p2[1]))),
            (0, 255, 0),  # green lines (change if you want)
            thickness
        )
    return img


# ----------------------------
# Measurement logic
# ----------------------------

def bbox_to_metric_dims(bw_px: float, bh_px: float, Z: float, intr: Intrinsics) -> Tuple[float, float]:
    """
    Convert bbox pixel width/height to meters using a pinhole model.
    Assumes object is approximately fronto-parallel at distance Z.
    """
    W = (bw_px * Z) / intr.fx
    H = (bh_px * Z) / intr.fy
    return float(W), float(H)


def estimate_depth_from_angle(W: float, angle_deg: float, depth_ratio: float, depth_min: float) -> float:
    """
    Heuristic depth estimate:
      D = W * depth_ratio * tan(angle)

    - angle=45° => D ≈ W * depth_ratio
    - angle small => D shrinks; clamp with depth_min

    depth_min is absolute meters clamp (e.g. 0.05).
    """
    theta = math.radians(max(0.0, min(85.0, angle_deg)))
    D = W * float(depth_ratio) * math.tan(theta)
    return float(max(D, depth_min))


def pixel_center_to_cam_xyz(u: float, v: float, Z: float, intr: Intrinsics) -> Tuple[float, float, float]:
    """
    Back-project pixel center (u,v) to camera frame at depth Z:
      X = (u-cx) * Z / fx
      Y = (v-cy) * Z / fy
      Z = Z
    """
    X = (u - intr.cx) * Z / intr.fx
    Y = (v - intr.cy) * Z / intr.fy
    return float(X), float(Y), float(Z)

# ----------------------------
# Multiple passes + sharpness selection (optional, for better autofocus on PiCamera2)
# ----------------------------

def sharpness_score(rgb: np.ndarray) -> float:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def capture_best_of_n(cam: CameraCapture, n: int = 6, sleep_s: float = 0.05) -> np.ndarray:
    """
    Capture n frames and return the sharpest one (highest Laplacian variance).
    Dropping the first frame often helps (AE/AWB settling).
    """
    frames: list[np.ndarray] = []
    for _ in range(n):
        frames.append(cam.capture_rgb())
        time.sleep(sleep_s)

    if len(frames) >= 2:
        frames = frames[1:]  # drop first

    return max(frames, key=sharpness_score)

# ----------------------------
# Main
# ----------------------------

def parse_args():
    p = argparse.ArgumentParser(description="YOLO + metric W/H + estimated depth + 3D box overlay")
    p.add_argument("--model", default=str(Path.home() / "models" / "yolov8n.pt"), help="Path to YOLO .pt model")
    p.add_argument("--intrinsics", required=True, help="Path to intrinsics JSON from calibrate_camera.py")

    # Capture
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--device_index", type=int, default=0, help="OpenCV fallback camera index")
    p.add_argument("--imgsz", type=int, default=320)
    
    # Autofocus (Picamera2 best-effort)
    p.add_argument("--af_mode", type=int, default=2)
    p.add_argument("--af_trigger", type=int, default=-1, help="Set >=0 to use; -1 disables")

    # YOLO
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--class_name", default="any", help="Target class label (e.g., chair). Use 'any' for all.")
    p.add_argument("--max_detections", type=int, default=5)

    # Distance input
    p.add_argument("--distance_source", default="constant", choices=["constant", "file"])
    p.add_argument("--distance_m", type=float, default=2.0, help="Used when distance_source=constant")
    p.add_argument("--distance_file", default=None, help="Used when distance_source=file; contains meters")

    # Depth estimate tuning
    p.add_argument("--angle_deg", type=float, default=45.0, help="Isometric-ish yaw angle assumption (deg)")
    p.add_argument("--depth_ratio", type=float, default=1.0, help="Extra scaling for depth estimate")
    p.add_argument("--depth_min", type=float, default=0.05, help="Clamp minimum depth in meters")

    # Undistort for measurement accuracy
    p.add_argument("--undistort", action="store_true")

    # Output
    p.add_argument("--outdir", default=str(Path.home() / "captures" / "yolo" / "measure_3d"))
    p.add_argument("--name", default="", help="Base name (no extension). If empty -> timestamp.")
    p.add_argument("--preview", action="store_true", help="Show a preview window")

    # Input image
    p.add_argument("--image", default=None, help="Path to input image. If set, skips camera capture.")

    # April tags for distance estimation (optional)
    p.add_argument("--use_apriltag", action="store_true",
               help="If set, try AprilTag; if found use it for distance/scale.")
    p.add_argument("--tag_size_m", type=float, default=0.05,
                help="AprilTag edge length in meters (e.g., 0.05 for 5cm).")
    p.add_argument("--tag_family", default="tag36h11",
                help="AprilTag family (e.g., tag36h11).")

    return p.parse_args()

def main():
    args = parse_args()


    intr = load_intrinsics(Path(args.intrinsics))


    # Distance provider (fallback if no AprilTag pose)
    dist_provider = build_distance_provider(args.distance_source, args.distance_m, args.distance_file)


    # ----------------------------
    # Output dirs (repo-root relative)
    # ----------------------------
    # Prefer placing captures under <repo>/captures/... (not scripts/captures)
    project_root = Path(__file__).resolve().parent
    # If this file lives in something like <repo>/src/, go up one
    if project_root.name in ("src", "scripts"):
        project_root = project_root.parent


    base_captures = project_root / "captures" / "yolo" / "measure_3d"
    img_dir = base_captures / "images"
    js_dir = base_captures / "data"
    img_dir.mkdir(parents=True, exist_ok=True)
    js_dir.mkdir(parents=True, exist_ok=True)


    # ----------------------------
    # Name priority: --name > --image stem > timestamp
    # ----------------------------
    if args.name.strip():
        base = args.name.strip()
    elif args.image:
        base = f"{Path(args.image).stem}_measured"
    else:
        base = time.strftime("%Y%m%d_%H%M%S")


    # ----------------------------
    # Acquire image (RGB)
    # ----------------------------
    if args.image:
        img_path_in = Path(args.image)
        if not img_path_in.exists():
            raise FileNotFoundError(f"--image not found: {img_path_in}")


        bgr_in = cv2.imread(str(img_path_in), cv2.IMREAD_COLOR)
        if bgr_in is None:
            raise RuntimeError(f"Failed to read image: {img_path_in}")


        rgb = cv2.cvtColor(bgr_in, cv2.COLOR_BGR2RGB)


    else:
        # Camera capture config
        cap_cfg = CaptureConfig(
            backend="rpicam-still",  # matches your still_image.sh behavior on Pi
            width=args.width,
            height=args.height,
            device_index=args.device_index,
        )
        cap_cfg.rpicam.width = args.width
        cap_cfg.rpicam.height = args.height
        cap_cfg.rpicam.time_ms = 2000
        cap_cfg.rpicam.preview = False
        cap_cfg.rpicam.autofocus = True
        cap_cfg.rpicam.af_mode = "continuous"


        with CameraCapture(cap_cfg) as cam:
            time.sleep(0.2)
            rgb = capture_best_of_n(cam, n=6, sleep_s=0.08)


    # Optional undistort BEFORE tag detection and YOLO so geometry is consistent
    if args.undistort:
        rgb = undistort_rgb(rgb, intr)


    # ----------------------------
    # AprilTag detection + drawing
    # ----------------------------
    apriltags: List[Dict[str, Any]] = []
    if args.use_apriltag:
        apriltags = detect_apriltags(
            rgb=rgb,
            intr=intr,
            tag_size_m=float(args.tag_size_m),
            families=str(args.tag_family),
        )
    apriltag_count = len(apriltags)


    annotated = rgb.copy()
    if apriltag_count > 0:
        annotated = draw_apriltag_boxes(annotated, apriltags, thickness=2)


    # ----------------------------
    # Distance selection (Z)
    # ----------------------------
    Z_fallback = float(dist_provider.get_distance_m())
    Z = Z_fallback
    tag_info = None


    if args.use_apriltag and apriltag_count > 0:
        with_pose = [t for t in apriltags if t.get("pose_t_m") is not None]
        if with_pose:
            best = max(with_pose, key=lambda t: float(t.get("decision_margin", 0.0)))
            # pose_t_m = [tx, ty, tz] in meters
            Z = float(best["pose_t_m"][2])
            tag_info = {
                "id": int(best["id"]),
                "z_m": Z,
                "tvec_m": best["pose_t_m"],
                "decision_margin": float(best["decision_margin"]),
            }


    distance_source_used = "apriltag" if tag_info is not None else args.distance_source


    # ----------------------------
    # YOLO inference (IMPORTANT: use BGR for best results)
    # ----------------------------
    model = YOLO(args.model)


    bgr_for_yolo = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


    results = model.predict(
        source=bgr_for_yolo,
        conf=float(args.conf),
        iou=float(args.iou),
        imgsz=int(args.imgsz),
        verbose=False,
    )
    r = results[0]


    detections_out: List[Dict[str, Any]] = []


    # If no boxes, still write outputs (with apriltag info)
    if r.boxes is None or len(r.boxes) == 0:
        out_img_path = img_dir / f"{base}.png"
        out_json_path = js_dir / f"{base}.json"


        cv2.imwrite(str(out_img_path), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))


        payload = {
            "timestamp": base,
            "image": str(out_img_path),
            "input_image": str(args.image) if args.image else None,
            "apriltag": {
                "enabled": bool(args.use_apriltag),
                "family": str(args.tag_family),
                "tag_size_m": float(args.tag_size_m),
                "count": int(apriltag_count),
                "tags": apriltags,
                "best_for_distance": tag_info,
            },
            "distance_m": float(Z),
            "distance_fallback_m": float(Z_fallback),
            "distance_source_used": distance_source_used,
            "intrinsics": str(Path(args.intrinsics)),
            "undistort": bool(args.undistort),
            "model": str(args.model),
            "conf": float(args.conf),
            "iou": float(args.iou),
            "imgsz": int(args.imgsz),
            "detections": [],
        }
        out_json_path.write_text(json.dumps(payload, indent=2))


        print("No YOLO detections.")
        print(f"Saved annotated image: {out_img_path}")
        print(f"Saved JSON: {out_json_path}")


        if args.preview:
            cv2.imshow("YOLO 3D Measure", cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return


    # ----------------------------
    # Parse YOLO boxes + draw + measure
    # ----------------------------
    boxes_xyxy = r.boxes.xyxy.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()
    clss = r.boxes.cls.cpu().numpy().astype(int)
    names = r.names  # class_id -> label


    chosen = []
    for i in range(len(boxes_xyxy)):
        cls_id = int(clss[i])
        label = names.get(cls_id, str(cls_id))
        if args.class_name != "any" and label != args.class_name:
            continue
        chosen.append(i)


    chosen.sort(key=lambda i: float(confs[i]), reverse=True)
    chosen = chosen[: max(1, int(args.max_detections))]


    for idx in chosen:
        x1, y1, x2, y2 = boxes_xyxy[idx]
        conf = float(confs[idx])
        cls_id = int(clss[idx])
        label = names.get(cls_id, str(cls_id))


        bw_px = float(max(1.0, x2 - x1))
        bh_px = float(max(1.0, y2 - y1))
        uc = float((x1 + x2) / 2.0)
        vc = float((y1 + y2) / 2.0)


        W, H = bbox_to_metric_dims(bw_px, bh_px, Z, intr)
        D = estimate_depth_from_angle(W, args.angle_deg, args.depth_ratio, args.depth_min)


        Xc, Yc, Zc = pixel_center_to_cam_xyz(uc, vc, Z, intr)


        corners_obj = make_cuboid_corners(W, H, D)
        yaw_rad = math.radians(float(args.angle_deg))
        corners_rot = rotate_yaw_y(corners_obj, yaw_rad)
        corners_cam = corners_rot + np.array([Xc, Yc, Zc], dtype=np.float64)


        annotated = draw_3d_box(annotated, intr, corners_cam, thickness=2)


        cv2.rectangle(
            annotated,
            (int(round(x1)), int(round(y1))),
            (int(round(x2)), int(round(y2))),
            (255, 0, 0),
            2,
        )
        cv2.putText(
            annotated,
            f"{label} {conf:.2f} W={W:.2f} H={H:.2f} D={D:.2f}m",
            (int(round(x1)), max(0, int(round(y1)) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


        detections_out.append(
            {
                "class": label,
                "class_id": cls_id,
                "confidence": conf,
                "bbox_xyxy_px": [float(x1), float(y1), float(x2), float(y2)],
                "bbox_wh_px": [bw_px, bh_px],
                "distance_m": float(Z),
                "dimensions_m": {"width": float(W), "height": float(H), "depth": float(D)},
                "assumptions": {
                    "width_height_from_pinhole": True,
                    "depth_estimated_from_angle": True,
                    "distance_source_used": distance_source_used,
                    "angle_deg": float(args.angle_deg),
                    "depth_ratio": float(args.depth_ratio),
                    "depth_min_m": float(args.depth_min),
                },
            }
        )


    # ----------------------------
    # Save outputs
    # ----------------------------
    out_img_path = img_dir / f"{base}.png"
    out_json_path = js_dir / f"{base}.json"


    cv2.imwrite(str(out_img_path), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))


    payload = {
        "timestamp": base,
        "image": str(out_img_path),
        "input_image": str(args.image) if args.image else None,
        "apriltag": {
            "enabled": bool(args.use_apriltag),
            "family": str(args.tag_family),
            "tag_size_m": float(args.tag_size_m),
            "count": int(apriltag_count),
            "tags": apriltags,
            "best_for_distance": tag_info,
        },
        "distance_m": float(Z),
        "distance_fallback_m": float(Z_fallback),
        "distance_source_used": distance_source_used,
        "intrinsics": str(Path(args.intrinsics)),
        "undistort": bool(args.undistort),
        "model": str(args.model),
        "conf": float(args.conf),
        "iou": float(args.iou),
        "imgsz": int(args.imgsz),
        "detections": detections_out,
    }
    out_json_path.write_text(json.dumps(payload, indent=2))


    print(f"Saved annotated image: {out_img_path}")
    print(f"Saved JSON: {out_json_path}")


    if args.preview:
        cv2.imshow("YOLO 3D Measure", cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()