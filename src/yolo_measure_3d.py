#!/usr/bin/env python3
"""
yolo_measure_3d.py

New module:
1) Capture a still image (reuse camera_capture.py, best-effort autofocus)
2) Detect objects with YOLO
3) Measure width/height in meters using known distance + intrinsics
4) Estimate depth using angle + tunable ratio
5) Draw a projected 3D bounding box
6) Save annotated image + JSON results

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
    p.add_argument("--class_name", default="chair", help="Target class label (e.g., chair). Use 'any' for all.")
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
    return p.parse_args()


def main():
    args = parse_args()

    intr = load_intrinsics(Path(args.intrinsics))

    # Distance provider
    dist_provider = build_distance_provider(args.distance_source, args.distance_m, args.distance_file)

    # Output dirs
    outdir = Path(args.outdir)
    img_dir = outdir / "images"
    js_dir = outdir / "data"
    img_dir.mkdir(parents=True, exist_ok=True)
    js_dir.mkdir(parents=True, exist_ok=True)

    # Name
    if args.name.strip():
        base = args.name.strip()
    else:
        base = time.strftime("%Y%m%d_%H%M%S")

    # Capture config
    af_trigger = None if args.af_trigger < 0 else int(args.af_trigger)
    cap_cfg = CaptureConfig(
        width=args.width,
        height=args.height,
        device_index=args.device_index,
        af_mode=int(args.af_mode) if args.af_mode is not None else None,
        af_trigger=af_trigger,
    )

    model = YOLO(args.model)

    with CameraCapture(cap_cfg) as cam:
        rgb = capture_best_of_n(cam, n=6, sleep_s=0.05)

    rgb_yolo = rgb
    rgb_meas = undistort_rgb(rgb, intr) if args.undistort else rgb

    # Run YOLO
    results = model.predict(
        source=rgb,
        conf=args.conf,
        iou=args.iou,
        verbose=False,
        imgsz=args.imgsz,
    )
    r = results[0]

    # Prepare outputs
    annotated = rgb.copy()
    detections_out: List[Dict[str, Any]] = []

    # Distance for this capture (meters)
    Z = float(dist_provider.get_distance_m())

    # Class names mapping
    names = r.names  # dict: class_id -> name

    # Extract boxes
    if r.boxes is None or len(r.boxes) == 0:
        # Save empty output
        img_path = img_dir / f"{base}_3dbox.png"
        js_path = js_dir / f"{base}.json"
        cv2.imwrite(str(img_path), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

        payload = {
            "timestamp": base,
            "image": str(img_path),
            "distance_m": Z,
            "intrinsics": str(Path(args.intrinsics)),
            "angle_deg": args.angle_deg,
            "depth_ratio": args.depth_ratio,
            "detections": []
        }
        js_path.write_text(json.dumps(payload, indent=2))
        print(f"No detections. Saved: {img_path} and {js_path}")
        return

    # Convert tensors -> numpy safely
    boxes_xyxy = r.boxes.xyxy.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()
    clss = r.boxes.cls.cpu().numpy().astype(int)

    # Filter + limit
    chosen = []
    for i in range(len(boxes_xyxy)):
        cls_id = clss[i]
        label = names.get(cls_id, str(cls_id))
        if args.class_name != "any" and label != args.class_name:
            continue
        chosen.append(i)

    # Sort by confidence desc
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

        # Metric width/height
        W, H = bbox_to_metric_dims(bw_px, bh_px, Z, intr)

        # Depth estimate from angle
        D = estimate_depth_from_angle(W, args.angle_deg, args.depth_ratio, args.depth_min)

        # Center in camera coordinates (approx)
        Xc, Yc, Zc = pixel_center_to_cam_xyz(uc, vc, Z, intr)

        # Build cuboid in object frame, rotate by yaw, place at center.
        corners_obj = make_cuboid_corners(W, H, D)
        yaw_rad = math.radians(args.angle_deg)  # reuse the same "angle" as yaw for drawing
        corners_rot = rotate_yaw_y(corners_obj, yaw_rad)

        corners_cam = corners_rot + np.array([Xc, Yc, Zc], dtype=np.float64)

        # Draw 3D box + also draw 2D bbox for debugging
        annotated = draw_3d_box(annotated, intr, corners_cam, thickness=2)
        cv2.rectangle(
            annotated,
            (int(round(x1)), int(round(y1))),
            (int(round(x2)), int(round(y2))),
            (255, 0, 0),
            2
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

        detections_out.append({
            "class": label,
            "class_id": cls_id,
            "confidence": conf,
            "bbox_xyxy_px": [float(x1), float(y1), float(x2), float(y2)],
            "bbox_wh_px": [bw_px, bh_px],
            "distance_m": Z,
            "dimensions_m": {
                "width": W,
                "height": H,
                "depth": D
            },
            "assumptions": {
                "width_height_from_pinhole": True,
                "depth_estimated_from_angle": True,
                "angle_deg": float(args.angle_deg),
                "depth_ratio": float(args.depth_ratio),
                "depth_min_m": float(args.depth_min),
                "distance_source": args.distance_source
            }
        })

    # Save outputs
    img_path = img_dir / f"{base}_3dbox.png"
    js_path = js_dir / f"{base}.json"
    cv2.imwrite(str(img_path), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

    payload = {
        "timestamp": base,
        "image": str(img_path),
        "distance_m": Z,
        "intrinsics": str(Path(args.intrinsics)),
        "undistort": bool(args.undistort),
        "model": str(args.model),
        "conf": float(args.conf),
        "iou": float(args.iou),
        "angle_deg": float(args.angle_deg),
        "depth_ratio": float(args.depth_ratio),
        "detections": detections_out
    }
    js_path.write_text(json.dumps(payload, indent=2))

    print(f"Saved annotated image: {img_path}")
    print(f"Saved JSON: {js_path}")

    if args.preview:
        cv2.imshow("YOLO 3D Measure", cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()