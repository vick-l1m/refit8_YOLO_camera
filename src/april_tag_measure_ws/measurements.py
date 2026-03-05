#!/usr/bin/env python3
"""
measurements.py

Measurement + geometry utilities:
- quick pixel-to-meter scaling from tag size
- tag pose estimation (solvePnP)
- ray-plane intersection to measure on tag plane

Exports:
- build_camera_matrix
- load_intrinsics_json
- compute_px_per_meter_from_tag
- solve_tag_pose
- intersect_ray_with_tag_plane

Date: 5/03/2026
Version: 1.0
Maintainer: Victor Lim - victor@polymaya.tech
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import cv2


def build_camera_matrix(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    return np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0,  0,  1]], dtype=np.float64)


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
        dist = np.array(data["dist"], dtype=np.float64).reshape(-1, 1)
    return K, dist


def compute_px_per_meter_from_tag(tag_corners_px: np.ndarray, tag_size_m: float) -> float:
    """Quick scale: average edge px / tag_size_m."""
    c = tag_corners_px.reshape(4, 2)
    edges = [
        np.linalg.norm(c[1] - c[0]),
        np.linalg.norm(c[2] - c[1]),
        np.linalg.norm(c[3] - c[2]),
        np.linalg.norm(c[0] - c[3]),
    ]
    avg_edge = float(np.mean(edges))
    if tag_size_m <= 0:
        raise ValueError("tag_size_m must be > 0")
    return avg_edge / tag_size_m


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
    Intersect pixel ray with tag plane (Z=0 in tag frame).

    Plane in camera frame:
      n = R[:,2]
      p0 = t

    Ray:
      d = inv(K)[u v 1]^T
      X = s*d
    """
    if dist is not None and np.linalg.norm(dist) > 0:
        pts = np.array([[[u, v]]], dtype=np.float64)
        und = cv2.undistortPoints(pts, K, dist, P=K)
        u, v = float(und[0, 0, 0]), float(und[0, 0, 1])

    d = (np.linalg.inv(K) @ np.array([[u], [v], [1.0]], dtype=np.float64)).reshape(3, 1)

    n = R[:, 2].reshape(3, 1)
    p0 = t

    denom = float(n.T @ d)
    if abs(denom) < 1e-9:
        raise RuntimeError("Ray is parallel to tag plane")

    s = float(n.T @ p0) / denom
    Xc = s * d  # (3,1) in camera frame

    # Convert to tag frame
    Xo = R.T @ (Xc - t)
    return Xo.reshape(3)