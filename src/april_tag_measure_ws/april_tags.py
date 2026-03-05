#!/usr/bin/env python3
"""
april_tags.py

AprilTag detection utilities.

Backends:
- pupil_apriltags (preferred if installed)
- OpenCV aruco AprilTag dictionaries fallback

Exports:
- detect_apriltags(gray, family) -> list[dict{id,corners,center,backend}]
- avg_tag_edge_px(corners) -> float

Date: 5/03/2026
Version: 1.0
Maintainer: Victor Lim - victor@polymaya.tech
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import cv2


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


def detect_apriltags(gray: np.ndarray, family: str) -> List[Dict[str, Any]]:
    """
    Returns list of detections:
      - id (int)
      - corners: (4,2) float32 pixels
      - center: (2,) float32
      - backend: "pupil_apriltags" or "opencv_aruco"
    """
    # 1) Try pupil_apriltags
    try:
        from pupil_apriltags import Detector  # type: ignore

        det = Detector(families=family)
        dets = det.detect(gray)

        out: List[Dict[str, Any]] = []
        for d in dets:
            corners = np.array(d.corners, dtype=np.float32)
            center = np.array(d.center, dtype=np.float32)
            out.append(
                {
                    "id": int(getattr(d, "tag_id", -1)),
                    "corners": corners,
                    "center": center,
                    "backend": "pupil_apriltags",
                }
            )
        return out
    except Exception:
        pass

    # 2) Fallback: OpenCV aruco AprilTag dictionaries
    try:
        aruco = cv2.aruco  # type: ignore

        fam_map = {
            "tag16h5": getattr(aruco, "DICT_APRILTAG_16h5", None),
            "tag25h9": getattr(aruco, "DICT_APRILTAG_25h9", None),
            "tag36h10": getattr(aruco, "DICT_APRILTAG_36h10", None),
            "tag36h11": getattr(aruco, "DICT_APRILTAG_36h11", None),
        }
        dict_id = fam_map.get(family, fam_map.get("tag36h11"))
        if dict_id is None:
            raise RuntimeError("OpenCV build missing AprilTag dictionaries")

        dictionary = aruco.getPredefinedDictionary(dict_id)
        params = aruco.DetectorParameters()

        if hasattr(aruco, "ArucoDetector"):
            detector = aruco.ArucoDetector(dictionary, params)
            corners_list, ids, _ = detector.detectMarkers(gray)
        else:
            corners_list, ids, _ = aruco.detectMarkers(gray, dictionary, parameters=params)

        out: List[Dict[str, Any]] = []
        if ids is None or len(ids) == 0:
            return out

        for corners, tag_id in zip(corners_list, ids.flatten()):
            c = np.array(corners, dtype=np.float32).reshape(-1, 2)
            center = c.mean(axis=0)
            out.append(
                {
                    "id": int(tag_id),
                    "corners": c,
                    "center": center.astype(np.float32),
                    "backend": "opencv_aruco",
                }
            )
        return out
    except Exception:
        return []