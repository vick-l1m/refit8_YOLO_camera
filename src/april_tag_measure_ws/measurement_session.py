#!/usr/bin/env python3
"""
measurement_session.py


Reusable AprilTag measurement session logic.


Purpose:
- Hold the image, tag detection, and measured lines
- Compute metric line lengths from image points
- Render annotated output images
- Export structured results for app or CLI use


Date: 19/03/2026
Version: 1.0
Maintainer: Victor Lim - victor@polymaya.tech
"""


from __future__ import annotations


import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


import cv2
import numpy as np


from src.april_tag_measure_ws.measurements import (
    compute_px_per_meter_from_tag,
    intersect_ray_with_tag_plane,
    solve_tag_pose,
)


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
    backend: str
    lines: List[LineMeasurement]


class MeasurementSession:
    def __init__(
        self,
        img_bgr: np.ndarray,
        tag_det: Dict[str, Any],
        tag_family: str,
        tag_size_m: float,
        K: Optional[np.ndarray],
        dist: Optional[np.ndarray],
    ) -> None:
        self.base = img_bgr.copy()
        self.img = img_bgr.copy()

        self.tag_det = tag_det
        self.tag_family = tag_family
        self.tag_size_m = tag_size_m
        self.backend = str(tag_det.get("backend", "unknown"))

        self.K = K
        self.dist = dist

        self.pose_available = False
        self.R: Optional[np.ndarray] = None
        self.t: Optional[np.ndarray] = None

        corners = self.tag_det["corners"]
        self.px_per_meter_quick: Optional[float] = None
        if tag_size_m > 0:
            self.px_per_meter_quick = compute_px_per_meter_from_tag(corners, tag_size_m)

        if self.K is not None:
            try:
                R, t = solve_tag_pose(corners, tag_size_m, self.K, self.dist)
                self.R, self.t = R, t
                self.pose_available = True
            except Exception:
                self.pose_available = False

        self.lines: List[LineMeasurement] = []

    def clear_lines(self) -> None:
        self.lines.clear()

    def add_line(self, p0: Tuple[int, int], p1: Tuple[int, int]) -> LineMeasurement:
        lm = self.measure_line_m(p0, p1)
        self.lines.append(lm)
        return lm

    def measure_line_m(
        self,
        p0: Tuple[int, int],
        p1: Tuple[int, int],
    ) -> LineMeasurement:
        x0, y0 = p0
        x1, y1 = p1
        pix_len = math.hypot(x1 - x0, y1 - y0)

        if (
            self.pose_available
            and self.K is not None
            and self.R is not None
            and self.t is not None
        ):
            X0 = intersect_ray_with_tag_plane(x0, y0, self.K, self.dist, self.R, self.t)
            X1 = intersect_ray_with_tag_plane(x1, y1, self.K, self.dist, self.R, self.t)
            length_m = float(np.linalg.norm(X1[:2] - X0[:2]))
            return LineMeasurement(p0=p0, p1=p1, length_m=length_m, mode="pose")

        if self.px_per_meter_quick is None or self.px_per_meter_quick <= 0:
            raise RuntimeError("No valid scale for quick mode")

        length_m = pix_len / self.px_per_meter_quick
        return LineMeasurement(p0=p0, p1=p1, length_m=length_m, mode="quick")

    def render(
        self,
        preview_line: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None,
    ) -> np.ndarray:
        img = self.base.copy()

        corners = self.tag_det["corners"].astype(int)
        cv2.polylines(img, [corners.reshape(-1, 1, 2)], True, (0, 255, 0), 2)
        c = tuple(self.tag_det["center"].astype(int))
        cv2.circle(img, c, 4, (0, 255, 0), -1)

        mode = "pose" if self.pose_available else "quick"
        status = (
            f"Tag {self.tag_det['id']} | backend={self.backend} | "
            f"mode={mode} | tag={self.tag_size_m:.4f} m"
        )
        cv2.putText(
            img,
            status,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (20, 20, 20),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            status,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )

        if not self.pose_available and self.px_per_meter_quick is not None:
            s2 = (
                f"quick scale: {self.px_per_meter_quick:.1f} px/m "
                f"(best when tag faces camera)"
            )
            cv2.putText(
                img,
                s2,
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (20, 20, 20),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                img,
                s2,
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (240, 240, 240),
                1,
                cv2.LINE_AA,
            )

        for i, lm in enumerate(self.lines, start=1):
            cv2.line(img, lm.p0, lm.p1, (255, 0, 255), 2)
            mid = ((lm.p0[0] + lm.p1[0]) // 2, (lm.p0[1] + lm.p1[1]) // 2)
            text = f"{i}: {lm.length_m:.4f} m ({lm.mode})"
            cv2.putText(
                img,
                text,
                (mid[0] + 6, mid[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (20, 20, 20),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                img,
                text,
                (mid[0] + 6, mid[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        if preview_line is not None:
            a, b = preview_line
            cv2.line(img, a, b, (0, 255, 255), 2)
            try:
                lm = self.measure_line_m(a, b)
                text = f"{lm.length_m:.4f} m ({lm.mode})"
                mid = ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)
                cv2.putText(
                    img,
                    text,
                    (mid[0] + 6, mid[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 0),
                    3,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    img,
                    text,
                    (mid[0] + 6, mid[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
            except Exception:
                pass

        self.img = img
        return img

    def results(self, image_path: str) -> Results:
        return Results(
            image=image_path,
            tag_id=int(self.tag_det["id"]),
            tag_family=self.tag_family,
            tag_size_m=self.tag_size_m,
            px_per_meter_quick=(
                float(self.px_per_meter_quick)
                if self.px_per_meter_quick is not None
                else None
            ),
            pose_available=self.pose_available,
            backend=self.backend,
            lines=self.lines.copy(),
        )