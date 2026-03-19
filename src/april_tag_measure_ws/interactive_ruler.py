#!/usr/bin/env python3
"""
interactive_ruler.py


Interactive UI to draw measurement lines and compute lengths in meters.


Depends on:
- measurement_session.py for reusable session logic
- measurements.py for pose/ray-plane math


Date: 19/03/2026
Version: 1.1
Maintainer: Victor Lim - victor@polymaya.tech
"""


from __future__ import annotations


from typing import Any, Dict, Optional


import cv2
import numpy as np

from src.april_tag_measure_ws.measurement_session import LineMeasurement, MeasurementSession, Results


class InteractiveRuler:
    def __init__(
        self,
        img_bgr: np.ndarray,
        tag_det: Dict[str, Any],
        tag_family: str,
        tag_size_m: float,
        K: Optional[np.ndarray],
        dist: Optional[np.ndarray],
    ):
        self.session = MeasurementSession(
            img_bgr=img_bgr,
            tag_det=tag_det,
            tag_family=tag_family,
            tag_size_m=tag_size_m,
            K=K,
            dist=dist,
        )

        # Backward-compatible attributes
        self.base = self.session.base
        self.img = self.session.img
        self.tag_det = self.session.tag_det
        self.tag_family = self.session.tag_family
        self.tag_size_m = self.session.tag_size_m
        self.backend = self.session.backend
        self.K = self.session.K
        self.dist = self.session.dist
        self.pose_available = self.session.pose_available
        self.R = self.session.R
        self.t = self.session.t
        self.px_per_meter_quick = self.session.px_per_meter_quick
        self.lines = self.session.lines

        self.dragging = False
        self.p0 = (0, 0)
        self.p1 = (0, 0)

    def measure_line_m(self, p0, p1) -> LineMeasurement:
        return self.session.measure_line_m(p0, p1)

    def redraw(self, preview_line=None) -> None:
        self.img = self.session.render(preview_line=preview_line)

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
                self.session.add_line(self.p0, self.p1)
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
            if key in (27, ord("q")):
                break
            if key == ord("r"):
                self.session.clear_lines()
            if key == ord("s"):
                cv2.destroyWindow(win_name)
                return self.results(image_path="")

        cv2.destroyWindow(win_name)
        return self.results(image_path="")

    def results(self, image_path: str) -> Results:
        return self.session.results(image_path=image_path)