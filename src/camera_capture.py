#!/usr/bin/env python3
"""
camera_capture.py

Shared camera capture helpers for Raspberry Pi + Arducam.

- Prefers Picamera2 (libcamera) if installed.
- Falls back to cv2.VideoCapture for USB/UVC devices.
- Provides best-effort autofocus setup for Picamera2.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class CaptureConfig:
    width: int = 1920
    height: int = 1080
    warmup_s: float = 0.2

    # Autofocus (Picamera2 best-effort)
    af_mode: Optional[int] = 2  # commonly: 0=manual, 1=auto, 2=continuous (varies by driver)
    af_trigger: Optional[int] = None  # some stacks support AfTrigger=0/1

    # If using OpenCV fallback:
    device_index: int = 0


class CameraCapture:
    """Camera abstraction that returns RGB frames as numpy arrays (H,W,3) uint8."""

    def __init__(self, cfg: CaptureConfig):
        self.cfg = cfg
        self._backend = None
        self._picam2 = None
        self._cap = None

    def start(self) -> None:
        # Try Picamera2 first
        try:
            from picamera2 import Picamera2  # type: ignore
            self._backend = "picamera2"
            self._picam2 = Picamera2()

            # Configure for still-like capture at requested resolution
            config = self._picam2.create_preview_configuration(
                main={"format": "RGB888", "size": (self.cfg.width, self.cfg.height)}
            )
            self._picam2.configure(config)
            self._picam2.start()
            time.sleep(self.cfg.warmup_s)

            self._apply_autofocus_picamera2()

            return
        except Exception:
            # Fallback to OpenCV capture
            self._backend = "opencv"

        import cv2  # local import to avoid dependency if not used
        self._cap = cv2.VideoCapture(self.cfg.device_index)
        if not self._cap.isOpened():
            raise RuntimeError(f"OpenCV VideoCapture failed to open device index {self.cfg.device_index}")

        # Try to set resolution (may not work on all devices)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
        time.sleep(self.cfg.warmup_s)

    def _apply_autofocus_picamera2(self) -> None:
        """Best-effort autofocus for Picamera2/libcamera."""
        if self._picam2 is None:
            return

        try:
            controls = self._picam2.camera_controls
        except Exception:
            controls = {}

        # AfMode
        if self.cfg.af_mode is not None and "AfMode" in controls:
            try:
                self._picam2.set_controls({"AfMode": int(self.cfg.af_mode)})
            except Exception:
                pass

        # AfTrigger (optional)
        if self.cfg.af_trigger is not None and "AfTrigger" in controls:
            try:
                self._picam2.set_controls({"AfTrigger": int(self.cfg.af_trigger)})
            except Exception:
                pass

    def capture_rgb(self) -> np.ndarray:
        if self._backend == "picamera2":
            assert self._picam2 is not None
            frame = self._picam2.capture_array()
            # Picamera2 with RGB888 should already be RGB
            if frame.ndim == 3 and frame.shape[2] == 3:
                return frame.astype(np.uint8)
            raise RuntimeError(f"Unexpected Picamera2 frame shape: {frame.shape}")

        # OpenCV fallback returns BGR -> convert to RGB
        import cv2
        assert self._cap is not None
        ok, bgr = self._cap.read()
        if not ok or bgr is None:
            raise RuntimeError("OpenCV capture failed")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb.astype(np.uint8)

    def stop(self) -> None:
        if self._backend == "picamera2" and self._picam2 is not None:
            try:
                self._picam2.stop()
            except Exception:
                pass
            self._picam2 = None

        if self._backend == "opencv" and self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    def __enter__(self) -> "CameraCapture":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()