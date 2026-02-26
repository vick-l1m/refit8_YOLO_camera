#!/usr/bin/env python3
"""
camera_capture.py

Shared camera capture helpers for Raspberry Pi + Arducam.

Backends:
- rpicam-still  (recommended for Arducam on Pi
- picamera2     (if installed)
- opencv        (USB/UVC fallback)

Exports:
- RPICamStillConfig
- CaptureConfig
- CameraCapture
"""

from __future__ import annotations

import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


# ----------------------------
# rpicam-still backend config
# ----------------------------
@dataclass
class RPICamStillConfig:
    width: int = 1920
    height: int = 1080
    time_ms: int = 3000

    # If preview=False, we pass -n to rpicam-still (no preview window).
    preview: bool = False

    # Autofocus args via your existing helper
    autofocus: bool = True
    af_mode: str = "continuous"  # auto | continuous | manual


# ----------------------------
# Unified capture config
# ----------------------------
@dataclass
class CaptureConfig:
    backend: str = "auto"  # auto | rpicam-still | picamera2 | opencv

    # Shared resolution preference (used by picamera2/opencv; rpicam has its own too)
    width: int = 1920
    height: int = 1080
    warmup_s: float = 0.2

    # Picamera2 best-effort autofocus
    af_mode: Optional[int] = 2
    af_trigger: Optional[int] = None

    # OpenCV fallback
    device_index: int = 0

    # rpicam-still config
    rpicam: RPICamStillConfig = field(default_factory=RPICamStillConfig)


class CameraCapture:
    """
    Camera abstraction.

    Methods:
    - capture_to_file(path): capture a still image to disk (rpicam-still backend only)
    - capture_rgb(): capture and return RGB ndarray (all backends)
    """

    def __init__(self, cfg: CaptureConfig):
        self.cfg = cfg
        self._backend: Optional[str] = None
        self._picam2 = None
        self._cap = None

    # ----------------------------
    # Backend selection / lifecycle
    # ----------------------------
    def start(self) -> None:
        backend = self.cfg.backend

        if backend == "rpicam-still":
            self._backend = "rpicam-still"
            return

        if backend in ("auto", "picamera2"):
            # Try Picamera2 first
            try:
                from picamera2 import Picamera2  # type: ignore
                self._backend = "picamera2"
                self._picam2 = Picamera2()

                config = self._picam2.create_preview_configuration(
                    main={"format": "RGB888", "size": (self.cfg.width, self.cfg.height)}
                )
                self._picam2.configure(config)
                self._picam2.start()
                time.sleep(self.cfg.warmup_s)

                self._apply_autofocus_picamera2()
                return
            except Exception:
                if backend == "picamera2":
                    raise
                # else fall through to opencv

        # OpenCV fallback
        self._backend = "opencv"
        import cv2  # local import
        self._cap = cv2.VideoCapture(self.cfg.device_index)
        if not self._cap.isOpened():
            raise RuntimeError(f"OpenCV VideoCapture failed to open device index {self.cfg.device_index}")

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
        time.sleep(self.cfg.warmup_s)

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

        self._backend = None

    def __enter__(self) -> "CameraCapture":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    # ----------------------------
    # Picamera2 autofocus
    # ----------------------------
    def _apply_autofocus_picamera2(self) -> None:
        if self._picam2 is None:
            return

        try:
            controls = self._picam2.camera_controls
        except Exception:
            controls = {}

        if self.cfg.af_mode is not None and "AfMode" in controls:
            try:
                self._picam2.set_controls({"AfMode": int(self.cfg.af_mode)})
            except Exception:
                pass

        if self.cfg.af_trigger is not None and "AfTrigger" in controls:
            try:
                self._picam2.set_controls({"AfTrigger": int(self.cfg.af_trigger)})
            except Exception:
                pass

    # ----------------------------
    # rpicam-still capture helpers
    # ----------------------------
    def _rpicam_autofocus_args(self) -> list[str]:
        if not self.cfg.rpicam.autofocus:
            return []
        from Arducam_ws.camera_focus import rpicam_autofocus_args  # your helper
        return rpicam_autofocus_args(enabled=True, mode=self.cfg.rpicam.af_mode)

    def capture_to_file(self, out_path: Path) -> None:
        """
        Capture still image to file.
        Only supported for rpicam-still backend.
        """
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if self.cfg.backend not in ("rpicam-still", "auto"):
            raise RuntimeError("capture_to_file() is intended for rpicam-still backend.")

        c = self.cfg.rpicam
        cmd = [
            "rpicam-still",
            "-t", str(c.time_ms),
            "-o", str(out_path),
            "--width", str(c.width),
            "--height", str(c.height),
        ]

        if not c.preview:
            cmd.insert(1, "-n")

        cmd += self._rpicam_autofocus_args()
        subprocess.run(cmd, check=True)

    # ----------------------------
    # Unified capture_rgb()
    # ----------------------------
    def capture_rgb(self) -> np.ndarray:
        """
        Capture and return RGB ndarray.
        - If backend is rpicam-still: capture via temp file.
        - If picamera2/opencv: capture from stream.
        """
        if self.cfg.backend == "rpicam-still":
            # Use temp file approach
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tf:
                tmp_path = Path(tf.name)
                self.capture_to_file(tmp_path)

                import cv2
                bgr = cv2.imread(str(tmp_path))
                if bgr is None:
                    raise RuntimeError("Failed to read temp capture")
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                return rgb.astype(np.uint8)

        # else: use streaming backends
        if self._backend is None:
            # allow single-shot without context manager
            self.start()

        if self._backend == "picamera2":
            assert self._picam2 is not None
            frame = self._picam2.capture_array()
            if frame.ndim == 3 and frame.shape[2] == 3:
                return frame.astype(np.uint8)
            raise RuntimeError(f"Unexpected Picamera2 frame shape: {frame.shape}")

        # OpenCV fallback returns BGR
        import cv2
        assert self._cap is not None
        ok, bgr = self._cap.read()
        if not ok or bgr is None:
            raise RuntimeError("OpenCV capture failed")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb.astype(np.uint8)