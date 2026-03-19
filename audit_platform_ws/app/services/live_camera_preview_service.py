"""
live_camera_preview_service.py

Single-backend camera service using Picamera2/OpenCV via CameraCapture.

Responsibilities:
- keep one camera session open
- provide latest preview frame for the Qt UI
- save still images to disk using the same backend

Preferred backend:
- picamera2
Fallback:
- opencv
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.camera_capture import CameraCapture, CaptureConfig


class LiveCameraPreviewService:
    def __init__(self) -> None:
        self._camera: Optional[CameraCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

        self._frame_lock = threading.Lock()
        self._camera_lock = threading.Lock()

        self._latest_frame: Optional[np.ndarray] = None

    def start(self) -> None:
        if self._running:
            return

        cfg = CaptureConfig(
            backend="auto",
            width=1980,
            height=1020,
            warmup_s=0.3,
        )

        self._camera = CameraCapture(cfg)
        self._camera.start()

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self) -> None:
        while self._running and self._camera is not None:
            try:
                with self._camera_lock:
                    if not self._running or self._camera is None:
                        break
                    frame = self._camera.capture_rgb()

                with self._frame_lock:
                    self._latest_frame = frame

            except Exception as e:
                print(f"[DEBUG] Preview loop capture failed: {e}")
                time.sleep(0.05)
                continue

            time.sleep(0.03)

    def get_latest_frame(self) -> Optional[np.ndarray]:
        with self._frame_lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    def capture_to_file(self, out_path: Path) -> None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if self._camera is None:
            raise RuntimeError("Camera is not running.")

        with self._camera_lock:
            frame = self._camera.capture_rgb()

        if frame is None:
            raise RuntimeError("Failed to capture frame.")
        
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        ok = cv2.imwrite(str(out_path), bgr)
        if not ok:
            raise RuntimeError(f"Failed to save image: {out_path}")

        with self._frame_lock:
            self._latest_frame = frame.copy()

    def stop(self) -> None:
        self._running = False

        thread = self._thread
        self._thread = None
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.5)

        with self._camera_lock:
            camera = self._camera
            self._camera = None
            if camera is not None:
                try:
                    camera.stop()
                except Exception:
                    pass

        with self._frame_lock:
            self._latest_frame = None



