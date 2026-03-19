"""
live_camera_preview_service.py

Embedded live camera preview service for Qt.

This service continuously captures RGB frames using CameraCapture and stores
the latest frame so the Qt UI can display it inside the application window.

Preferred backend:
- picamera2
Fallback:
- opencv

Notes:
- Do NOT use rpicam-still for live preview here.
- Still image capture remains handled separately by ItemCaptureService.
"""
from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np

from src.camera_capture import CameraCapture, CaptureConfig


class LiveCameraPreviewService:
    def __init__(self) -> None:
        self._camera: Optional[CameraCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None

    def start(self) -> None:
        if self._running:
            return

        cfg = CaptureConfig(
            backend="auto",   # tries picamera2 first, then opencv
            width=1280,
            height=720,
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
                frame = self._camera.capture_rgb()
                with self._lock:
                    self._latest_frame = frame
            except Exception:
                # Avoid crashing the whole UI if one frame fails
                time.sleep(0.05)
                continue

            time.sleep(0.03)  # ~30 FPS target

    def get_latest_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    def stop(self) -> None:
        self._running = False

        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)

        self._thread = None

        if self._camera is not None:
            try:
                self._camera.stop()
            except Exception:
                pass

        self._camera = None

        with self._lock:
            self._latest_frame = None