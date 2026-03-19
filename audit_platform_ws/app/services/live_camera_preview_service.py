from __future__ import annotations

import shutil
import subprocess
from typing import Optional


class LiveCameraPreviewService:
    """
    Starts/stops a live Raspberry Pi camera preview using rpicam-hello.
    """

    def __init__(self) -> None:
        self._process: Optional[subprocess.Popen] = None

    def start(self) -> subprocess.Popen:
        if self._process is not None and self._process.poll() is None:
            return self._process

        if shutil.which("rpicam-hello") is None:
            raise RuntimeError(
                "rpicam-hello not found. Make sure rpicam-apps is installed."
            )

        # -t 0 means run continuously until killed
        self._process = subprocess.Popen(
            ["rpicam-hello", "-t", "0"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return self._process

    def stop(self) -> None:
        if self._process is None:
            return

        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()

        self._process = None