# camera_focus.py
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AutofocusConfig:
    enabled: bool = True
    mode: int = 2          # 2 is commonly "continuous"
    fallback_mode: int = 1 # fallback if mode fails


def enable_autofocus(picam2, cfg: AutofocusConfig = AutofocusConfig()) -> bool:
    """
    Best-effort autofocus enable for Picamera2/libcamera.
    Returns True if autofocus was successfully enabled, else False.

    Works with a Picamera2 instance (picamera2.Picamera2).
    Safe to call even if AF controls don't exist.
    """
    if not cfg.enabled:
        return False

    try:
        controls = getattr(picam2, "camera_controls", None)
        if not controls:
            print("[AF] No camera_controls found on object; cannot enable autofocus.")
            return False

        if "AfMode" not in controls:
            print("[AF] AfMode not exposed by this driver/camera.")
            return False

        # Try requested mode
        try:
            picam2.set_controls({"AfMode": int(cfg.mode)})
            print(f"[AF] Enabled autofocus: AfMode={cfg.mode}")
            return True
        except Exception as e:
            print(f"[AF] AfMode={cfg.mode} failed: {e}")

        # Try fallback
        try:
            picam2.set_controls({"AfMode": int(cfg.fallback_mode)})
            print(f"[AF] Enabled autofocus (fallback): AfMode={cfg.fallback_mode}")
            return True
        except Exception as e2:
            print(f"[AF] AfMode={cfg.fallback_mode} failed: {e2}")
            return False

    except Exception as e:
        print(f"[AF] Unexpected autofocus error: {e}")
        return False

def rpicam_autofocus_args(enabled: bool, mode: str = "continuous") -> list[str]:
    if not enabled:
        return []
    return ["--autofocus-mode", mode]