#!/usr/bin/env python3
"""
capture_calibration_images.py

Takes a photo every N seconds for camera calibration.
- Uses camera_capture.py (rpicam-still backend)
- Saves images to cv/yolo/calibration/arducam_1920x1080/
- Auto-numbers images
- Displays each captured image
- Press 'q' in preview window to stop

IMPORTANT:
Use the same resolution you will use for measurement (default: 1920x1080).
"""

import time
from pathlib import Path
import argparse

import cv2

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.camera_capture import CameraCapture, CaptureConfig, RPICamStillConfig

def main():
    parser = argparse.ArgumentParser(description="Auto capture calibration images.")
    parser.add_argument("--interval", type=float, default=5.0,
                        help="Seconds between captures (default: 5)")
    parser.add_argument("--width", type=int, default=1920,
                        help="Capture width (must match measurement resolution)")
    parser.add_argument("--height", type=int, default=1080,
                        help="Capture height (must match measurement resolution)")
    parser.add_argument("--time-ms", type=int, default=3000,
                        help="rpicam-still capture time in ms (default: 3000)")
    parser.add_argument("--autofocus", action="store_true",
                        help="Enable autofocus")
    parser.add_argument("--af-mode", type=str, default="continuous",
                        choices=["auto", "continuous", "manual"],
                        help="Autofocus mode (default: continuous)")
    args = parser.parse_args()

    # --------------------------------------------
    # Calibration folder
    # --------------------------------------------
    project_root = Path(__file__).resolve().parents[1]
    calib_dir = project_root / "calibration" / "arducam_1920x1080"
    calib_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving calibration images to: {calib_dir}")
    print("Press 'q' in the preview window to stop.")
    print("Move and tilt the checkerboard between captures!\n")

    # --------------------------------------------
    # Camera configuration (matches your still script)
    # --------------------------------------------
    cap_cfg = CaptureConfig(
        backend="rpicam-still",
        rpicam=RPICamStillConfig(
            width=args.width,
            height=args.height,
            time_ms=args.time_ms,
            preview=False,  # we show preview manually via OpenCV
            autofocus=args.autofocus,
            af_mode=args.af_mode,
        ),
    )

    cam = CameraCapture(cap_cfg)

    counter = 1

    try:
        while True:
            filename = f"calib_{args.width}x{args.height}_{counter:03d}.jpg"
            out_path = calib_dir / filename

            print(f"[{counter:03d}] Capturing -> {filename}")
            cam.capture_to_file(out_path)

            # Show preview
            img = cv2.imread(str(out_path))
            if img is not None:
                cv2.imshow("Calibration Capture", img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Stopping capture.")
                break

            counter += 1
            time.sleep(args.interval)

    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()