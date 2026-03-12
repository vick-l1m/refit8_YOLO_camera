#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

python3 "$PROJECT_ROOT/camera_calibration/calibrate_camera.py" \
  --images "$PROJECT_ROOT/calibration/arducam_1920x1080/*.jpg" \
  --rows 6 \
  --cols 8 \
  --square 0.035 \
  --output "$PROJECT_ROOT/calibration/arducam_1920x1080/arducam_1920x1080_intrinsics.json"
