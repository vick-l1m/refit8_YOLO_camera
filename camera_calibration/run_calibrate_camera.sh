#!/usr/bin/env bash
set -euo pipefail

python3 calibrate_camera.py \
  --images "calib_imgs/*.jpg" \
  --rows 6 --cols 8 \
  --square 0.035 \
  --output cv/yolo/intrinsics/intrinsics_1920x1080.json
  "$@"