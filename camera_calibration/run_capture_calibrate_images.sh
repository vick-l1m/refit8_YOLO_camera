#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python3 capture_calibration_images.py --autofocus \
  --interval 3 \
  --width 1920 \
  --height 1080 \
  --af-mode auto \
  "$@"
