#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"   # scripts/ is one level down

cd "$PROJECT_ROOT"

# Run the module with sensible defaults.
# Any args you pass to this script override/extend the defaults via "$@"

python3 src/yolo_still_image.py \
  --width 1920 \
  --height 1080 \
  --time-ms 300 \
  --model "$HOME/models/yolov8n.pt" \
  --imgsz 320 \
  --conf 0.25 \
  --no-preview \
  "$@"