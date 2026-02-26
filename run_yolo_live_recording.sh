#!/usr/bin/env bash
set -euo pipefail

# Defaults
# WIDTH=1280 HEIGHT=720 FPS=30 TIME_MS=10000 NAME="" OUTDIR fixed below PLAY=0 PREVIEW=1
#
# Any args you pass to this script override defaults via "$@"

python3 src/yolo_live_record.py \
  --width 1280 \
  --height 720 \
  --fps 30 \
  --time-ms 10000 \
  --model "$HOME/models/yolov8n.pt" \
  --imgsz 320 \
  --conf 0.25 \
  --infer-interval 0.5 \
  "$@"