#!/usr/bin/env bash
set -e

# -----------------------
# USER SETTINGS
# -----------------------
IMAGE="$HOME/Downloads/kitchen_super_cropped.jpg"
TAG_SIZE_M="0.05"
FAMILY="tag25h9"

# Optional intrinsics (relative to repo root works)
INTRINSICS_JSON="calibration/fake_intrinsics_1920x1080.json"

NAME="apriltag_ruler"

# -----------------------
# RUN
# -----------------------
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

python3 "$REPO_ROOT/src/april_tag_measure_ws/april_tag_measure.py" \
  "$IMAGE" \
  --tag-size-m "$TAG_SIZE_M" \
  --family "$FAMILY" \
  --intrinsics "$INTRINSICS_JSON" \
  --name "$NAME"