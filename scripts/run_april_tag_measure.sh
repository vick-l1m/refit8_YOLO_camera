#!/usr/bin/env bash
set -euo pipefail

# run_april_tag_measure.sh
#
# Usage:
#   ./run_april_tag_measure.sh /path/to/image.jpg --name image_measured [other args...]
#
# Notes:
# - The FIRST argument must be the image path.
# - Everything after the image path is forwarded to the Python program unchanged.
# - No default image is provided; missing image is an error.

if [[ $# -lt 1 ]]; then
  echo "ERROR: Missing image path."
  echo "Usage: $0 /path/to/image.jpg [--name NAME] [--tag-size-m 0.05] [--family tag25h9] ..."
  exit 2
fi

IMAGE="$1"
shift

if [[ ! -f "$IMAGE" ]]; then
  echo "ERROR: Image file not found: $IMAGE"
  exit 2
fi

# -----------------------
# USER DEFAULT SETTINGS (can be overridden by CLI args)
# -----------------------
TAG_SIZE_M_DEFAULT="0.05"
FAMILY_DEFAULT="tag25h9"

# -----------------------
# RUN
# -----------------------
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$REPO_ROOT/src/april_tag_measure_ws/april_tag_measure.py"

# Provide defaults, but allow user to override by passing their own flags after IMAGE.
python3 "$PY" \
  "$IMAGE" \
  --tag-size-m "$TAG_SIZE_M_DEFAULT" \
  --family "$FAMILY_DEFAULT" \
  "$@"


