#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MEASURE_PY="${MEASURE_PY:-$PROJECT_ROOT/src/yolo_measure_3d.py}"

# -----------------------------------------------------------------------------
# Required / important paths
# -----------------------------------------------------------------------------
MODEL_PATH="${MODEL_PATH:-$HOME/models/yolov8n.pt}"
INTRINSICS_PATH="${INTRINSICS_PATH:-$PROJECT_ROOT/calibration/arducam_1920x1080/arducam_intrinsics_1920x1080.json}"
OUTDIR="${OUTDIR:-$HOME/captures/yolo/measure_3d}"
NAME="${NAME:-}"   # optional; if empty -> timestamp used

# -----------------------------------------------------------------------------
# Capture settings
# -----------------------------------------------------------------------------
WIDTH="${WIDTH:-1920}"
HEIGHT="${HEIGHT:-1080}"
TIME_MS="${TIME_MS:-300}"          # capture time (ms)
AF_MODE="${AF_MODE:-continuous}"   # auto|continuous|manual (depends on rpicam version)

# -----------------------------------------------------------------------------
# YOLO detection settings
# -----------------------------------------------------------------------------
CONF="${CONF:-0.25}"
IOU="${IOU:-0.45}"
CLASS_NAME="${CLASS_NAME:-chair}"     # or "any"
MAX_DETECTIONS="${MAX_DETECTIONS:-5}"

# -----------------------------------------------------------------------------
# Distance settings
# -----------------------------------------------------------------------------
DISTANCE_SOURCE="${DISTANCE_SOURCE:-constant}"         # constant|file
DISTANCE_M="${DISTANCE_M:-2.0}"
DISTANCE_FILE="${DISTANCE_FILE:-/tmp/range_m.txt}"

# -----------------------------------------------------------------------------
# 3D estimation tuning
# -----------------------------------------------------------------------------
ANGLE_DEG="${ANGLE_DEG:-45.0}"
DEPTH_RATIO="${DEPTH_RATIO:-1.0}"
DEPTH_MIN="${DEPTH_MIN:-0.05}"

# -----------------------------------------------------------------------------
# Toggles
# -----------------------------------------------------------------------------
UNDISTORT="${UNDISTORT:-1}"   # 1 -> add --undistort
PREVIEW="${PREVIEW:-0}"       # 1 -> add --preview

# -----------------------------------------------------------------------------
# Parse optional --image from CLI (and strip it from args passed to Python)
# - We want:
#     ./run_yolo_measure_3d.sh --image foo.jpg
#   to skip capture, but still pass the rest of "$@" through.
# -----------------------------------------------------------------------------
IMAGE_ARG=""
PASS_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image)
      if [[ $# -lt 2 ]]; then
        echo "Error: --image requires a path" >&2
        exit 2
      fi
      IMAGE_ARG="$2"
      shift 2
      ;;
    --image=*)
      IMAGE_ARG="${1#*=}"
      shift 1
      ;;
    *)
      PASS_ARGS+=("$1")
      shift 1
      ;;
  esac
done

# -----------------------------------------------------------------------------
# Decide image path: use provided --image, otherwise capture
# -----------------------------------------------------------------------------
CAP_DIR="${CAP_DIR:-$OUTDIR/input}"
mkdir -p "$CAP_DIR"

if [[ -z "$NAME" ]]; then
  NAME="$(date +%Y%m%d_%H%M%S)"
fi

if [[ -n "$IMAGE_ARG" ]]; then
  # Use provided image, skip capture
  CAPTURE_PATH="$IMAGE_ARG"
  if [[ ! -f "$CAPTURE_PATH" ]]; then
    echo "Error: --image file not found: $CAPTURE_PATH" >&2
    exit 2
  fi
else
  # Capture a new still
  CAPTURE_PATH="$CAP_DIR/${NAME}.jpg"

  # Force resolution to match intrinsics
  # If your rpicam-still doesn't support --autofocus-mode, remove that line.
  rpicam-still \
    -t "$TIME_MS" \
    --width "$WIDTH" \
    --height "$HEIGHT" \
    --autofocus-mode "$AF_MODE" \
    -n \
    -o "$CAPTURE_PATH"
fi

# -----------------------------------------------------------------------------
# Optional flags
# -----------------------------------------------------------------------------
EXTRA_FLAGS=()
if [[ "$UNDISTORT" == "1" ]]; then
  EXTRA_FLAGS+=(--undistort)
fi
if [[ "$PREVIEW" == "1" ]]; then
  EXTRA_FLAGS+=(--preview)
fi

# -----------------------------------------------------------------------------
# Run processing
# PASS_ARGS are the user-provided args (minus --image), so -h still works.
# -----------------------------------------------------------------------------
exec "$PYTHON_BIN" "$MEASURE_PY" \
  --model "$MODEL_PATH" \
  --intrinsics "$INTRINSICS_PATH" \
  --outdir "$OUTDIR" \
  --name "$NAME" \
  --image "$CAPTURE_PATH" \
  --conf "$CONF" \
  --iou "$IOU" \
  --class_name "$CLASS_NAME" \
  --max_detections "$MAX_DETECTIONS" \
  --distance_source "$DISTANCE_SOURCE" \
  --distance_m "$DISTANCE_M" \
  --distance_file "$DISTANCE_FILE" \
  --angle_deg "$ANGLE_DEG" \
  --depth_ratio "$DEPTH_RATIO" \
  --depth_min "$DEPTH_MIN" \
  "${EXTRA_FLAGS[@]}" \
  "${PASS_ARGS[@]}"
