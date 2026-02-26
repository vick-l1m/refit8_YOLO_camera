#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# run_yolo_measure_3d.sh
#
# Starts yolo_measure_3d.py with configurable parameters.
# You can edit defaults below OR override at runtime by passing CLI args to this
# script, e.g.:
#   ./run_yolo_measure_3d.sh --distance_m 1.8 --angle_deg 55 --class_name any
# -----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"   # scripts/ is one level down

cd "$PROJECT_ROOT"

# --- Python interpreter ---
# If you're using a venv, point this to it. Otherwise "python3".
PYTHON_BIN="${PYTHON_BIN:-python3}"

# --- Path to the measurement module ---
MEASURE_PY="${MEASURE_PY:-$PROJECT_ROOT/yolo_measure_3d.py}"

# -----------------------------------------------------------------------------
# Required / important paths
# -----------------------------------------------------------------------------

# YOLO model (.pt)
MODEL_PATH="${MODEL_PATH:-$HOME/models/yolov8n.pt}"

# Camera intrinsics file created by calibrate_camera.py
# Must match your capture resolution (e.g., 1920x1080).
INTRINSICS_PATH="${INTRINSICS_PATH:-$PROJECT_ROOT/intrinsics/intrinsics_1920x1080.json}"

# Output folder. Script will create:
#   $OUTDIR/images/*.png
#   $OUTDIR/data/*.json
OUTDIR="${OUTDIR:-$HOME/captures/yolo/measure_3d}"

# Optional base filename (no extension). If empty, the python script uses timestamp.
NAME="${NAME:-}"

# -----------------------------------------------------------------------------
# Capture settings
# -----------------------------------------------------------------------------

# Capture resolution (must match intrinsics resolution)
WIDTH="${WIDTH:-1920}"
HEIGHT="${HEIGHT:-1080}"

# OpenCV fallback camera index (only used if your camera_capture backend falls back to OpenCV)
DEVICE_INDEX="${DEVICE_INDEX:-0}"

# Picamera2 autofocus best-effort (only used if camera_capture is using Picamera2 backend)
AF_MODE="${AF_MODE:-2}"          # 0=manual, 1=auto, 2=continuous (varies by driver)
AF_TRIGGER="${AF_TRIGGER:--1}"   # -1 disables, >=0 attempts a focus trigger

# -----------------------------------------------------------------------------
# YOLO detection settings
# -----------------------------------------------------------------------------

CONF="${CONF:-0.25}"          # Confidence threshold
IOU="${IOU:-0.45}"            # NMS IoU threshold
CLASS_NAME="${CLASS_NAME:-chair}"  # "chair" or "any" to accept all detected classes
MAX_DETECTIONS="${MAX_DETECTIONS:-5}"

# -----------------------------------------------------------------------------
# Distance settings (meters)
# -----------------------------------------------------------------------------

# Choose how distance is provided:
#  - constant : uses DISTANCE_M
#  - file     : reads distance (meters) from DISTANCE_FILE each run
DISTANCE_SOURCE="${DISTANCE_SOURCE:-constant}"

DISTANCE_M="${DISTANCE_M:-2.0}"                  # Used when DISTANCE_SOURCE=constant
DISTANCE_FILE="${DISTANCE_FILE:-/tmp/range_m.txt}"  # Used when DISTANCE_SOURCE=file

# -----------------------------------------------------------------------------
# 3D estimation tuning
# -----------------------------------------------------------------------------

# Angle used for:
#  - depth estimate heuristic
#  - and yaw rotation of the drawn 3D box
ANGLE_DEG="${ANGLE_DEG:-45.0}"

# Scales depth up/down. If depth looks too big/small, tune this first.
DEPTH_RATIO="${DEPTH_RATIO:-1.0}"

# Minimum depth clamp (meters) so it never collapses to zero.
DEPTH_MIN="${DEPTH_MIN:-0.05}"

# Undistort using intrinsics before measuring (recommended if your lens is wide-angle)
UNDISTORT="${UNDISTORT:-1}"      # 1 enables --undistort, 0 disables

# Preview window after processing (requires GUI / X display)
PREVIEW="${PREVIEW:-0}"          # 1 enables --preview, 0 disables

# -----------------------------------------------------------------------------
# Convert toggles to flags
# -----------------------------------------------------------------------------
UNDISTORT_FLAG=()
if [[ "$UNDISTORT" == "1" ]]; then
  UNDISTORT_FLAG=(--undistort)
fi

PREVIEW_FLAG=()
if [[ "$PREVIEW" == "1" ]]; then
  PREVIEW_FLAG=(--preview)
fi

NAME_FLAG=()
if [[ -n "${NAME}" ]]; then
  NAME_FLAG=(--name "$NAME")
fi

# -----------------------------------------------------------------------------
# Build command
# -----------------------------------------------------------------------------
CMD=(
  "$PYTHON_BIN" "$MEASURE_PY"
  --model "$MODEL_PATH"
  --intrinsics "$INTRINSICS_PATH"
  --outdir "$OUTDIR"
  "${NAME_FLAG[@]}"

  --width "$WIDTH"
  --height "$HEIGHT"
  --device_index "$DEVICE_INDEX"
  --af_mode "$AF_MODE"
  --af_trigger "$AF_TRIGGER"

  --conf "$CONF"
  --iou "$IOU"
  --class_name "$CLASS_NAME"
  --max_detections "$MAX_DETECTIONS"

  --distance_source "$DISTANCE_SOURCE"
  --distance_m "$DISTANCE_M"
  --distance_file "$DISTANCE_FILE"

  --angle_deg "$ANGLE_DEG"
  --depth_ratio "$DEPTH_RATIO"
  --depth_min "$DEPTH_MIN"

  "${UNDISTORT_FLAG[@]}"
  "${PREVIEW_FLAG[@]}"
)

# Append any extra CLI args passed to this script (overrides without editing)
CMD+=("$@")

echo "Running:"
printf '  %q' "${CMD[@]}"
echo
echo

exec "${CMD[@]}"