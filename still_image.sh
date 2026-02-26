#!/usr/bin/env bash
set -euo pipefail

# Defaults
WIDTH=1920
HEIGHT=1080
NAME=""            # if empty -> auto timestamp
OUTDIR="$HOME/captures/photos"
DISPLAY=1          # 1 = show with feh, 0 = don't
PREVIEW=0          # 0 = use -n (no preview), 1 = allow preview window
TIME_MS=300        # rpicam-still capture time (ms)

usage() {
  cat <<EOF
Usage: $0 [options]

Options:
  -W, --width <px>        Capture width (default: $WIDTH)
  -H, --height <px>       Capture height (default: $HEIGHT)
  -n, --name <name>       Base filename (without extension). Default: auto timestamp
  -o, --outdir <dir>      Output directory (default: $OUTDIR)
      --display           Display image fullscreen after capture (default)
      --no-display        Do not display after capture
      --preview           Enable preview window (omit -n)
      --no-preview        Disable preview window (-n) (default)
  -t, --time <ms>         Capture time in ms (default: $TIME_MS)
  -h, --help              Show help

Examples:
  $0
  $0 -W 1280 -H 720 --no-display
  $0 --name my_photo --display
EOF
}

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    -W|--width) WIDTH="$2"; shift 2 ;;
    -H|--height) HEIGHT="$2"; shift 2 ;;
    -n|--name) NAME="$2"; shift 2 ;;
    -o|--outdir) OUTDIR="$2"; shift 2 ;;
    --display) DISPLAY=1; shift ;;
    --no-display) DISPLAY=0; shift ;;
    --preview) PREVIEW=1; shift ;;
    --no-preview) PREVIEW=0; shift ;;
    -t|--time) TIME_MS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1"; usage; exit 1 ;;
  esac
done

mkdir -p "$OUTDIR"

if [[ -z "$NAME" ]]; then
  NAME="photo_$(date +%Y%m%d_%H%M%S)"
fi

OUT="$OUTDIR/${NAME}.jpg"

# Preview flag: -n disables preview
PREVIEW_FLAG=""
if [[ "$PREVIEW" -eq 0 ]]; then
  PREVIEW_FLAG="-n"
fi

echo "Saving still to: $OUT"
rpicam-still $PREVIEW_FLAG -t "$TIME_MS" -o "$OUT" --width "$WIDTH" --height "$HEIGHT"

if [[ "$DISPLAY" -eq 1 ]]; then
  feh -F "$OUT"
fi