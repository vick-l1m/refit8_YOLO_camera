#!/usr/bin/env bash
set -euo pipefail

# Defaults
WIDTH=1280
HEIGHT=720
FPS=30
TIME_MS=10000
NAME=""             # if empty -> auto timestamp
OUTDIR="$HOME/captures/videos"
PLAY=0              # 1 = play after recording, 0 = don't
PREVIEW=1           # 1 = show preview while recording, 0 = disable preview (-n)
AUTOFOCUS=1          # 1 = enable autofocus (default on)
AF_MODE="continuous" # auto | continuous | manual (default: continuous)

usage() {
  cat <<EOF
Usage: $0 [options]

Options:
  -W, --width <px>        Capture width (default: $WIDTH)
  -H, --height <px>       Capture height (default: $HEIGHT)
  -f, --fps <fps>         Framerate (default: $FPS)
  -t, --time <ms>         Duration in ms (default: $TIME_MS)
  -n, --name <name>       Base filename (without extension). Default: auto timestamp
  -o, --outdir <dir>      Output directory (default: $OUTDIR)
      --play              Play fullscreen after recording
      --no-play           Don't play after recording (default)
      --preview           Show preview while recording (default)
      --no-preview        Disable preview (-n)
      --autofocus         Enable autofocus (default)
      --no-autofocus      Disable autofocus
      --af-mode <mode>    Autofocus mode: auto | continuous | manual
  -h, --help              Show help

Examples:
  $0
  $0 -t 30000 --play
  $0 --name test_run --no-preview --play
EOF
}

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    -W|--width) WIDTH="$2"; shift 2 ;;
    -H|--height) HEIGHT="$2"; shift 2 ;;
    -f|--fps) FPS="$2"; shift 2 ;;
    -t|--time) TIME_MS="$2"; shift 2 ;;
    -n|--name) NAME="$2"; shift 2 ;;
    -o|--outdir) OUTDIR="$2"; shift 2 ;;
    --play) PLAY=1; shift ;;
    --no-play) PLAY=0; shift ;;
    --preview) PREVIEW=1; shift ;;
    --no-preview) PREVIEW=0; shift ;;
    --autofocus) AUTOFOCUS=1; shift ;;
    --no-autofocus) AUTOFOCUS=0; shift ;;
    --af-mode) AF_MODE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1"; usage; exit 1 ;;
  esac
done

mkdir -p "$OUTDIR"

if [[ -z "$NAME" ]]; then
  NAME="video_$(date +%Y%m%d_%H%M%S)"
fi

OUT="$OUTDIR/${NAME}.mp4"

PREVIEW_FLAG=""
if [[ "$PREVIEW" -eq 0 ]]; then
  PREVIEW_FLAG="-n"
fi

AF_FLAG=""
if [[ "$AUTOFOCUS" -eq 1 ]]; then
  AF_FLAG="--autofocus-mode $AF_MODE"
fi

echo "Saving video to: $OUT"
rpicam-vid $PREVIEW_FLAG $AF_FLAG -t "$TIME_MS" -o "$OUT" --width "$WIDTH" --height "$HEIGHT" --framerate "$FPS"

if [[ "$PLAY" -eq 1 ]]; then
  mpv --fs "$OUT"
fi