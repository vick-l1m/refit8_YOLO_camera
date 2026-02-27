#!/usr/bin/env python3
"""
yolo_live_record.py

Record video (optional) + run YOLO at fixed intervals.
When new object classes appear, save an annotated snapshot and update events.json.

Camera I/O uses shared helpers from camera_capture.py:
- CameraCapture
- CaptureConfig
- RPICamStillConfig

Backends:
- auto (picamera2 if available, else opencv)
- picamera2
- opencv
- rpicam-still (works but not ideal for high-FPS video)
"""

from __future__ import annotations

import argparse
import subprocess
import time
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
from ultralytics import YOLO

from yolo_helpers import (
    ensure_dir,
    now_ts,
    summarize_detections,
    update_global_stats,
    stats_to_objects,
    write_events_json,
    ClassStats,
)

from camera_capture import CameraCapture, CaptureConfig, RPICamStillConfig


# ----------------------------
# Args / Config
# ----------------------------

@dataclass
class AppConfig:
    # Capture / run
    width: int
    height: int
    fps: int
    time_ms: int
    name: str

    # Output paths (inside repo captures/)
    outdir: Path

    # UI / behavior
    play: bool
    preview: bool
    record: bool  # ✅ video writing optional

    # Camera backend selection
    backend: str              # auto | picamera2 | opencv | rpicam-still
    device_index: int         # OpenCV device index
    warmup_s: float

    # Picamera2 autofocus
    af_mode: int
    af_trigger: int

    # rpicam-still options (only used if backend=rpicam-still)
    rpicam_time_ms: int
    rpicam_preview: bool
    rpicam_autofocus: bool
    rpicam_af_mode: str

    # YOLO
    model: str
    imgsz: int
    conf: float
    infer_interval_s: float


def parse_args(project_root: Path) -> AppConfig:
    p = argparse.ArgumentParser(
        description="YOLO live preview + optional video recording + snapshots on new object classes."
    )

    # Record options (match your bash script style)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument(
        "--time-ms",
        type=int,
        default=10000,
        help="Duration in ms. Use 0 to run until 'q'. (Default: 10000)",
    )
    p.add_argument("--name", type=str, default="", help="Base filename (no extension). Default: timestamp")

    # Output base directory inside repo
    p.add_argument(
        "--outdir",
        type=str,
        default=str(project_root / "captures"),
        help="Base captures directory (default: <repo>/captures)",
    )

    # Optional behaviors
    p.add_argument("--play", action="store_true", help="Play recorded video fullscreen after recording (mpv)")
    p.add_argument("--no-preview", action="store_true", help="Disable preview window (no cv2.imshow)")
    p.add_argument(
        "--no-record",
        action="store_true",
        help="Preview + YOLO + snapshots, but do not write a video file.",
    )

    # Camera selection (camera_capture.py)
    p.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "picamera2", "opencv", "rpicam-still"],
        help="Camera backend selection",
    )
    p.add_argument("--device-index", type=int, default=0, help="OpenCV camera device index")
    p.add_argument("--warmup-s", type=float, default=0.2, help="Warmup sleep after camera start")

    # Picamera2 autofocus
    p.add_argument("--af-mode", type=int, default=2)
    p.add_argument("--af-trigger", type=int, default=-1, help="-1 disables; >=0 attempts AfTrigger")

    # rpicam-still options (only used if backend=rpicam-still)
    p.add_argument("--rpicam-time-ms", type=int, default=3000)
    p.add_argument(
        "--rpicam-preview",
        action="store_true",
        help="Let rpicam-still show its own preview window (usually off).",
    )
    p.add_argument("--no-rpicam-autofocus", action="store_true")
    p.add_argument("--rpicam-af-mode", type=str, default="continuous", choices=["auto", "continuous", "manual"])

    # YOLO
    p.add_argument("--model", type=str, default=str(Path.home() / "models" / "yolov8n.pt"))
    p.add_argument("--imgsz", type=int, default=320)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--infer-interval", type=float, default=0.5, help="Seconds between YOLO inferences (default 0.5)")

    a = p.parse_args()

    return AppConfig(
        width=a.width,
        height=a.height,
        fps=a.fps,
        time_ms=a.time_ms,
        name=a.name.strip(),
        outdir=Path(a.outdir),
        play=a.play,
        preview=(not a.no_preview),
        record=(not a.no_record),
        backend=a.backend,
        device_index=a.device_index,
        warmup_s=a.warmup_s,
        af_mode=a.af_mode,
        af_trigger=a.af_trigger,
        rpicam_time_ms=a.rpicam_time_ms,
        rpicam_preview=a.rpicam_preview,
        rpicam_autofocus=(not a.no_rpicam_autofocus),
        rpicam_af_mode=a.rpicam_af_mode,
        model=a.model,
        imgsz=a.imgsz,
        conf=a.conf,
        infer_interval_s=a.infer_interval,
    )


# ----------------------------
# Main
# ----------------------------

def main():
    project_root = Path(__file__).resolve().parents[1]
    cfg = parse_args(project_root)

    # Output locations
    base_captures = cfg.outdir
    video_outdir = base_captures / "yolo" / "videos"
    ensure_dir(video_outdir)

    # Default name
    if not cfg.name:
        cfg.name = f"yolo_video_{now_ts()}"

    video_path = video_outdir / f"{cfg.name}.mp4"

    # Snapshots + JSON log folder
    snapshots_dir = base_captures / "yolo" / "videos" / "data" / cfg.name
    ensure_dir(snapshots_dir)
    events_json_path = snapshots_dir / "events.json"

    # Init YOLO
    model_path = Path(cfg.model)
    if not model_path.exists():
        raise FileNotFoundError(f"YOLO model not found: {model_path}")
    yolo = YOLO(str(model_path))

    # Build capture config (shared)
    af_trigger: Optional[int] = None if cfg.af_trigger < 0 else int(cfg.af_trigger)
    cap_cfg = CaptureConfig(
        backend=cfg.backend,
        width=cfg.width,
        height=cfg.height,
        warmup_s=cfg.warmup_s,
        device_index=cfg.device_index,
        af_mode=int(cfg.af_mode),
        af_trigger=af_trigger,
        rpicam=RPICamStillConfig(
            width=cfg.width,
            height=cfg.height,
            time_ms=cfg.rpicam_time_ms,
            preview=cfg.rpicam_preview,
            autofocus=cfg.rpicam_autofocus,
            af_mode=cfg.rpicam_af_mode,
        ),
    )

    # Optional video writer
    writer = None
    if cfg.record:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, float(cfg.fps), (cfg.width, cfg.height))
        if not writer.isOpened():
            raise RuntimeError("Failed to open cv2.VideoWriter. Try another codec/container if needed.")

    # Shared state for inference thread
    latest_annotated_bgr = None
    latest_lock = threading.Lock()
    inference_busy = False
    last_infer_t = 0.0

    # New-object trigger + logging
    seen_labels = set()
    global_stats: dict[str, ClassStats] = {}
    events: list[dict] = []

    def run_inference(frame_rgb, infer_time: float):
        nonlocal latest_annotated_bgr, inference_busy, seen_labels, global_stats, events
        try:
            results = yolo.predict(frame_rgb, imgsz=cfg.imgsz, conf=cfg.conf, verbose=False)
            r0 = results[0]

            annotated_bgr = r0.plot()  # typically BGR
            summary = summarize_detections(r0)

            # Update global stats
            update_global_stats(global_stats, summary["detections"])

            # Check for new labels
            labels_now = {d["label"] for d in summary["detections"]}
            new_labels = sorted([lab for lab in labels_now if lab not in seen_labels])

            if new_labels:
                stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(infer_time))
                primary = new_labels[0].replace(" ", "_")
                snap_name = f"{stamp}_{primary}.jpg"
                snap_path = snapshots_dir / snap_name
                cv2.imwrite(str(snap_path), annotated_bgr)

                for lab in new_labels:
                    confs = [float(d["confidence"]) for d in summary["detections"] if d["label"] == lab]
                    event = {
                        "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(infer_time)),
                        "new_label": lab,
                        "snapshot": str(snap_path),
                        "count_in_frame": len(confs),
                        "avg_conf_in_frame": (sum(confs) / len(confs)) if confs else 0.0,
                        "max_conf_in_frame": max(confs) if confs else 0.0,
                    }
                    events.append(event)
                    seen_labels.add(lab)

                # Update JSON on each trigger
                write_events_json(
                    events_json_path,
                    video_path=str(video_path) if writer is not None else "",
                    snapshots_dir=str(snapshots_dir),
                    yolo_model=str(model_path),
                    imgsz=cfg.imgsz,
                    conf_threshold=cfg.conf,
                    capture_width=cfg.width,
                    capture_height=cfg.height,
                    fps=cfg.fps,
                    infer_interval_s=cfg.infer_interval_s,
                    objects=stats_to_objects(global_stats),
                    events=events,
                )

            # Update latest annotated frame
            with latest_lock:
                latest_annotated_bgr = annotated_bgr

        except Exception as e:
            print("Inference error:", e)
        finally:
            inference_busy = False

    # Timing
    start_t = time.time()
    end_t = None
    if cfg.time_ms > 0:
        end_t = start_t + (cfg.time_ms / 1000.0)

    if writer is not None:
        print(f"Recording -> {video_path}")
    else:
        print("No-record mode: preview/snapshots only (video will NOT be saved).")

    print(f"Snapshots -> {snapshots_dir}")
    print("Press 'q' to stop early.")

    try:
        with CameraCapture(cap_cfg) as cam:
            while True:
                now = time.time()
                if end_t is not None and now >= end_t:
                    break

                frame_rgb = cam.capture_rgb()

                # Kick off inference periodically (non-blocking)
                if (now - last_infer_t >= cfg.infer_interval_s) and not inference_busy:
                    inference_busy = True
                    last_infer_t = now
                    threading.Thread(
                        target=run_inference,
                        args=(frame_rgb.copy(), now),
                        daemon=True
                    ).start()

                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                with latest_lock:
                    display_bgr = latest_annotated_bgr if latest_annotated_bgr is not None else frame_bgr

                # Ensure correct size
                if display_bgr.shape[1] != cfg.width or display_bgr.shape[0] != cfg.height:
                    display_bgr = cv2.resize(display_bgr, (cfg.width, cfg.height), interpolation=cv2.INTER_LINEAR)

                # Write to video (optional)
                if writer is not None:
                    writer.write(display_bgr)

                # Preview window
                if cfg.preview:
                    cv2.imshow("YOLO Live (q to quit)", display_bgr)
                    if (cv2.waitKey(1) & 0xFF) == ord("q"):
                        break

    finally:
        if writer is not None:
            writer.release()
        if cfg.preview:
            cv2.destroyAllWindows()

        # Always write final JSON (even if no new objects ever appeared)
        write_events_json(
            events_json_path,
            video_path=str(video_path) if writer is not None else "",
            snapshots_dir=str(snapshots_dir),
            yolo_model=str(model_path),
            imgsz=cfg.imgsz,
            conf_threshold=cfg.conf,
            capture_width=cfg.width,
            capture_height=cfg.height,
            fps=cfg.fps,
            infer_interval_s=cfg.infer_interval_s,
            objects=stats_to_objects(global_stats),
            events=stats_to_objects(global_stats) and events or events,
        )

    # Optional playback (only if we recorded)
    if cfg.play and (writer is not None):
        subprocess.run(["mpv", "--fs", str(video_path)], check=False)


if __name__ == "__main__":
    main()