#!/usr/bin/env python3
import argparse
import subprocess
import time
import threading
from dataclasses import dataclass
from pathlib import Path

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

from Arducam_ws.camera_focus import enable_autofocus, AutofocusConfig

# ----------------------------
# Camera abstraction
# ----------------------------

class BaseCamera:
    def start(self):
        raise NotImplementedError

    def read(self):
        """Return an RGB frame (H, W, 3) uint8."""
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError


class Picamera2Camera(BaseCamera):
    def __init__(self, camera_num=0, size=(1280, 720), autofocus: bool = True, af_mode: int = 2):
        from picamera2 import Picamera2
        self.camera_num = camera_num
        self.size = size
        self.autofocus = autofocus
        self.af_mode = af_mode
        self.picam2 = Picamera2(camera_num=camera_num)

    def start(self):
        self.picam2.configure(
            self.picam2.create_preview_configuration(main={"format": "RGB888", "size": self.size})
        )
        self.picam2.start()
        time.sleep(0.2)

        # Enable autofocus automatically (best effort)
        enable_autofocus(self.picam2, AutofocusConfig(enabled=self.autofocus, mode=self.af_mode))

    def read(self):
        return self.picam2.capture_array()

    def stop(self):
        self.picam2.stop()


class OpenCVCamera(BaseCamera):
    def __init__(self, device=0, size=(1280, 720)):
        self.device = device
        self.size = size
        self.cap = None

    def start(self):
        self.cap = cv2.VideoCapture(self.device)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera device {self.device}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.size[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[1])

    def read(self):
        ok, frame_bgr = self.cap.read()
        if not ok or frame_bgr is None:
            raise RuntimeError("Failed to read frame from OpenCV camera")
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    def stop(self):
        if self.cap:
            self.cap.release()


def make_camera(backend: str, *, camera_num: int, device: str, size: tuple,
                autofocus: bool, af_mode: int) -> BaseCamera:
    backend = backend.lower()
    if backend in ("picam2", "picamera2", "libcamera"):
        return Picamera2Camera(camera_num=camera_num, size=size, autofocus=autofocus, af_mode=af_mode)  
    if backend in ("opencv", "v4l2", "usb"):
        dev = device
        if isinstance(dev, str) and dev.isdigit():
            dev = int(dev)
        return OpenCVCamera(device=dev, size=size)
    raise ValueError(f"Unknown camera backend: {backend}")


# ----------------------------
# Args / Config
# ----------------------------

@dataclass
class AppConfig:
    # Record options (match record_videos.sh)
    width: int
    height: int
    fps: int
    time_ms: int
    name: str
    play: bool
    preview: bool

    # Camera backend selection
    backend: str
    camera_num: int
    device: str

    # YOLO
    model: str
    imgsz: int
    conf: float
    infer_interval_s: float

    # Autofocus (Picamera2 only)
    autofocus: bool
    af_mode: int


def parse_args() -> AppConfig:
    p = argparse.ArgumentParser(
        description="Record video with cv2.VideoWriter + YOLO snapshots when new object classes appear."
    )

    # Record options (same spirit as record_videos.sh)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--time-ms", type=int, default=10000, help="Duration in ms (default 10000). Use 0 to run until 'q'.")
    p.add_argument("--name", type=str, default="", help="Base filename (no extension). Default: timestamp")
    p.add_argument("--play", action="store_true", help="Play video fullscreen after recording (mpv)")
    p.add_argument("--no-preview", action="store_true", help="Disable preview window (no cv2.imshow)")

    # Camera selection
    p.add_argument("--backend", type=str, default="picam2", choices=["picam2", "opencv"])
    p.add_argument("--camera-num", type=int, default=0)
    p.add_argument("--device", type=str, default="0")

    # YOLO
    p.add_argument("--model", type=str, default=str(Path.home() / "models" / "yolov8n.pt"))
    p.add_argument("--imgsz", type=int, default=320)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--infer-interval", type=float, default=0.5, help="Seconds between YOLO inferences (default 0.5)")

    # Autofocus (Picamera2)
    p.add_argument("--no-autofocus", action="store_true", help="Disable autofocus (default: enabled)")
    p.add_argument("--af-mode", type=int, default=2)

    a = p.parse_args()

    return AppConfig(
        width=a.width,
        height=a.height,
        fps=a.fps,
        time_ms=a.time_ms,
        name=a.name,
        play=a.play,
        preview=(not a.no_preview),
        backend=a.backend,
        camera_num=a.camera_num,
        device=a.device,
        model=a.model,
        imgsz=a.imgsz,
        conf=a.conf,
        infer_interval_s=a.infer_interval,
        autofocus=(not a.no_autofocus),
        af_mode=a.af_mode,
    )


# ----------------------------
# Main
# ----------------------------

def main():
    cfg = parse_args()

    # Output locations
    video_outdir = Path.home() / "captures" / "yolo" / "videos"
    ensure_dir(video_outdir)

    if not cfg.name.strip():
        cfg.name = f"yolo_video_{now_ts()}"

    video_path = video_outdir / f"{cfg.name}.mp4"

    # Snapshots + JSON log folder
    snapshots_dir = Path.home() / "captures" / "yolo" / "videos" / "data" / cfg.name
    ensure_dir(snapshots_dir)
    events_json_path = snapshots_dir / "events.json"

    # Init YOLO
    model_path = Path(cfg.model)
    if not model_path.exists():
        raise FileNotFoundError(f"YOLO model not found: {model_path}")
    yolo = YOLO(str(model_path))

    # Init camera
    cam = make_camera(
        cfg.backend,
        camera_num=cfg.camera_num,
        device=cfg.device,
        size=(cfg.width, cfg.height),
        autofocus=cfg.autofocus,
        af_mode=cfg.af_mode,
    )
    cam.start()

    # Video writer (mp4)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, float(cfg.fps), (cfg.width, cfg.height))
    if not writer.isOpened():
        cam.stop()
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

            # Update global stats every inference
            update_global_stats(global_stats, summary["detections"])

            # Check for new labels
            labels_now = {d["label"] for d in summary["detections"]}
            new_labels = sorted([lab for lab in labels_now if lab not in seen_labels])

            if new_labels:
                # Save one snapshot for this inference when *any* new label appears
                # (includes all detections/boxes in the frame)
                stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(infer_time))
                # Make filename readable; include first new label
                primary = new_labels[0].replace(" ", "_")
                snap_name = f"{stamp}_{primary}.jpg"
                snap_path = snapshots_dir / snap_name
                cv2.imwrite(str(snap_path), annotated_bgr)

                # Record events for each new label
                for lab in new_labels:
                    # Pull confidences for that label in this inference
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

                # Update JSON file on each trigger
                write_events_json(
                    events_json_path,
                    video_path=video_path,
                    snapshots_dir=snapshots_dir,
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

            # Always update latest annotated for preview + recording overlay
            with latest_lock:
                latest_annotated_bgr = annotated_bgr

        except Exception as e:
            print("Inference error:", e)
        finally:
            inference_busy = False

    # Timing
    start_t = time.time()
    end_t = (start_t + (cfg.time_ms / 1000.0)) if cfg.time_ms > 0 else None

    print(f"Recording -> {video_path}")
    print(f"Snapshots -> {snapshots_dir}")
    print("Press 'q' to stop early.")

    try:
        while True:
            now = time.time()
            if end_t is not None and now >= end_t:
                break

            frame_rgb = cam.read()

            # Kick off inference periodically (non-blocking)
            if (now - last_infer_t >= cfg.infer_interval_s) and not inference_busy:
                inference_busy = True
                last_infer_t = now
                threading.Thread(
                    target=run_inference,
                    args=(frame_rgb.copy(), now),
                    daemon=True
                ).start()

            # Choose frame for writing: use annotated if available else raw
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            with latest_lock:
                display_bgr = latest_annotated_bgr if latest_annotated_bgr is not None else frame_bgr

            # Ensure correct size (just in case)
            if display_bgr.shape[1] != cfg.width or display_bgr.shape[0] != cfg.height:
                display_bgr = cv2.resize(display_bgr, (cfg.width, cfg.height), interpolation=cv2.INTER_LINEAR)

            # Write to video
            writer.write(display_bgr)

            # Preview window (maps to PREVIEW=1)
            if cfg.preview:
                cv2.imshow("YOLO Record (q to quit)", display_bgr)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break

    finally:
        writer.release()
        if cfg.preview:
            cv2.destroyAllWindows()
        cam.stop()

        # Write final JSON (even if no new objects ever appeared)
        write_events_json(
            events_json_path,
            video_path=video_path,
            snapshots_dir=snapshots_dir,
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

    # Optional playback (maps to PLAY=1)
    if cfg.play:
        subprocess.run(["mpv", "--fs", str(video_path)], check=False)


if __name__ == "__main__":
    main()