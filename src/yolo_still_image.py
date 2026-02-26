#!/usr/bin/env python3
import argparse
import json
import subprocess
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


def ensure_dirs(images_dir: Path, data_dir: Path) -> None:
    images_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)


def capture_still_rpicam(out_path: Path, width: int, height: int, time_ms: int, preview: bool) -> None:
    """
    Capture a still image using rpicam-still.
    If preview=False -> uses -n (no preview window).
    """
    cmd = [
        "rpicam-still",
        "-t", str(time_ms),
        "-o", str(out_path),
        "--width", str(width),
        "--height", str(height),
    ]
    if not preview:
        cmd.insert(1, "-n")  # disable preview

    subprocess.run(cmd, check=True)


def summarize_detections(results0) -> dict:
    """
    Build summary:
      - per class: count, avg_conf, max_conf
      - detections list: label, conf, bbox (xyxy)
    """
    names = results0.names  # {class_id: "name"} or list-like

    dets = []
    per_class = {}  # label -> {"count": int, "conf_sum": float, "max_conf": float}

    if results0.boxes is None or len(results0.boxes) == 0:
        return {
            "objects": {},
            "detections": [],
        }

    boxes = results0.boxes
    cls_list = boxes.cls.tolist()
    conf_list = boxes.conf.tolist()
    xyxy_list = boxes.xyxy.tolist()

    for cls_id, conf, xyxy in zip(cls_list, conf_list, xyxy_list):
        cls_id_int = int(cls_id)
        label = names[cls_id_int] if isinstance(names, (dict, list)) else str(cls_id_int)

        dets.append(
            {
                "label": label,
                "confidence": float(conf),
                "bbox_xyxy": [float(x) for x in xyxy],
            }
        )

        if label not in per_class:
            per_class[label] = {"count": 0, "conf_sum": 0.0, "max_conf": 0.0}
        per_class[label]["count"] += 1
        per_class[label]["conf_sum"] += float(conf)
        per_class[label]["max_conf"] = max(per_class[label]["max_conf"], float(conf))

    # finalize avg_conf
    objects = {}
    for label, v in per_class.items():
        objects[label] = {
            "count": v["count"],
            "avg_confidence": v["conf_sum"] / v["count"] if v["count"] else 0.0,
            "max_confidence": v["max_conf"],
        }

    return {
        "objects": objects,
        "detections": dets,
    }


def main():
    p = argparse.ArgumentParser(
        description="Capture a still image with rpicam-still, run YOLO, save annotated image + JSON summary."
    )

    # Still-image capture params (mirrors your still script knobs)
    p.add_argument("--width", type=int, default=1920, help="Capture width (default: 1920)")
    p.add_argument("--height", type=int, default=1080, help="Capture height (default: 1080)")
    p.add_argument("--name", type=str, default="", help="Base filename without extension (default: auto timestamp)")
    p.add_argument("--time-ms", type=int, default=300, help="Capture time in ms (default: 300)")
    p.add_argument("--display", action="store_true", help="Display annotated image fullscreen with feh")
    p.add_argument("--no-preview", action="store_true", help="Disable preview window (adds -n)")

    # YOLO params
    p.add_argument("--model", type=str, default=str(Path.home() / "models" / "yolov8n.pt"),
                   help="Path to YOLO .pt model (default: ~/models/yolov8n.pt)")
    p.add_argument("--imgsz", type=int, default=320, help="YOLO inference size (default: 320)")
    p.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold (default: 0.25)")

    args = p.parse_args()

    # Output folders (as requested)
    images_dir = Path.home() / "captures" / "yolo" / "images"
    data_dir = images_dir / "data"
    ensure_dirs(images_dir, data_dir)

    # Name + paths
    if not args.name.strip():
        args.name = f"yolo_photo_{time.strftime('%Y%m%d_%H%M%S')}"

    out_img_path = images_dir / f"{args.name}.jpg"
    out_json_path = data_dir / f"{args.name}.json"

    # 1) capture still (raw overwrite to same path first, then we overwrite with annotated)
    print(f"[1/3] Capturing still -> {out_img_path}")
    capture_still_rpicam(
        out_path=out_img_path,
        width=args.width,
        height=args.height,
        time_ms=args.time_ms,
        preview=(not args.no_preview),
    )

    # 2) run YOLO on captured image
    print(f"[2/3] Running YOLO model -> {args.model}")
    model = YOLO(args.model)
    frame_bgr = cv2.imread(str(out_img_path))
    if frame_bgr is None:
        raise RuntimeError(f"Failed to read captured image: {out_img_path}")

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    results = model.predict(frame_rgb, imgsz=args.imgsz, conf=args.conf, verbose=False)
    r0 = results[0]

    # Ultralytics plot() returns an annotated image (typically BGR suitable for cv2.imwrite)
    annotated_bgr = r0.plot()

    # Save annotated image (overwrites the captured image, now with boxes/labels)
    cv2.imwrite(str(out_img_path), annotated_bgr)
    print(f"Saved annotated image -> {out_img_path}")

    # 3) write JSON summary
    summary = summarize_detections(r0)
    payload = {
        "image": str(out_img_path),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "yolo": {
            "model": args.model,
            "imgsz": args.imgsz,
            "conf_threshold": args.conf,
        },
        "capture": {
            "width": args.width,
            "height": args.height,
            "time_ms": args.time_ms,
            "preview": (not args.no_preview),
        },
        **summary,
    }

    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"[3/3] Saved JSON -> {out_json_path}")

    # Optional display
    if args.display:
        subprocess.run(["feh", "-F", str(out_img_path)], check=False)


if __name__ == "__main__":
    main()