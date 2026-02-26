import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Any, Tuple


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def now_ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def summarize_detections(results0) -> dict:
    """
    Build summary for a single YOLO result:
      - objects: per class {count, avg_confidence, max_confidence}
      - detections: list of {label, confidence, bbox_xyxy}
    """
    names = results0.names  # dict-like or list-like
    dets = []
    per_class = {}  # label -> {"count": int, "conf_sum": float, "max_conf": float}

    if results0.boxes is None or len(results0.boxes) == 0:
        return {"objects": {}, "detections": []}

    boxes = results0.boxes
    cls_list = boxes.cls.tolist()
    conf_list = boxes.conf.tolist()
    xyxy_list = boxes.xyxy.tolist()

    for cls_id, conf, xyxy in zip(cls_list, conf_list, xyxy_list):
        cls_id_int = int(cls_id)
        label = names[cls_id_int] if isinstance(names, (dict, list)) else str(cls_id_int)

        dets.append(
            {"label": label, "confidence": float(conf), "bbox_xyxy": [float(x) for x in xyxy]}
        )

        if label not in per_class:
            per_class[label] = {"count": 0, "conf_sum": 0.0, "max_conf": 0.0}
        per_class[label]["count"] += 1
        per_class[label]["conf_sum"] += float(conf)
        per_class[label]["max_conf"] = max(per_class[label]["max_conf"], float(conf))

    objects = {}
    for label, v in per_class.items():
        objects[label] = {
            "count": v["count"],
            "avg_confidence": (v["conf_sum"] / v["count"]) if v["count"] else 0.0,
            "max_confidence": v["max_conf"],
        }

    return {"objects": objects, "detections": dets}


@dataclass
class ClassStats:
    count: int = 0
    conf_sum: float = 0.0
    max_conf: float = 0.0

    def update(self, conf: float) -> None:
        self.count += 1
        self.conf_sum += conf
        self.max_conf = max(self.max_conf, conf)

    @property
    def avg_conf(self) -> float:
        return (self.conf_sum / self.count) if self.count else 0.0


def update_global_stats(global_stats: Dict[str, ClassStats], detections: List[dict]) -> None:
    """
    Update running stats (count/avg/max confidence) across the whole session.
    """
    for d in detections:
        label = d["label"]
        conf = float(d["confidence"])
        if label not in global_stats:
            global_stats[label] = ClassStats()
        global_stats[label].update(conf)


def stats_to_objects(global_stats: Dict[str, ClassStats]) -> Dict[str, dict]:
    """
    Convert running stats into the requested JSON format:
      unique_objects + counts + confidence
    """
    out = {}
    for label, st in global_stats.items():
        out[label] = {
            "count": st.count,
            "avg_confidence": st.avg_conf,
            "max_confidence": st.max_conf,
        }
    return out


def write_events_json(
    out_path: Path,
    *,
    video_path: Path,
    snapshots_dir: Path,
    yolo_model: str,
    imgsz: int,
    conf_threshold: float,
    capture_width: int,
    capture_height: int,
    fps: int,
    infer_interval_s: float,
    objects: Dict[str, dict],
    events: List[dict],
) -> None:
    payload = {
        "video": str(video_path),
        "snapshots_dir": str(snapshots_dir),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "yolo": {
            "model": yolo_model,
            "imgsz": imgsz,
            "conf_threshold": conf_threshold,
            "infer_interval_s": infer_interval_s,
        },
        "capture": {
            "width": capture_width,
            "height": capture_height,
            "fps": fps,
        },
        "objects": objects,   # unique_objects + counts + confidence
        "events": events,     # per-new-object event log
    }

    ensure_dir(out_path.parent)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)