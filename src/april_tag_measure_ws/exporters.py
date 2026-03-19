#!/usr/bin/env python3
"""
exporters.py


Export helpers for AprilTag measurement results.


Purpose:
- Save annotated measurement image
- Save JSON results
- Append per-line measurements to a CSV file


Date: 19/03/2026
Version: 1.0
Maintainer: Victor Lim - victor@polymaya.tech
"""


from __future__ import annotations


import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable


import cv2


from src.april_tag_measure_ws.measurement_session import Results


CSV_HEADERS = [
    "image",
    "tag_id",
    "tag_family",
    "tag_size_m",
    "px_per_meter_quick",
    "pose_available",
    "backend",
    "line_index",
    "p0_x",
    "p0_y",
    "p1_x",
    "p1_y",
    "length_m",
    "mode",
]


def save_annotated_image(image_bgr, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(output_path), image_bgr)
    if not ok:
        raise RuntimeError(f"Failed to save annotated image: {output_path}")
    return output_path


def save_results_json(results: Results, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asdict(results), indent=2))
    return output_path


def append_results_csv(results: Results, csv_path: Path) -> Path:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()

    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)

        if not file_exists:
            writer.writeheader()

        for idx, line in enumerate(results.lines, start=1):
            writer.writerow(
                {
                    "image": results.image,
                    "tag_id": results.tag_id,
                    "tag_family": results.tag_family,
                    "tag_size_m": results.tag_size_m,
                    "px_per_meter_quick": results.px_per_meter_quick,
                    "pose_available": results.pose_available,
                    "backend": results.backend,
                    "line_index": idx,
                    "p0_x": line.p0[0],
                    "p0_y": line.p0[1],
                    "p1_x": line.p1[0],
                    "p1_y": line.p1[1],
                    "length_m": line.length_m,
                    "mode": line.mode,
                }
            )

    return csv_path