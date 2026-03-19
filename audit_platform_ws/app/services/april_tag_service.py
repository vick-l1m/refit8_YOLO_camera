#!/usr/bin/env python3
# Metadata:
# File: audit_platform_ws/app/services/april_tag_service.py
# Purpose: Service layer for AprilTag detection, session creation, and exports.
# Date: 19/03/2026
# Version: 1.1
# Maintainer: Victor Lim - victor@polymaya.tech

"""
april_tag_service.py


Provides reusable service methods that bridge the audit app and the AprilTag
measurement modules in src/april_tag_measure_ws.
"""
from __future__ import annotations


from pathlib import Path
from typing import Any, Dict, Optional

import cv2

from src.april_tag_measure_ws.april_tags import detect_apriltags, avg_tag_edge_px
from src.april_tag_measure_ws.measurements import (
    build_camera_matrix,
    load_intrinsics_json,
)
from src.april_tag_measure_ws.measurement_session import MeasurementSession
from src.april_tag_measure_ws.exporters import (
    append_results_csv,
    save_annotated_image,
)


class AprilTagService:
    TAG_SIZE_M_DEFAULT = 0.05
    FAMILY_DEFAULT = "tag25h9"

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root

    def get_default_intrinsics_path(self) -> Path:
        return (
            self.repo_root
            / "calibration"
            / "arducam_1920x1080"
            / "arducam_intrinsics_1920x1080.json"
        )

    def load_image(self, image_path: Path):
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to read image: {image_path}")
        return img

    def detect_best_tag(
        self,
        image_path: Path,
        family: str = FAMILY_DEFAULT,
    ) -> Optional[Dict[str, Any]]:
        img = self.load_image(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        dets = detect_apriltags(gray, family)
        if not dets:
            return None

        dets_sorted = sorted(
            dets,
            key=lambda d: avg_tag_edge_px(d["corners"]),
            reverse=True,
        )
        return dets_sorted[0]

    def build_session(
        self,
        image_path: Path,
        tag_det: Dict[str, Any],
        tag_family: str,
        tag_size_m: float,
        intrinsics_path: Optional[Path] = None,
        fx: float = 0.0,
        fy: float = 0.0,
        cx: float = 0.0,
        cy: float = 0.0,
    ) -> MeasurementSession:
        img = self.load_image(image_path)

        K = None
        dist = None

        if intrinsics_path is not None:
            K, dist = load_intrinsics_json(intrinsics_path)
        elif fx > 0 and fy > 0:
            K = build_camera_matrix(fx, fy, cx, cy)

        return MeasurementSession(
            img_bgr=img,
            tag_det=tag_det,
            tag_family=tag_family,
            tag_size_m=tag_size_m,
            K=K,
            dist=dist,
        )

    def detect_and_build_session(
        self,
        image_path: Path,
        family: str,
        tag_size_m: float,
        intrinsics_path: Optional[Path] = None,
        fx: float = 0.0,
        fy: float = 0.0,
        cx: float = 0.0,
        cy: float = 0.0,
    ) -> Optional[MeasurementSession]:
        tag = self.detect_best_tag(image_path=image_path, family=family)
        if tag is None:
            return None

        return self.build_session(
            image_path=image_path,
            tag_det=tag,
            tag_family=family,
            tag_size_m=tag_size_m,
            intrinsics_path=intrinsics_path,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
        )

    def detect_and_build_session_with_defaults(
        self,
        image_path: Path,
    ) -> Optional[MeasurementSession]:
        intrinsics_path = self.get_default_intrinsics_path()
        if not intrinsics_path.exists():
            intrinsics_path = None

        return self.detect_and_build_session(
            image_path=image_path,
            family=self.FAMILY_DEFAULT,
            tag_size_m=self.TAG_SIZE_M_DEFAULT,
            intrinsics_path=intrinsics_path,
        )

    def save_measurement_outputs(
        self,
        session: MeasurementSession,
        source_image_path: Optional[Path],
        item_dir: Optional[Path],
        item_base_image_name: Optional[str],
        csv_path: Optional[Path] = None,
    ) -> Dict[str, Path]:
        if item_dir is None:
            raise RuntimeError("Current item directory is not set.")

        if item_base_image_name is None or not item_base_image_name.strip():
            raise RuntimeError("Current item base image name is not set.")

        item_dir.mkdir(parents=True, exist_ok=True)

        image_stem = Path(item_base_image_name).stem

        annotated_image_path = item_dir / f"{image_stem}_measured.jpg"

        if csv_path is None:
            csv_path = item_dir / f"{image_stem}_data.csv"

        rendered = session.render()
        save_annotated_image(rendered, annotated_image_path)

        results = session.results(
            image_path=str(source_image_path) if source_image_path is not None else ""
        )
        append_results_csv(results, csv_path)

        return {
            "annotated_image_path": annotated_image_path,
            "csv_path": csv_path,
        }