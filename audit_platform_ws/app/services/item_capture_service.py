"""
item_capture_service.py

Defines the ItemCaptureService class, which handles capturing images of items using the camera and saving them along with metadata.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from app.models.app_context import AppContext
from src.camera_capture import CameraCapture, CaptureConfig, RPICamStillConfig


class ItemCaptureService:
    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root
        self.items_root = repo_root / "audit_platform_ws" / "app" / "data" / "items"
        self.images_dir = self.items_root / "images"

    def capture_base_item_image(self, context: AppContext) -> Dict[str, Any]:
        self.images_dir.mkdir(parents=True, exist_ok=True)

        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        image_name = f"{timestamp}.jpg"
        image_path = self.images_dir / image_name

        self._capture_to_path(image_path)

        item_dir = self.items_root / Path(image_name).stem
        item_dir.mkdir(parents=True, exist_ok=True)

        csv_path = item_dir / "item.csv"
        self._write_item_csv(
            csv_path=csv_path,
            context=context,
            image_name=image_name,
            image_path=image_path,
            time_taken=now.isoformat(timespec="seconds"),
        )

        return {
            "image_name": image_name,
            "image_path": image_path,
            "item_dir": item_dir,
            "csv_path": csv_path,
            "time_taken": now.isoformat(timespec="seconds"),
        }

    def capture_additional_item_image(self, item_dir: Path) -> Dict[str, Any]:
        if not item_dir.exists():
            raise FileNotFoundError(f"Item directory does not exist: {item_dir}")

        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        image_name = f"{timestamp}.jpg"
        image_path = item_dir / image_name

        self._capture_to_path(image_path)

        return {
            "image_name": image_name,
            "image_path": image_path,
            "time_taken": now.isoformat(timespec="seconds"),
        }

    def _capture_to_path(self, image_path: Path) -> None:
        capture_cfg = CaptureConfig(
            backend="rpicam-still",
            rpicam=RPICamStillConfig(
                width=1920,
                height=1080,
                time_ms=3000,
                preview=True,
                autofocus=True,
                af_mode="continuous",
            ),
        )

        with CameraCapture(capture_cfg) as camera:
            camera.capture_to_file(image_path)

    def _write_item_csv(
        self,
        csv_path: Path,
        context: AppContext,
        image_name: str,
        image_path: Path,
        time_taken: str,
    ) -> None:
        audit = context.current_audit

        headers = [
            "image_name",
            "image_path",
            "audit_id",
            "location_id",
            "location_name",
            "asset_category_id",
            "asset_category_name",
            "time_taken",
        ]

        row = {
            "image_name": image_name,
            "image_path": str(image_path),
            "audit_id": audit.audit_id if audit else "",
            "location_id": audit.selected_location_id if audit else "",
            "location_name": audit.selected_location_name if audit else "",
            "asset_category_id": audit.selected_asset_category_id if audit else "",
            "asset_category_name": audit.selected_asset_category_name if audit else "",
            "time_taken": time_taken,
        }

        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerow(row)