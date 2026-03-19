"""
item_capture_service.py

Defines the ItemCaptureService class, which handles capturing images of items
using the camera and saving them along with metadata.

Storage structure:

app/data/items/
    item_1/
        item_1.jpg
        item_1_data.csv
        item_1_additional_images/
            item_1_1.jpg
            item_1_2.jpg
    item_2/
        item_2.jpg
        item_2_data.csv
        item_2_additional_images/
            item_2_1.jpg
"""

from __future__ import annotations

import csv
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from app.models.app_context import AppContext


class ItemCaptureService:
    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root
        self.items_root = repo_root / "audit_platform_ws" / "app" / "data" / "items"

    def capture_base_item_image(self, context: AppContext) -> Dict[str, Any]:
        self.items_root.mkdir(parents=True, exist_ok=True)

        now = datetime.now()
        time_taken = now.isoformat(timespec="seconds")

        item_number = self._get_next_item_number()
        item_name = f"item_{item_number}"

        item_dir = self.items_root / item_name
        item_dir.mkdir(parents=True, exist_ok=False)

        image_name = f"{item_name}.jpg"
        image_path = item_dir / image_name

        csv_name = f"{item_name}_data.csv"
        csv_path = item_dir / csv_name

        additional_images_dir = item_dir / f"{item_name}_additional_images"
        additional_images_dir.mkdir(parents=True, exist_ok=True)

        self._capture_to_path(context, image_path)

        self._write_item_csv(
            csv_path=csv_path,
            context=context,
            item_name=item_name,
            image_name=image_name,
            image_path=image_path,
            item_dir=item_dir,
            additional_images_dir=additional_images_dir,
            time_taken=time_taken,
        )

        return {
            "item_name": item_name,
            "image_name": image_name,
            "image_path": image_path,
            "item_dir": item_dir,
            "csv_path": csv_path,
            "additional_images_dir": additional_images_dir,
            "time_taken": time_taken,
        }

    def capture_additional_item_image(self, context: AppContext) -> Dict[str, Any]:
        item_dir = context.current_item_dir
        base_image_name = context.current_item_base_image_name

        if not item_dir:
            raise ValueError("No current item directory is set in context.")

        if not base_image_name:
            raise ValueError("No base image name is set in context.")

        item_dir = Path(item_dir)

        if not item_dir.exists():
            raise FileNotFoundError(f"Item directory does not exist: {item_dir}")

        item_name = Path(base_image_name).stem
        additional_images_dir = item_dir / f"{item_name}_additional_images"
        additional_images_dir.mkdir(parents=True, exist_ok=True)

        now = datetime.now()
        time_taken = now.isoformat(timespec="seconds")

        additional_number = self._get_next_additional_image_number(
            additional_images_dir=additional_images_dir,
            item_name=item_name,
        )

        image_name = f"{item_name}_{additional_number}.jpg"
        image_path = additional_images_dir / image_name

        self._capture_to_path(context, image_path)

        return {
            "item_name": item_name,
            "image_name": image_name,
            "image_path": image_path,
            "item_dir": item_dir,
            "additional_images_dir": additional_images_dir,
            "time_taken": time_taken,
        }

    def _capture_to_path(self, context: AppContext, image_path: Path) -> None:
        if context.preview_service is None:
            raise RuntimeError("Preview/camera service is not available.")

        context.preview_service.capture_to_file(image_path)

    def _get_next_item_number(self) -> int:
        max_num = 0
        pattern = re.compile(r"^item_(\d+)$")

        if not self.items_root.exists():
            return 1

        for path in self.items_root.iterdir():
            if not path.is_dir():
                continue

            match = pattern.match(path.name)
            if match:
                max_num = max(max_num, int(match.group(1)))

        return max_num + 1

    def _get_next_additional_image_number(
        self,
        additional_images_dir: Path,
        item_name: str,
    ) -> int:
        max_num = 0
        pattern = re.compile(rf"^{re.escape(item_name)}_(\d+)\.jpg$")

        if not additional_images_dir.exists():
            return 1

        for path in additional_images_dir.iterdir():
            if not path.is_file():
                continue

            match = pattern.match(path.name)
            if match:
                max_num = max(max_num, int(match.group(1)))

        return max_num + 1

    def _write_item_csv(
        self,
        csv_path: Path,
        context: AppContext,
        item_name: str,
        image_name: str,
        image_path: Path,
        item_dir: Path,
        additional_images_dir: Path,
        time_taken: str,
    ) -> None:
        audit = context.current_audit

        rows = [
            ["field", "value"],
            ["item_name", item_name],
            ["image_name", image_name],
            ["image_path", str(image_path)],
            ["item_dir", str(item_dir)],
            ["additional_images_dir", str(additional_images_dir)],
            ["audit_id", audit.audit_id if audit else ""],
            ["location_id", audit.selected_location_id if audit else ""],
            ["location_name", audit.selected_location_name if audit else ""],
            ["asset_category_id", audit.selected_asset_category_id if audit else ""],
            ["asset_category_name", audit.selected_asset_category_name if audit else ""],
            ["time_taken", time_taken],
        ]

        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(rows)

