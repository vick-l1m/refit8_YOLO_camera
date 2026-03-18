from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List


class UIAdapter:
    def show_start_screen(self) -> None:
        print("\n=== START SCREEN ===")
        print("1. Create New Audit")
        print("2. Edit Audit")

    def show_location_screen(self, locations: List[Dict[str, Any]]) -> None:
        print("\n=== LOCATION SELECT ===")
        for idx, loc in enumerate(locations, start=1):
            print(f"{idx}. {loc['name']} ({loc['id']})")

    def show_asset_category_screen(self, categories: List[Dict[str, Any]]) -> None:
        print("\n=== ASSET CATEGORY SELECT ===")
        for idx, category in enumerate(categories, start=1):
            print(f"{idx}. {category['name']} ({category['id']})")

    def show_item_menu_screen(self) -> None:
        print("\n=== ITEM MENU ===")
        print("1. Add New Item")
        print("2. Back to Location Select")
        print("3. Back to Asset Category Select")

    def show_camera_capture_screen(self) -> None:
        print("\n=== CAMERA CAPTURE ===")
        print("A live camera preview window should now be open.")
        print("Commands:")
        print("  1 -> Take Picture")
        print("  2 -> Cancel")

    def show_item_image_menu_screen(self, image_path: Path) -> None:
        print("\n=== ADDITIONAL IMAGES MENU ===")
        print(f"Current object image: {image_path}")
        print("1. Take New Image")
        print("2. End")

    def show_camera_capture_additional_screen(self) -> None:
        print("\n=== ADDITIONAL IMAGE CAPTURE ===")
        print("The original image for this item should be open.")
        print("A live camera preview should also be running.")
        print("Take more images of the same object from other angles.")
        print("Commands:")
        print("  1 -> Take Picture")
        print("  2 -> Cancel")

    def show_error(self, message: str) -> None:
        print(f"\n[ERROR] {message}")

    def show_info(self, message: str) -> None:
        print(f"\n[INFO] {message}")

