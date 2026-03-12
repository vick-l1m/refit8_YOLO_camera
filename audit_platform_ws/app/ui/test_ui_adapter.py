from __future__ import annotations

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

    def show_error(self, message: str) -> None:
        print(f"\n[ERROR] {message}")

    def show_info(self, message: str) -> None:
        print(f"\n[INFO] {message}")