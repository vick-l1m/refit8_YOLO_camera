#!/usr/bin/env python3
"""
location_service.py

LocationService class for loading and validating location data.
The location data is saved as a JSON file in data/definitions/locations.json

load_locations() reads the JSON file, validates its structure, and returns a list of location dictionaries with 'id' and 'name' keys.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


class LocationService:
    def __init__(self, definitions_dir: Path) -> None:
        self.definitions_dir = definitions_dir

    def load_locations(self) -> List[Dict[str, Any]]:
        locations_path = self.definitions_dir / "locations.json"

        if not locations_path.exists():
            raise FileNotFoundError(f"Locations file not found: {locations_path}")

        with locations_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise ValueError("locations.json must contain a top-level JSON object")

        if "locations" not in data:
            raise ValueError("locations.json must contain a 'locations' key")

        locations = data["locations"]

        if not isinstance(locations, list):
            raise ValueError("'locations' must be a list")

        validated: List[Dict[str, Any]] = []
        for i, loc in enumerate(locations):
            if not isinstance(loc, dict):
                raise ValueError(f"Location entry at index {i} must be an object")

            if "id" not in loc or "name" not in loc:
                raise ValueError(
                    f"Location entry at index {i} must contain 'id' and 'name'"
                )

            if not isinstance(loc["id"], str) or not loc["id"].strip():
                raise ValueError(f"Location entry at index {i} has invalid 'id'")

            if not isinstance(loc["name"], str) or not loc["name"].strip():
                raise ValueError(f"Location entry at index {i} has invalid 'name'")

            validated.append(
                {
                    "id": loc["id"].strip(),
                    "name": loc["name"].strip(),
                }
            )

        return validated