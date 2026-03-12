#!/usr/bin/env python3
"""
asset_category_service.py

AssetCategoryService class for loading and validating asset category data.
The asset category data is saved as a JSON file in data/definitions/asset_categories.json

load_asset_categories() reads the JSON file, validates its structure, and returns a list of asset category dictionaries with 'id' and 'name' keys.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


class AssetCategoryService:
    def __init__(self, definitions_dir: Path) -> None:
        self.definitions_dir = definitions_dir

    def load_asset_categories(self) -> List[Dict[str, Any]]:
        categories_path = self.definitions_dir / "asset_categories.json"

        if not categories_path.exists():
            raise FileNotFoundError(
                f"Asset categories file not found: {categories_path}"
            )

        with categories_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise ValueError("asset_categories.json must contain a top-level JSON object")

        if "asset_categories" not in data:
            raise ValueError("asset_categories.json must contain an 'asset_categories' key")

        categories = data["asset_categories"]

        if not isinstance(categories, list):
            raise ValueError("'asset_categories' must be a list")

        validated: List[Dict[str, Any]] = []
        for i, category in enumerate(categories):
            if not isinstance(category, dict):
                raise ValueError(f"Asset category entry at index {i} must be an object")

            if "id" not in category or "name" not in category:
                raise ValueError(
                    f"Asset category entry at index {i} must contain 'id' and 'name'"
                )

            if not isinstance(category["id"], str) or not category["id"].strip():
                raise ValueError(f"Asset category entry at index {i} has invalid 'id'")

            if not isinstance(category["name"], str) or not category["name"].strip():
                raise ValueError(f"Asset category entry at index {i} has invalid 'name'")

            validated.append(
                {
                    "id": category["id"].strip(),
                    "name": category["name"].strip(),
                }
            )

        return validated