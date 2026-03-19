#!/usr/bin/env python3
"""
app_context.py

Defines the AppContext model, which represents the application context and holds shared state.

Date: 19/03/2026
Version: 1.1
Maintainer: Victor Lim - victor@polymaya.tech
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.models.audit_session import AuditSession

@dataclass
class AppContext:
    repo_root: Path
    current_state: str = "START"
    current_audit: Optional[AuditSession] = None
    # Location
    available_locations: List[Dict[str, Any]] = field(default_factory=list)
    current_location: Optional[Dict[str, Any]] = None

    # Asset Category
    available_asset_categories: List[Dict[str, Any]] = field(default_factory=list)
    current_asset_category: Optional[Dict[str, Any]] = None

    # Images
    last_captured_image_path: Optional[Path] = None
    last_captured_image_name: Optional[str] = None

    current_item_dir: Optional[Path] = None
    current_item_base_image_name: Optional[str] = None

    preview_service: Optional[Any] = None
    base_image_preview_process: Optional[Any] = None

    # AprilTag measurement
    apriltag_detected: bool = False
    apriltag_detection_family: str = "tag25h9"
    apriltag_tag_size_m: Optional[float] = None
    apriltag_intrinsics_path: Optional[Path] = None

    apriltag_source_image_path: Optional[Path] = None
    apriltag_annotated_image_path: Optional[Path] = None
    apriltag_measurements_csv_path: Optional[Path] = None

    apriltag_selected_tag: Optional[Dict[str, Any]] = None
    apriltag_session: Optional[Any] = None