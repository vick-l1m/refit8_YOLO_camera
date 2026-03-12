#!/usr/bin/env python3
"""
audit_session.py

Defines the AuditSession model, which represents a single audit session with its metadata and associated captures.

"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

@dataclass
class AuditSession:
    audit_id: str
    created_at: str
    audit_name: Optional[str] = None

    # Location
    selected_location_id: Optional[str] = None
    selected_location_name: Optional[str] = None
    
    # Asset Category
    selected_asset_category_id: Optional[str] = None
    selected_asset_category_name: Optional[str] = None

    