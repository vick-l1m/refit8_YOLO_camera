#!/usr/bin/env python3
"""
app_context.py

Defines the AppContext model, which represents the application context and holds shared state.

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
    # Form
    available_forms: List[Dict[str, Any]] = field(default_factory=list)
    current_form: Optional[Dict[str, Any]] = None
    
