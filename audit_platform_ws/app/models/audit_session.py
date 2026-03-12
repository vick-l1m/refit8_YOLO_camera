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
    audit_name: Optional[str] = None
    created_at: datetime
    