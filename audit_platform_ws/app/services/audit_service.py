#!/usr/bin/env python3
"""
audit_services.py

Creates a new audit session with a unique ID and timestamp.

"""
from __future__ import annotations

from datetime import datetime

from app.models.audit_session import AuditSession

class AuditService:
    def create_new_audit(self) -> AuditSession:
        now = datetime.now()
        audit_id = now.strftime("%Y%m%d%H%M%S")
        audit_name = f"Audit {now.strftime('%Y-%m-%d %H:%M:%S')}"
        created_at = now.isoformat(timespec='seconds')

        return AuditSession(
            audit_id=audit_id,
            audit_name=audit_name,
            created_at=created_at
        )