import logging
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from backend.app.core.dependencies import get_current_user, require_permission
from backend.app.core.permissions import Permission
from backend.app.database.repositories.audit_repo import AuditRepository
from backend.app.database.repositories.timeline_repo import TimelineRepository
from backend.app.ai.timeline.event_extractor import event_extractor

logger = logging.getLogger("archivegov.timeline_router")

router = APIRouter(tags=["Historical Timeline"])
timeline_repo = TimelineRepository()
audit_repo = AuditRepository()

class GenerateTimelineRequest(BaseModel):
    timeline_id: Optional[str] = Field("vault_master", description="Target timeline collection id")
    department: Optional[str] = Field(None, description="Filter by department, or null for all")
    document_id: Optional[str] = Field(None, description="Optional specific document context")

@router.post("/generate")
def generate_timeline(
    payload: GenerateTimelineRequest,
    user: dict = Depends(require_permission(Permission.DOCUMENT_VIEW))
):
    """
    Generate or refresh chronological timeline milestones from vault documents (Module 2).
    """
    events = event_extractor.generate_timeline(
        timeline_id=payload.timeline_id or "vault_master",
        department=payload.department,
        doc_id=payload.document_id
    )
    audit_repo.log(
        user_id=user["_id"],
        action="TIMELINE_GENERATE",
        resource_type="timeline",
        resource_id=payload.timeline_id or "vault_master",
        metadata={"total_events": len(events), "department": payload.department}
    )
    return {
        "success": True,
        "timeline_id": payload.timeline_id or "vault_master",
        "total_events": len(events),
        "events": events
    }

@router.get("")
@router.get("/")
def get_timeline(
    department: Optional[str] = Query(None),
    document_id: Optional[str] = Query(None),
    certainty: Optional[str] = Query(None, description="Filter: verified, inferred, uncertain, conflicting"),
    entity: Optional[str] = Query(None, description="Filter by person, location, or official entity"),
    user: dict = Depends(require_permission(Permission.DOCUMENT_VIEW))
):
    """
    Retrieve chronological timeline events with filtering by certainty, entity, and department.
    """
    events = timeline_repo.find_events(
        doc_id=document_id,
        department=department,
        certainty=certainty,
        entity=entity
    )
    # If no events found in repository, dynamically generate from vault
    if not events:
        events = event_extractor.generate_timeline(
            timeline_id="vault_master",
            department=department,
            doc_id=document_id
        )
        if certainty and certainty != "all":
            events = [e for e in events if e.get("certainty") == certainty]
        if entity:
            events = [e for e in events if any(entity.lower() in ent.lower() for ent in e.get("entities", []))]

    return {
        "total_events": len(events),
        "events": events
    }

@router.get("/{timeline_id}")
def get_timeline_by_id(
    timeline_id: str,
    user: dict = Depends(require_permission(Permission.DOCUMENT_VIEW))
):
    """
    Retrieve events for a specific timeline group.
    """
    events = timeline_repo.find_by_timeline_id(timeline_id)
    if not events:
        events = event_extractor.generate_timeline(timeline_id=timeline_id)
    return {
        "timeline_id": timeline_id,
        "total_events": len(events),
        "events": events
    }
