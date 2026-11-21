import logging
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from backend.app.core.dependencies import get_current_user, require_permission
from backend.app.core.permissions import Permission
from backend.app.database.repositories.audit_repo import AuditRepository
from backend.app.services.lens_service import lens_service

logger = logging.getLogger("archivegov.lens_router")

router = APIRouter(tags=["Document Intelligence Lens"])
audit_repo = AuditRepository()

class ExplainRegionRequest(BaseModel):
    region_id: Optional[str] = Field(None, description="Optional region ID selected in canvas")
    text: Optional[str] = Field(None, description="Direct text snippet to explain")
    query: Optional[str] = Field(None, description="Specific archival question")

@router.get("/{document_id}/regions")
def get_document_regions(
    document_id: str,
    page: int = Query(1, ge=1),
    granularity: Optional[str] = Query(None, description="Filter by word, line, or block"),
    user: dict = Depends(require_permission(Permission.DOCUMENT_VIEW))
):
    """
    Document Intelligence Lens: Retrieve normalized and pixel-level bounding boxes (Module 4).
    """
    res = lens_service.get_regions(document_id, page_number=page, granularity=granularity)
    if "error" in res:
        raise HTTPException(status_code=404, detail=res["error"])
    return res

@router.post("/{document_id}/explain")
def explain_document_region(
    document_id: str,
    payload: ExplainRegionRequest,
    user: dict = Depends(require_permission(Permission.DOCUMENT_VIEW))
):
    """
    Document Intelligence Lens: AI-powered contextual explanation with strict epistemic labeling.
    """
    res = lens_service.explain_region(
        doc_id=document_id,
        region_id=payload.region_id,
        text=payload.text,
        query=payload.query
    )
    if "error" in res:
        raise HTTPException(status_code=404, detail=res["error"])

    audit_repo.log(
        user_id=user["_id"],
        action="LENS_EXPLAIN",
        resource_type="document",
        resource_id=document_id,
        metadata={"region_id": payload.region_id, "query": payload.query}
    )
    return res

@router.get("/{document_id}/related")
def get_related_documents(
    document_id: str,
    region_id: Optional[str] = Query(None),
    user: dict = Depends(require_permission(Permission.DOCUMENT_VIEW))
):
    """
    Document Intelligence Lens: Discover cross-vault related folios and records.
    """
    related = lens_service.get_related_records(document_id, region_id=region_id)
    return {
        "document_id": document_id,
        "related_records": related,
        "total_related": len(related)
    }
