from fastapi import APIRouter, Depends, HTTPException, status
from backend.app.schemas.restoration import ReviewDecisionInput
from backend.app.core.dependencies import get_current_user, require_permission
from backend.app.core.permissions import Role, Permission
from backend.app.database.repositories.document_repo import DocumentRepository
from backend.app.database.repositories.review_repo import ReviewRepository
from backend.app.database.repositories.notification_repo import NotificationRepository
from backend.app.database.repositories.audit_repo import AuditRepository

router = APIRouter(prefix="/api/v1/reviews", tags=["Human Review & Approval"])
doc_repo = DocumentRepository()
review_repo = ReviewRepository()
notif_repo = NotificationRepository()
audit_repo = AuditRepository()

@router.get("/{document_id}")
def get_reviews(document_id: str, user: dict = Depends(get_current_user)):
    reviews = review_repo.find_by_document_id(document_id)
    return reviews

@router.post("/approve/{document_id}")
def approve_restoration(document_id: str, user: dict = Depends(require_permission(Permission.RESTORATION_APPROVE))):
    decision_input = ReviewDecisionInput(decision="approve", comments="Approved via archival sign-off.")
    return submit_review_decision(document_id=document_id, decision_input=decision_input, user=user)

@router.post("/{document_id}/decision")
def submit_review_decision(
    document_id: str,
    decision_input: ReviewDecisionInput,
    user: dict = Depends(require_permission(Permission.RESTORATION_APPROVE))
):
    # Only Reviewer and Admin can approve/reject
    if user.get("role") not in [Role.REVIEWER.value, Role.ADMIN.value]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only authorized Reviewers or System Administrators can sign off archival records."
        )

    doc = doc_repo.find_by_id(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    decision = decision_input.decision.lower()
    if decision not in ["approve", "reject", "needs_correction"]:
        raise HTTPException(status_code=400, detail="Decision must be approve, reject, or needs_correction.")

    status_map = {
        "approve": "approved",
        "reject": "rejected",
        "needs_correction": "needs_correction"
    }
    new_status = status_map[decision]

    # Save review record
    review_record = review_repo.create({
        "document_id": document_id,
        "version_id": decision_input.related_version_id or doc.get("current_version_id"),
        "reviewer_id": user["_id"],
        "reviewer_name": user.get("name", "Archival Reviewer"),
        "decision": decision,
        "comments": decision_input.comments
    })

    # Update document status
    doc_repo.update(document_id, {
        "status": new_status,
        "assigned_reviewer": user["_id"]
    })

    # Create notification for uploader
    notif_repo.create({
        "user_id": doc.get("uploaded_by", "all"),
        "type": f"document_{new_status}",
        "title": f"Document {decision.capitalize()}ed",
        "message": f"Archival record '{doc['title']}' review decision: {decision.upper()}. Comments: {decision_input.comments}",
        "related_document_id": document_id
    })

    # Audit log
    audit_repo.log(
        user_id=user["_id"],
        action=f"REVIEW_{decision.upper()}",
        resource_type="document",
        resource_id=document_id,
        metadata={"decision": decision, "comments": decision_input.comments}
    )

    return {
        "success": True,
        "message": f"Document review decision '{decision}' recorded successfully.",
        "new_status": new_status,
        "review": review_record
    }

@router.post("/{document_id}/archive")
def archive_record(document_id: str, user: dict = Depends(get_current_user)):
    """Move approved document to permanent sovereign preservation archive."""
    doc = doc_repo.find_by_id(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    if doc.get("status") != "approved" and user.get("role") != Role.ADMIN.value:
        raise HTTPException(status_code=400, detail="Only approved documents can be committed to permanent archive.")

    doc_repo.update(document_id, {"status": "archived"})

    audit_repo.log(
        user_id=user["_id"],
        action="PERMANENT_ARCHIVE_SEALED",
        resource_type="document",
        resource_id=document_id
    )

    return {"success": True, "message": "Document committed to Permanent Sovereign Vault."}

