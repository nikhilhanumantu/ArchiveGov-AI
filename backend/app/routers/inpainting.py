import uuid
from fastapi import APIRouter, Depends, HTTPException
from backend.app.schemas.restoration import InpaintingRequest
from backend.app.core.dependencies import get_current_user
from backend.app.database.repositories.document_repo import DocumentRepository
from backend.app.database.repositories.version_repo import DocumentVersionRepository
from backend.app.database.repositories.audit_repo import AuditRepository
from backend.app.storage.local_storage import storage
from backend.app.ai.inpainting.lama_adapter import lama_adapter

router = APIRouter(prefix="/api/v1/inpainting", tags=["Selective Inpainting"])
doc_repo = DocumentRepository()
version_repo = DocumentVersionRepository()
audit_repo = AuditRepository()

@router.post("/apply")
def apply_selective_inpainting(request: InpaintingRequest, user: dict = Depends(get_current_user)):
    doc = doc_repo.find_by_id(request.document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    # Determine input image (prefer current restored or fallback to original)
    input_file_key = doc.get("restored_file_key") or doc["original_file_key"]
    input_path = storage.resolve_path(input_file_key)
    if not input_path or not input_path.exists():
        raise HTTPException(status_code=404, detail="Source image file not found.")

    # Save mask if provided
    mask_key = None
    if request.mask_data_base64:
        mask_key = f"mask_{uuid.uuid4().hex[:8]}.png"
        mask_dest = storage.masks / mask_key
        # Extract base64 and save
        import base64
        data = base64.b64decode(request.mask_data_base64.split(",")[-1])
        with open(mask_dest, "wb") as f:
            f.write(data)

    output_key = f"inpaint_{uuid.uuid4().hex[:8]}.png"
    output_path = str(storage.restored / output_key)

    inpaint_result = lama_adapter.inpaint(
        image_path=str(input_path),
        mask_path_or_bytes=request.mask_data_base64 or "",
        output_path=output_path,
        feather_radius=request.brush_feather
    )

    # Create new document version
    new_version = version_repo.create({
        "version_id": f"VER-{uuid.uuid4().hex[:8].upper()}",
        "document_id": request.document_id,
        "version_number": "v1.3",
        "parent_version_id": doc.get("current_version_id"),
        "version_type": "inpainted",
        "file_key": output_key,
        "mask_file_key": mask_key,
        "model_used": inpaint_result["engine"],
        "parameters": {
            "feather": request.brush_feather,
            "reconstructed_percentage": inpaint_result["reconstructed_percentage"]
        },
        "created_by": user["_id"],
        "notes": request.notes or "Selective inpainting applied via archival studio."
    })

    # Update document record
    doc_repo.update(request.document_id, {
        "restored_file_key": output_key,
        "current_version_id": new_version["version_id"],
        "status": "restored"
    })

    audit_repo.log(
        user_id=user["_id"],
        action="INPAINTING_APPLIED",
        resource_type="document",
        resource_id=request.document_id,
        metadata={"version_id": new_version["version_id"], "engine": inpaint_result["engine"]}
    )

    return {
        "success": True,
        "message": "Selective inpainting completed and version saved.",
        "version": new_version,
        "result": inpaint_result,
        "output_file_key": output_key
    }

