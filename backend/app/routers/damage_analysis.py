from fastapi import APIRouter, Depends, HTTPException
from backend.app.core.dependencies import get_current_user
from backend.app.database.repositories.document_repo import DocumentRepository
from backend.app.storage.local_storage import storage
from backend.app.ai.damage_detection.damage_detector import damage_detector

router = APIRouter(prefix="/api/v1/damage-analysis", tags=["Damage Analysis"])
doc_repo = DocumentRepository()

@router.get("/{document_id}")
def get_damage_analysis(document_id: str, force_recompute: bool = False, user: dict = Depends(get_current_user)):
    doc = doc_repo.find_by_id(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Archival record not found.")

    # Return cached analysis if already run with current engine version and not forced
    cached = doc.get("damage_analysis")
    if cached and not force_recompute and cached.get("engine_version") == damage_detector.ENGINE_VERSION and cached.get("defects"):
        return cached

    file_key = doc.get("original_file_key")
    if not file_key:
        raise HTTPException(status_code=404, detail="Document has no original file asset.")
        
    file_path = storage.resolve_path(file_key)
    if not file_path or not file_path.exists():
        raise HTTPException(status_code=404, detail="Archival image file missing.")

    analysis = damage_detector.analyze_document(str(file_path))
    doc_repo.update(document_id, {"damage_analysis": analysis})
    return analysis

