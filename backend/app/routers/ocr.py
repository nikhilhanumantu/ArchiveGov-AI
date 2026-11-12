from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from backend.app.schemas.ocr import OCRSaveRequest, OCRExtractRequest
from backend.app.core.dependencies import get_current_user, require_permission
from backend.app.core.permissions import Permission
from backend.app.database.repositories.document_repo import DocumentRepository
from backend.app.database.repositories.ocr_repo import OCRRepository
from backend.app.database.repositories.audit_repo import AuditRepository
from backend.app.storage.local_storage import storage
from backend.app.ai.ocr.paddleocr_adapter import paddleocr_adapter
from backend.app.ai.ocr.ocr_processor import ocr_exporter

router = APIRouter(prefix="/api/v1/ocr", tags=["OCR & Text Extraction"])
doc_repo = DocumentRepository()
ocr_repo = OCRRepository()
audit_repo = AuditRepository()

@router.post("/extract")
def extract_ocr(request: OCRExtractRequest, user: dict = Depends(require_permission(Permission.OCR_RUN))):
    return get_ocr_data(
        document_id=request.document_id,
        use_restored=bool(request.use_restored),
        force_rescan=bool(request.force_rescan),
        user=user
    )

@router.get("/{document_id}")
def get_ocr_data(document_id: str, use_restored: bool = True, force_rescan: bool = False, user: dict = Depends(require_permission(Permission.OCR_RUN))):
    doc = doc_repo.find_by_id(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Archival document not found.")

    # Check if already processed (unless force_rescan is requested)
    if not force_rescan:
        saved = ocr_repo.find_by_document_id(document_id)
        if saved and saved.get("extracted_text"):
            return saved

    # Execute OCR
    target_key = doc.get("restored_file_key") if (use_restored and doc.get("restored_file_key")) else doc["original_file_key"]
    file_path = storage.resolve_path(target_key)
    if not file_path or not file_path.exists():
        raise HTTPException(status_code=404, detail="Document image file not found.")

    ocr_result = paddleocr_adapter.extract_text(str(file_path), lang=doc.get("language", "en"))
    
    # Save to MongoDB
    saved = ocr_repo.save_result({
        "document_id": document_id,
        "version_id": doc.get("current_version_id", "v1.0"),
        "extracted_text": ocr_result["extracted_text"],
        "corrected_text": ocr_result["extracted_text"],
        "language": ocr_result["language"],
        "text_regions": ocr_result["regions"],
        "regions": ocr_result["regions"],
        "average_confidence": ocr_result["average_confidence"],
        "model_used": ocr_result["engine"],
        "engine": ocr_result["engine"],
        "word_count": ocr_result.get("word_count", 0),
        "line_count": ocr_result.get("line_count", len(ocr_result["regions"]))
    })
    
    # Update search index on document
    doc_repo.update(document_id, {"ocr_text": ocr_result["extracted_text"]})
    return saved

@router.post("/{document_id}/save")
def save_corrected_transcription(document_id: str, request: OCRSaveRequest, user: dict = Depends(get_current_user)):
    doc = doc_repo.find_by_id(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    updated = ocr_repo.update_corrected_text(document_id, request.corrected_text)
    doc_repo.update(document_id, {"ocr_text": request.corrected_text})
    
    audit_repo.log(
        user_id=user["_id"],
        action="OCR_TRANSCRIPTION_CORRECTED",
        resource_type="document",
        resource_id=document_id
    )

    return {
        "success": True,
        "message": "Archival transcription updated and verified.",
        "ocr_result": updated
    }

@router.get("/{document_id}/export/{format_type}")
def export_ocr_file(document_id: str, format_type: str, user: dict = Depends(get_current_user)):
    doc = doc_repo.find_by_id(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    ocr_record = ocr_repo.find_by_document_id(document_id)
    if not ocr_record:
        # Run OCR first if not found
        get_ocr_data(document_id, True, user)
        ocr_record = ocr_repo.find_by_document_id(document_id)

    text_to_export = ocr_record.get("corrected_text") or ocr_record.get("extracted_text", "")
    title = doc.get("title", "Government Archival Record")

    format_type = format_type.lower()
    if format_type == "txt":
        file_key = ocr_exporter.export_txt(document_id, title, text_to_export)
        media_type = "text/plain"
    elif format_type == "json":
        file_key = ocr_exporter.export_json(document_id, title, ocr_record)
        media_type = "application/json"
    elif format_type == "docx":
        file_key = ocr_exporter.export_docx(document_id, title, text_to_export)
        media_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    elif format_type == "pdf":
        file_key = ocr_exporter.export_pdf(document_id, title, text_to_export)
        media_type = "application/pdf"
    else:
        raise HTTPException(status_code=400, detail="Invalid export format. Supported: txt, docx, json, pdf")

    file_path = storage.resolve_path(file_key, "exports")
    return FileResponse(path=str(file_path), media_type=media_type, filename=Path(file_path).name)

