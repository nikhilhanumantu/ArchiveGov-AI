import uuid
import pypdf
import shutil
import logging
from pathlib import Path
from typing import Optional, List
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, status, BackgroundTasks
from backend.app.core.dependencies import get_current_user, require_permission
from backend.app.core.permissions import Permission
from backend.app.storage.local_storage import storage
from backend.app.database.repositories.document_repo import DocumentRepository
from backend.app.database.repositories.version_repo import DocumentVersionRepository
from backend.app.database.repositories.audit_repo import AuditRepository
from backend.app.database.repositories.notification_repo import NotificationRepository
from backend.app.database.repositories.ocr_repo import OCRRepository
from backend.app.schemas.documents import DocumentUpdateInput
from backend.app.ai.preprocessing.opencv_processor import opencv_processor
from backend.app.ai.ocr.paddleocr_adapter import paddleocr_adapter

logger = logging.getLogger("archivegov.documents")

router = APIRouter(prefix="/api/v1/documents", tags=["Document Management"])
doc_repo = DocumentRepository()
version_repo = DocumentVersionRepository()
audit_repo = AuditRepository()
notif_repo = NotificationRepository()
ocr_repo = OCRRepository()

@router.post("/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: str = Form(...),
    department_id: str = Form("Land Revenue & Settlement"),
    category: str = Form("Historical Land Records"),
    original_date: str = Form("1912"),
    language: str = Form("English & Tamil Vernacular"),
    archive_reference_id: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    user: dict = Depends(require_permission(Permission.DOCUMENT_CREATE))
):
    # 1. Save uploaded file to originals
    file_key, file_path, file_size, checksum = storage.save_upload(file)
    
    # Check for PDF and rasterize page 1 to high-resolution PNG
    is_pdf = Path(file_key).suffix.lower() == ".pdf" or (bool(file.content_type) and "pdf" in file.content_type.lower())
    pdf_file_key = None
    if is_pdf:
        pdf_file_key = file_key
        rasterized_stem = Path(file_key).stem
        rasterized_key = f"{rasterized_stem}_page1.png"
        rasterized_path = storage.originals / rasterized_key
        try:
            page1_img = opencv_processor.load_image(str(file_path))
            opencv_processor.save_image(page1_img, str(rasterized_path))
            file_key = rasterized_key
            file_path = rasterized_path
        except Exception as e:
            logger.warning(f"PDF rasterization exception: {e}")
    
    # Check for duplicate
    existing_duplicate = doc_repo.find_by_checksum(checksum)
    is_duplicate = existing_duplicate is not None
    
    # 2. Generate unique Document ID
    doc_id = f"DOC-{uuid.uuid4().hex[:8].upper()}"
    ref_id = archive_reference_id if archive_reference_id and archive_reference_id.strip() else f"REC-{original_date}-{doc_id[-4:]}"
    
    # 3. Fast high-contrast contrast enhancement for instant preview (under 100ms)
    restored_file_key = f"restored_{file_key}"
    restored_path = storage.restored / restored_file_key
    try:
        opencv_processor.process_pipeline(
            input_path=str(file_path),
            output_path=str(restored_path),
            do_deskew=True,
            do_clahe=True,
            do_denoise=False
        )
    except Exception as e:
        logger.warning(f"Initial OpenCV restoration pipeline note: {e}")
        shutil.copy2(str(file_path), str(restored_path))

    # 4. Create Document Record in MongoDB immediately
    doc_data = {
        "document_id": doc_id,
        "title": title.strip(),
        "department_id": department_id,
        "category": category,
        "original_date": original_date,
        "language": language,
        "archive_reference_id": ref_id,
        "description": description or f"Archival government folio ingested for AI restoration.",
        "original_file_key": file_key,
        "restored_file_key": restored_file_key,
        "original_filename": file.filename,
        "file_size": file_size,
        "checksum": checksum,
        "is_duplicate_notice": is_duplicate,
        "current_version_id": "v1.1",
        "status": "restored",
        "ocr_text": "",
        "uploaded_by": user["_id"],
        "uploader_name": user.get("name", "Archive Officer"),
        "assigned_reviewer": None
    }
    if pdf_file_key:
        doc_data["pdf_file_key"] = pdf_file_key
    created_doc = doc_repo.create(doc_data)
    
    # 5. Create Versions V1.0 (Original Upload) and V1.1 (AI Restored)
    v1_0 = version_repo.create({
        "version_id": f"VER-{uuid.uuid4().hex[:8].upper()}",
        "document_id": doc_id,
        "version_number": "v1.0",
        "parent_version_id": None,
        "version_type": "original",
        "file_key": file_key,
        "pdf_file_key": pdf_file_key,
        "model_used": "Raw Archival Scan (Unmodified Ingestion)",
        "parameters": {"checksum": checksum, "file_size_bytes": file_size},
        "created_by": user["_id"],
        "notes": "Immutable original master archival upload with page 1 rasterization." if pdf_file_key else "Immutable original master archival upload."
    })

    v1_1 = version_repo.create({
        "version_id": f"VER-{uuid.uuid4().hex[:8].upper()}",
        "document_id": doc_id,
        "version_number": "v1.1",
        "parent_version_id": v1_0["version_id"],
        "version_type": "restored",
        "file_key": restored_file_key,
        "model_used": "OpenCV Sovereign Preprocessing & Contrast Enhancement",
        "parameters": {"clahe": True, "denoise": False, "deskew": True},
        "created_by": user["_id"],
        "notes": "Automated initial contrast enhancement and text deskewing on ingestion."
    })
    
    doc_repo.update(doc_id, {"current_version_id": v1_1["version_id"]})

    # 6. Execute OCR in the background so upload returns in under 500ms
    def _async_ingest_ocr(target_img_path: str, lang: str, target_doc_id: str, ver_id: str):
        try:
            ocr_res = paddleocr_adapter.extract_text(target_img_path, lang=lang)
            extracted_text = ocr_res.get("extracted_text", "")
            text_regions = ocr_res.get("regions", [])
            avg_conf = ocr_res.get("average_confidence", 0.0)
            engine_name = ocr_res.get("engine", "PaddleOCR Multilingual Engine")

            ocr_repo.save_result({
                "document_id": target_doc_id,
                "version_id": ver_id,
                "extracted_text": extracted_text,
                "corrected_text": extracted_text,
                "language": lang,
                "text_regions": text_regions,
                "regions": text_regions,
                "average_confidence": avg_conf,
                "model_used": engine_name,
                "engine": engine_name,
                "word_count": len(extracted_text.split()) if extracted_text else 0,
                "line_count": len(text_regions)
            })
            doc_repo.update(target_doc_id, {"ocr_text": extracted_text})
            logger.info(f"Background ingestion OCR completed for {target_doc_id}: {len(text_regions)} lines extracted.")
        except Exception as ocr_err:
            logger.warning(f"Background ingestion OCR error for {target_doc_id}: {ocr_err}")

    ocr_target = str(restored_path) if restored_path.exists() else str(file_path)
    background_tasks.add_task(_async_ingest_ocr, ocr_target, language, doc_id, v1_1["version_id"])

    # 7. Audit Log & Notification
    audit_repo.log(
        user_id=user["_id"],
        action="DOCUMENT_UPLOAD",
        resource_type="document",
        resource_id=doc_id,
        metadata={"filename": file.filename, "size": file_size}
    )
    
    notif_repo.create({
        "user_id": user["_id"],
        "type": "document_uploaded",
        "title": "Document Ingested",
        "message": f"Historical record '{title}' successfully registered into sovereign vault.",
        "related_document_id": doc_id
    })

    return {
        "success": True,
        "message": "Archival document ingested successfully.",
        "document": created_doc,
        "redirect_url": f"/app/restoration?document_id={doc_id}"
    }

@router.get("")
def list_documents(
    query: Optional[str] = None,
    department: Optional[str] = None,
    category: Optional[str] = None,
    status: Optional[str] = None,
    skip: int = 0,
    limit: int = 50,
    user: dict = Depends(get_current_user)
):
    docs, total = doc_repo.list_documents(
        query=query,
        department=department,
        category=category,
        status=status,
        skip=skip,
        limit=limit
    )
    return {
        "documents": docs,
        "total": total,
        "skip": skip,
        "limit": limit
    }

@router.get("/{document_id}")
def get_document(document_id: str, user: dict = Depends(get_current_user)):
    doc = doc_repo.find_by_id(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Archival record not found.")
    return doc

@router.put("/{document_id}")
def update_document(document_id: str, input_data: DocumentUpdateInput, user: dict = Depends(get_current_user)):
    doc = doc_repo.find_by_id(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Archival record not found.")
        
    update_dict = {k: v for k, v in input_data.dict().items() if v is not None}
    updated = doc_repo.update(document_id, update_dict)
    
    audit_repo.log(
        user_id=user["_id"],
        action="DOCUMENT_METADATA_UPDATE",
        resource_type="document",
        resource_id=document_id,
        metadata=update_dict
    )
    return updated

@router.delete("/{document_id}")
def delete_document(document_id: str, user: dict = Depends(require_permission(Permission.DOCUMENT_DELETE))):
    from bson import ObjectId
    doc = doc_repo.find_by_id(document_id)
    if not doc:
        doc = doc_repo.collection.find_one({"document_id": document_id})
        if not doc and ObjectId.is_valid(document_id):
            doc = doc_repo.collection.find_one({"_id": ObjectId(document_id)})
            
    if not doc:
        raise HTTPException(status_code=404, detail="Archival record not found.")

    doc_id = doc.get("document_id", document_id)
    title = doc.get("title", doc_id)

    # Keep the record and its version keys until all physical cleanup succeeds.
    keys = {doc.get(name) for name in ("pdf_file_key", "original_file_key", "restored_file_key")}
    keys.update(v.get("file_key") for v in version_repo.find_by_document_id(doc_id))
    for key in filter(None, keys):
        path = storage.resolve_path(key)
        if path:
            path.unlink(missing_ok=True)
    if Path(doc_id).name != doc_id or doc_id in {".", ".."}:
        raise HTTPException(status_code=400, detail="Invalid document identifier.")
    stages_dir = storage.stages / doc_id
    if stages_dir.is_dir():
        shutil.rmtree(stages_dir)

    if not doc_repo.delete(doc_id):
        raise HTTPException(status_code=404, detail="Archival record no longer exists.")

    audit_repo.log(
        user_id=user["_id"],
        action="DOCUMENT_DELETED",
        resource_type="document",
        resource_id=doc_id,
        metadata={"title": title, "deleted_by": user.get("name")}
    )
    return {
        "success": True,
        "message": f"Archival record '{title}' ({doc_id}) successfully deleted from database.",
        "deleted_document_id": doc_id
    }

