import csv
import io
from typing import Optional
from fastapi import APIRouter, Depends, Response
from fastapi.responses import StreamingResponse
from backend.app.core.dependencies import get_current_user
from backend.app.database.repositories.document_repo import DocumentRepository

router = APIRouter(prefix="/api/v1/archive", tags=["Digital Archive"])
doc_repo = DocumentRepository()

@router.get("/search")
def search_archive(
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
    counts = doc_repo.count_by_status()
    return {
        "documents": docs,
        "total": total,
        "counts": counts,
        "skip": skip,
        "limit": limit
    }

@router.get("/export-csv")
def export_archive_csv(user: dict = Depends(get_current_user)):
    docs, _ = doc_repo.list_documents(limit=1000)
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Headers
    writer.writerow([
        "Document ID", "Title", "Archive Reference ID", "Department",
        "Category", "Original Date", "Language", "Status", "Created At"
    ])
    
    for d in docs:
        writer.writerow([
            d.get("document_id", ""),
            d.get("title", ""),
            d.get("archive_reference_id", ""),
            d.get("department_id", ""),
            d.get("category", ""),
            d.get("original_date", ""),
            d.get("language", ""),
            d.get("status", ""),
            d.get("created_at", "")
        ])
        
    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=archivegov_records.csv"}
    )

