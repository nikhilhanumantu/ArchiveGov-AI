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
    def generate_csv():
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Headers
        writer.writerow([
            "Document ID", "Title", "Archive Reference ID", "Department",
            "Category", "Original Date", "Language", "Status", "Created At"
        ])
        yield output.getvalue()
        output.seek(0)
        output.truncate(0)
        
        page_size = 500
        skip = 0
        while True:
            # Stable pagination iterates through full result set beyond 1,000 records
            batch, total = doc_repo.list_documents(
                skip=skip,
                limit=page_size,
                sort_by="created_at",
                sort_dir=-1
            )
            if not batch:
                break
                
            for d in batch:
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
            yield output.getvalue()
            output.seek(0)
            output.truncate(0)
            
            skip += len(batch)
            if skip >= total or len(batch) < page_size:
                break

    return StreamingResponse(
        generate_csv(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=archivegov_records.csv"}
    )

