from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import FileResponse
from pathlib import Path
from backend.app.storage.local_storage import storage
import mimetypes

router = APIRouter(prefix="/api/v1/files", tags=["File Storage"])

@router.get("/{file_key}")
def get_file(file_key: str):
    file_path = storage.resolve_path(file_key)
    if not file_path or not file_path.exists():
        raise HTTPException(status_code=404, detail="Requested archival asset not found.")
        
    mime_type, _ = mimetypes.guess_type(str(file_path))
    if not mime_type:
        mime_type = "application/octet-stream"
        
    return FileResponse(
        path=str(file_path),
        media_type=mime_type,
        filename=file_path.name
    )

