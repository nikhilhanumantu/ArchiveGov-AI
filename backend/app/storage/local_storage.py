import os
import uuid
import hashlib
import shutil
from pathlib import Path
from typing import Tuple, Optional
from fastapi import UploadFile, HTTPException
from backend.app.core.config import settings

class LocalStorage:
    def __init__(self):
        self.base_dir = Path(settings.UPLOAD_DIR)
        self.originals = Path(settings.ORIGINALS_DIR)
        self.working = Path(settings.WORKING_DIR)
        self.restored = Path(settings.RESTORED_DIR)
        self.masks = Path(settings.MASKS_DIR)
        self.exports = Path(settings.EXPORTS_DIR)
        
        for d in [self.originals, self.working, self.restored, self.masks, self.exports]:
            d.mkdir(parents=True, exist_ok=True)

    def save_upload(self, upload_file: UploadFile) -> Tuple[str, str, int, str]:
        """
        Saves uploaded file safely into originals.
        Returns (file_key, file_path, file_size, sha256_checksum)
        """
        ext = Path(upload_file.filename).suffix.lower()
        if ext not in settings.ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"Unsupported file type '{ext}'. Allowed: {settings.ALLOWED_EXTENSIONS}")
            
        file_key = f"{uuid.uuid4().hex}{ext}"
        destination = self.originals / file_key
        
        hasher = hashlib.sha256()
        size = 0
        
        with open(destination, "wb") as buffer:
            while chunk := upload_file.file.read(1024 * 64):
                size += len(chunk)
                if size > settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024:
                    destination.unlink(missing_ok=True)
                    raise HTTPException(status_code=400, detail=f"File exceeds maximum size of {settings.MAX_UPLOAD_SIZE_MB}MB")
                hasher.update(chunk)
                buffer.write(chunk)
                
        checksum = hasher.hexdigest()
        return file_key, str(destination), size, checksum

    def save_bytes(self, data: bytes, category: str = "working", extension: str = ".png") -> Tuple[str, str]:
        """Save raw bytes to a given category folder."""
        target_dir = {
            "originals": self.originals,
            "working": self.working,
            "restored": self.restored,
            "masks": self.masks,
            "exports": self.exports
        }.get(category, self.working)
        
        file_key = f"{uuid.uuid4().hex}{extension}"
        dest = target_dir / file_key
        with open(dest, "wb") as f:
            f.write(data)
        return file_key, str(dest)

    def resolve_path(self, file_key: str, category: Optional[str] = None) -> Optional[Path]:
        """Safely resolve file path preventing path traversal."""
        safe_name = Path(file_key).name
        if category:
            p = self.base_dir / category / safe_name
            if p.exists():
                return p
        # Check all categories
        for cat in ["originals", "restored", "working", "masks", "exports"]:
            p = self.base_dir / cat / safe_name
            if p.exists():
                return p
        return None

    def get_path(self, category: str, file_key: str) -> Path:
        """Return direct destination path for category and key."""
        target_dir = {
            "originals": self.originals,
            "working": self.working,
            "restored": self.restored,
            "masks": self.masks,
            "exports": self.exports
        }.get(category, self.originals)
        return target_dir / Path(file_key).name

    def copy_file(self, src_path: str, dest_category: str) -> Tuple[str, str]:
        ext = Path(src_path).suffix.lower()
        new_key = f"{uuid.uuid4().hex}{ext}"
        target_dir = self.base_dir / dest_category
        dest_path = target_dir / new_key
        shutil.copy2(src_path, dest_path)
        return new_key, str(dest_path)

storage = LocalStorage()

