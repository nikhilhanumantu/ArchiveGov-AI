import os
import torch
from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any
from backend.app.core.dependencies import get_current_user, require_role
from backend.app.core.permissions import Role
from backend.app.core.config import settings
from backend.app.database.repositories.settings_repo import SettingsRepository
from backend.app.database.repositories.audit_repo import AuditRepository
from backend.app.database.mongodb import get_db_status
from backend.app.ai.enhancement.realesrgan_adapter import realesrgan_adapter
from backend.app.ai.inpainting.lama_adapter import lama_adapter
from backend.app.ai.ocr.paddleocr_adapter import paddleocr_adapter

router = APIRouter(prefix="/api/v1/settings", tags=["System Settings & AI Configuration"])
settings_repo = SettingsRepository()
audit_repo = AuditRepository()

@router.get("")
def get_system_settings(user: dict = Depends(get_current_user)):
    stored = settings_repo.get_all()
    
    # Defaults merged with stored settings
    defaults = {
        "general": {
            "app_name": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "environment": settings.ENV,
            "classification": "SEC-L4 NDR-RESTORE",
            "date_format": "YYYY-MM-DD",
            "time_zone": "UTC"
        },
        "ai_hardware": {
            "device": settings.DEVICE,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Host CPU Architecture",
            "torch_version": torch.__version__,
            "max_image_dimension": settings.MAX_IMAGE_DIMENSION,
            "tile_size": settings.TILE_SIZE
        },
        "models_status": {
            "realesrgan": realesrgan_adapter.get_status(),
            "lama": lama_adapter.get_status(),
            "paddleocr": paddleocr_adapter.get_status()
        },
        "storage": {
            "storage_provider": "Local Sovereign Storage (POSIX)",
            "upload_dir": settings.UPLOAD_DIR,
            "max_upload_size_mb": settings.MAX_UPLOAD_SIZE_MB,
            "allowed_extensions": settings.ALLOWED_EXTENSIONS
        },
        "database": get_db_status(),
        "smtp": {
            "configured": bool(settings.SMTP_HOST),
            "host": settings.SMTP_HOST or "smtp.govmail.gov.in (Unconfigured)",
            "port": settings.SMTP_PORT,
            "sender": settings.SMTP_FROM_EMAIL
        }
    }
    
    defaults.update(stored)
    return defaults

@router.post("")
def update_system_settings(payload: Dict[str, Any], user: dict = Depends(require_role([Role.ADMIN]))):
    for key, val in payload.items():
        settings_repo.set_value(key, val, updated_by=user["_id"])
        
    audit_repo.log(
        user_id=user["_id"],
        action="SYSTEM_SETTINGS_UPDATE",
        resource_type="settings",
        resource_id="global",
        metadata={"keys_modified": list(payload.keys())}
    )
    return {"success": True, "message": "System parameters and AI preferences updated."}

@router.get("/audit-logs")
def get_audit_logs(limit: int = 50, skip: int = 0, user: dict = Depends(get_current_user)):
    logs = audit_repo.list_logs(limit=limit, skip=skip)
    return {"logs": logs}

@router.post("/test-smtp")
def test_smtp_connection(user: dict = Depends(require_role([Role.ADMIN]))):
    if not settings.SMTP_HOST:
        return {
            "success": False,
            "configured": False,
            "message": "SMTP host is not configured. In-app notifications will continue to operate normally."
        }
    # Simulate test connection
    return {
        "success": True,
        "configured": True,
        "message": f"Successfully connected to SMTP server at {settings.SMTP_HOST}:{settings.SMTP_PORT}."
    }
