import os
from pathlib import Path
from typing import List, Optional
from pydantic import Field, AliasChoices
from pydantic_settings import BaseSettings

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
BACKEND_DIR = BASE_DIR / "backend"
UPLOADS_DIR = BASE_DIR / "uploads"
FRONTEND_DIR = BASE_DIR / "frontend"
MODELS_DIR = BASE_DIR / "ai_models"

class Settings(BaseSettings):
    APP_NAME: str = "ArchiveGov AI"
    APP_VERSION: str = "2.4.0"
    TAGLINE: str = "Reviving Government History with Generative AI"
    ENV: str = "development"
    DEBUG: bool = True
    
    # Server
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    
    # Security
    SECRET_KEY: str = "archivegov-secret-key-super-secure-production-ready-2024"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24 hours
    COOKIE_NAME: str = "archivegov_session"
    
    # Database (MongoDB)
    MONGODB_URI: str = Field(
        default="mongodb://127.0.0.1:27017",
        validation_alias=AliasChoices("MONGODB_URI", "MONGO_URI")
    )
    DATABASE_NAME: str = Field(
        default="archivegov_db",
        validation_alias=AliasChoices("DATABASE_NAME", "MONGO_DB_NAME")
    )
    
    # Storage
    UPLOAD_DIR: str = str(UPLOADS_DIR)
    ORIGINALS_DIR: str = str(UPLOADS_DIR / "originals")
    WORKING_DIR: str = str(UPLOADS_DIR / "working")
    RESTORED_DIR: str = str(UPLOADS_DIR / "restored")
    MASKS_DIR: str = str(UPLOADS_DIR / "masks")
    EXPORTS_DIR: str = str(UPLOADS_DIR / "exports")
    MAX_UPLOAD_SIZE_MB: int = 50
    ALLOWED_EXTENSIONS: List[str] = [".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".pdf"]
    
    # AI & Compute
    DEVICE: str = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    ENABLE_REALESRGAN: bool = True
    ENABLE_GFPGAN: bool = True
    GFPGAN_MODEL_PATH: str = "ai_models/gfpgan/GFPGANv1.4.pth"
    ENABLE_FINEGRAIN_ENHANCER: bool = True
    FINEGRAIN_SPACE_ID: str = "finegrain/finegrain-image-enhancer"
    HF_TOKEN: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("HF_TOKEN", "HUGGINGFACE_API_TOKEN")
    )
    
    # OpenRouter Google Gemini Nano Banana 2.1
    OPENROUTER_API_KEY: str = ""
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    OPENROUTER_IMAGE_MODEL: str = "google/gemini-nano-banana-2.1"
    
    # Google AI Studio / Gemini API
    GOOGLE_AI_STUDIO_API_KEY: str = ""
    GOOGLE_AI_STUDIO_BASE_URL: str = "https://generativelanguage.googleapis.com/v1beta"
    
    ENABLE_LAMA: bool = True
    ENABLE_PADDLEOCR: bool = True
    TILE_SIZE: int = 512
    MAX_IMAGE_DIMENSION: int = 4096
    
    # SMTP / Notifications
    SMTP_HOST: Optional[str] = None
    SMTP_PORT: int = 587
    SMTP_USERNAME: Optional[str] = None
    SMTP_PASSWORD: Optional[str] = None
    SMTP_FROM_EMAIL: str = "noreply@archivegov.ai"
    SMTP_USE_TLS: bool = True
    
    # CORS
    CORS_ORIGINS: List[str] = ["*"]

    class Config:
        env_file = ".env"
        extra = "allow"

settings = Settings()

# Ensure directories exist
for p in [settings.ORIGINALS_DIR, settings.WORKING_DIR, settings.RESTORED_DIR, settings.MASKS_DIR, settings.EXPORTS_DIR]:
    Path(p).mkdir(parents=True, exist_ok=True)

