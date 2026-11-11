import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import pymongo
from pymongo import MongoClient
import mongomock
from bson import json_util
from backend.app.core.config import settings

logger = logging.getLogger("archivegov.database")

_client = None
_db = None
_is_real_mongo = False
_mock_storage_path = Path(__file__).resolve().parent.parent.parent / "data" / "mongodb_fallback.json"

def init_db():
    global _client, _db, _is_real_mongo
    
    # Try real MongoDB first
    try:
        real_client = MongoClient(settings.MONGODB_URI, serverSelectionTimeoutMS=2500, connectTimeoutMS=2500, timeoutMS=3000)
        # Verify connection
        real_client.admin.command("ping")
        _client = real_client
        _db = _client[settings.DATABASE_NAME]
        _is_real_mongo = True
        logger.info(f"Connected successfully to real MongoDB at {settings.MONGODB_URI.split('@')[-1] if '@' in settings.MONGODB_URI else settings.MONGODB_URI}, DB: {settings.DATABASE_NAME}")
    except Exception as e:
        logger.warning(f"Could not connect to real MongoDB ({e}). Activating persistent embedded MongoDB engine.")
        _client = mongomock.MongoClient()
        _db = _client[settings.DATABASE_NAME]
        _is_real_mongo = False
        _load_mock_from_disk()

    # Create indexes
    create_indexes(_db)
    return _db

def get_db():
    global _db
    if _db is None:
        init_db()
    return _db

def get_db_status() -> Dict[str, Any]:
    global _db, _is_real_mongo
    if _db is None:
        init_db()
    
    status = {
        "engine": "MongoDB (Native Driver)" if _is_real_mongo else "MongoDB (Embedded Persistent Engine)",
        "is_connected": True,
        "is_real_mongodb": _is_real_mongo,
        "database_name": settings.DATABASE_NAME,
        "collections": {}
    }
    try:
        collections = _db.list_collection_names()
        for col_name in collections:
            status["collections"][col_name] = _db[col_name].count_documents({})
    except Exception as e:
        status["error"] = str(e)
    return status

def create_indexes(db):
    try:
        # Users indexes
        db.users.create_index([("email", pymongo.ASCENDING)], unique=True)
        db.users.create_index([("employee_id", pymongo.ASCENDING)], unique=True)
        
        # Documents indexes
        db.documents.create_index([("document_id", pymongo.ASCENDING)], unique=True)
        db.documents.create_index([("status", pymongo.ASCENDING)])
        db.documents.create_index([("created_at", pymongo.DESCENDING)])
        
        # Restoration jobs indexes
        db.restoration_jobs.create_index([("job_id", pymongo.ASCENDING)], unique=True)
        db.restoration_jobs.create_index([("document_id", pymongo.ASCENDING)])
        db.restoration_jobs.create_index([("status", pymongo.ASCENDING)])
        
        # Document versions indexes
        db.document_versions.create_index([("version_id", pymongo.ASCENDING)], unique=True)
        db.document_versions.create_index([("document_id", pymongo.ASCENDING), ("version_number", pymongo.ASCENDING)])
        
        # OCR results indexes
        db.ocr_results.create_index([("document_id", pymongo.ASCENDING)])
        
        # Reviews
        db.reviews.create_index([("document_id", pymongo.ASCENDING)])
        
        # Notifications
        db.notifications.create_index([("user_id", pymongo.ASCENDING), ("is_read", pymongo.ASCENDING)])
        
        # Audit logs
        db.audit_logs.create_index([("timestamp", pymongo.DESCENDING)])
        
        # Settings
        db.system_settings.create_index([("setting_key", pymongo.ASCENDING)], unique=True)
    except Exception as ex:
        logger.warning(f"Index creation note: {ex}")

def sync_disk_if_mock():
    """If using embedded engine, save state to disk for restart persistence."""
    global _db, _is_real_mongo
    if not _is_real_mongo and _db is not None:
        try:
            _mock_storage_path.parent.mkdir(parents=True, exist_ok=True)
            data = {}
            for col_name in _db.list_collection_names():
                data[col_name] = list(_db[col_name].find())
            with open(_mock_storage_path, "w", encoding="utf-8") as f:
                f.write(json_util.dumps(data))
        except Exception as e:
            logger.error(f"Failed to persist embedded database: {e}")

def _load_mock_from_disk():
    global _db
    if _mock_storage_path.exists():
        try:
            with open(_mock_storage_path, "r", encoding="utf-8") as f:
                data = json_util.loads(f.read())
                for col_name, docs in data.items():
                    if docs:
                        _db[col_name].insert_many(docs)
            logger.info("Restored persistent embedded database collections from disk.")
        except Exception as e:
            logger.warning(f"Could not restore embedded database from disk: {e}")

