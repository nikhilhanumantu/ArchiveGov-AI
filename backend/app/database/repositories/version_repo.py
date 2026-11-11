from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from backend.app.database.mongodb import get_db, sync_disk_if_mock

class DocumentVersionRepository:
    @property
    def collection(self):
        return get_db().document_versions

    def create(self, version_data: Dict[str, Any]) -> Dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        version_data["created_at"] = version_data.get("created_at", now)
        
        result = self.collection.insert_one(version_data)
        version_data["_id"] = str(result.inserted_id)
        sync_disk_if_mock()
        return version_data

    def find_by_id(self, version_id: str) -> Optional[Dict[str, Any]]:
        v = self.collection.find_one({"version_id": version_id})
        if v and "_id" in v:
            v["_id"] = str(v["_id"])
        return v

    def find_by_document_id(self, document_id: str) -> List[Dict[str, Any]]:
        cursor = self.collection.find({"document_id": document_id}).sort("created_at", 1)
        versions = []
        for v in cursor:
            v["_id"] = str(v["_id"])
            versions.append(v)
        return versions

    def get_latest_version(self, document_id: str) -> Optional[Dict[str, Any]]:
        cursor = self.collection.find({"document_id": document_id}).sort("created_at", -1).limit(1)
        for v in cursor:
            v["_id"] = str(v["_id"])
            return v
        return None

