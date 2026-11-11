from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from backend.app.database.mongodb import get_db, sync_disk_if_mock

class AuditRepository:
    @property
    def collection(self):
        return get_db().audit_logs

    def log(self, user_id: str, action: str, resource_type: str, resource_id: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        entry = {
            "user_id": user_id,
            "action": action,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "metadata": metadata or {},
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        res = self.collection.insert_one(entry)
        entry["_id"] = str(res.inserted_id)
        sync_disk_if_mock()
        return entry

    def list_logs(self, limit: int = 50, skip: int = 0) -> List[Dict[str, Any]]:
        cursor = self.collection.find().sort("timestamp", -1).skip(skip).limit(limit)
        logs = []
        for l in cursor:
            l["_id"] = str(l["_id"])
            logs.append(l)
        return logs

