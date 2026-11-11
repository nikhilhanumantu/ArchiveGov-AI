from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from bson import ObjectId
from backend.app.database.mongodb import get_db, sync_disk_if_mock

class NotificationRepository:
    @property
    def collection(self):
        return get_db().notifications

    def create(self, notif_data: Dict[str, Any]) -> Dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        notif_data["created_at"] = notif_data.get("created_at", now)
        notif_data["is_read"] = notif_data.get("is_read", False)
        
        result = self.collection.insert_one(notif_data)
        notif_data["_id"] = str(result.inserted_id)
        sync_disk_if_mock()
        return notif_data

    def list_for_user(self, user_id: str, limit: int = 30) -> List[Dict[str, Any]]:
        cursor = self.collection.find({"$or": [{"user_id": user_id}, {"user_id": "all"}]}).sort("created_at", -1).limit(limit)
        notifs = []
        for n in cursor:
            n["_id"] = str(n["_id"])
            notifs.append(n)
        return notifs

    def count_unread(self, user_id: str) -> int:
        return self.collection.count_documents({
            "$or": [{"user_id": user_id}, {"user_id": "all"}],
            "is_read": False
        })

    def mark_as_read(self, notif_id: str) -> bool:
        try:
            oid = ObjectId(notif_id) if ObjectId.is_valid(notif_id) else notif_id
        except Exception:
            oid = notif_id
        res = self.collection.update_one({"_id": oid}, {"$set": {"is_read": True}})
        sync_disk_if_mock()
        return res.modified_count > 0

    def mark_all_as_read(self, user_id: str) -> int:
        res = self.collection.update_many(
            {"$or": [{"user_id": user_id}, {"user_id": "all"}], "is_read": False},
            {"$set": {"is_read": True}}
        )
        sync_disk_if_mock()
        return res.modified_count

