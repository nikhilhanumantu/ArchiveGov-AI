from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from backend.app.database.mongodb import get_db, sync_disk_if_mock

class ReviewRepository:
    def __init__(self):
        self.db = get_db()
        self.collection = self.db.reviews

    def create(self, review_data: Dict[str, Any]) -> Dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        review_data["created_at"] = review_data.get("created_at", now)
        
        result = self.collection.insert_one(review_data)
        review_data["_id"] = str(result.inserted_id)
        sync_disk_if_mock()
        return review_data

    def find_by_document_id(self, document_id: str) -> List[Dict[str, Any]]:
        cursor = self.collection.find({"document_id": document_id}).sort("created_at", -1)
        reviews = []
        for r in cursor:
            r["_id"] = str(r["_id"])
            reviews.append(r)
        return reviews

    def list_recent(self, limit: int = 20) -> List[Dict[str, Any]]:
        cursor = self.collection.find().sort("created_at", -1).limit(limit)
        reviews = []
        for r in cursor:
            r["_id"] = str(r["_id"])
            reviews.append(r)
        return reviews

