from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Tuple
import re
from bson import ObjectId
from backend.app.database.mongodb import get_db, sync_disk_if_mock

class DocumentRepository:
    @property
    def collection(self):
        return get_db().documents

    def create(self, doc_data: Dict[str, Any]) -> Dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        doc_data["created_at"] = doc_data.get("created_at", now)
        doc_data["updated_at"] = now
        doc_data["status"] = doc_data.get("status", "uploaded")
        doc_data["is_deleted"] = False
        
        result = self.collection.insert_one(doc_data)
        doc_data["_id"] = str(result.inserted_id)
        sync_disk_if_mock()
        return doc_data

    def find_by_id(self, document_id: str) -> Optional[Dict[str, Any]]:
        doc = self.collection.find_one({"document_id": document_id, "is_deleted": {"$ne": True}})
        if not doc:
            try:
                oid = ObjectId(document_id) if ObjectId.is_valid(document_id) else None
                if oid:
                    doc = self.collection.find_one({"_id": oid, "is_deleted": {"$ne": True}})
            except Exception:
                pass
        if doc and "_id" in doc:
            doc["_id"] = str(doc["_id"])
        return doc

    def find_by_checksum(self, checksum: str) -> Optional[Dict[str, Any]]:
        doc = self.collection.find_one({"checksum": checksum, "is_deleted": {"$ne": True}})
        if doc and "_id" in doc:
            doc["_id"] = str(doc["_id"])
        return doc

    def update(self, document_id: str, update_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        update_data["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.collection.update_one(
            {"document_id": document_id},
            {"$set": update_data}
        )
        sync_disk_if_mock()
        return self.find_by_id(document_id)

    def list_documents(
        self,
        query: Optional[str] = None,
        department: Optional[str] = None,
        category: Optional[str] = None,
        status: Optional[str] = None,
        skip: int = 0,
        limit: int = 50,
        sort_by: str = "created_at",
        sort_dir: int = -1
    ) -> Tuple[List[Dict[str, Any]], int]:
        filters: Dict[str, Any] = {"is_deleted": {"$ne": True}}
        
        if department and department != "all":
            filters["department_id"] = department
        if category and category != "all":
            filters["category"] = category
        if status and status != "all":
            filters["status"] = status
            
        if query and query.strip():
            regex = re.compile(re.escape(query.strip()), re.IGNORECASE)
            filters["$or"] = [
                {"title": regex},
                {"document_id": regex},
                {"archive_reference_id": regex},
                {"description": regex},
                {"ocr_text": regex}
            ]
            
        total = self.collection.count_documents(filters)
        cursor = self.collection.find(filters).sort(sort_by, sort_dir).skip(skip).limit(limit)
        
        results = []
        for d in cursor:
            d["_id"] = str(d["_id"])
            results.append(d)
        return results, total

    def list_all(self, limit: int = 50) -> List[Dict[str, Any]]:
        docs, _ = self.list_documents(limit=limit)
        return docs

    def count_by_status(self) -> Dict[str, int]:
        counts = {
            "total": self.collection.count_documents({"is_deleted": {"$ne": True}}),
            "uploaded": self.collection.count_documents({"status": "uploaded", "is_deleted": {"$ne": True}}),
            "processing": self.collection.count_documents({"status": "processing", "is_deleted": {"$ne": True}}),
            "restored": self.collection.count_documents({"status": "restored", "is_deleted": {"$ne": True}}),
            "pending_review": self.collection.count_documents({"status": "pending_review", "is_deleted": {"$ne": True}}),
            "approved": self.collection.count_documents({"status": "approved", "is_deleted": {"$ne": True}}),
            "archived": self.collection.count_documents({"status": "archived", "is_deleted": {"$ne": True}}),
            "rejected": self.collection.count_documents({"status": "rejected", "is_deleted": {"$ne": True}})
        }
        return counts

    def soft_delete(self, document_id: str) -> bool:
        res = self.collection.update_one(
            {"document_id": document_id},
            {"$set": {"is_deleted": True, "updated_at": datetime.now(timezone.utc).isoformat()}}
        )
        sync_disk_if_mock()
        return res.modified_count > 0

