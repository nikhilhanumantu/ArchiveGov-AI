from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from backend.app.database.mongodb import get_db, sync_disk_if_mock

class RAGChunkRepository:
    """Repository for indexed archival chunks (Module 1 Copilot)."""
    @property
    def collection(self):
        return get_db().rag_chunks

    def insert_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not chunks:
            return []
        now = datetime.now(timezone.utc).isoformat()
        for c in chunks:
            c["created_at"] = c.get("created_at", now)
        res = self.collection.insert_many(chunks)
        for i, inserted_id in enumerate(res.inserted_ids):
            chunks[i]["_id"] = str(inserted_id)
        sync_disk_if_mock()
        return chunks

    def find_by_doc(self, doc_id: str) -> List[Dict[str, Any]]:
        cursor = self.collection.find({"doc_id": doc_id}).sort("chunk_order", 1)
        results = []
        for c in cursor:
            c["_id"] = str(c["_id"])
            results.append(c)
        return results

    def search_all(self, department: Optional[str] = None) -> List[Dict[str, Any]]:
        query: Dict[str, Any] = {}
        if department and department != "all":
            query["department_id"] = department
        cursor = self.collection.find(query).sort("created_at", -1)
        results = []
        for c in cursor:
            c["_id"] = str(c["_id"])
            results.append(c)
        return results

    def delete_by_doc(self, doc_id: str) -> int:
        res = self.collection.delete_many({"doc_id": doc_id})
        sync_disk_if_mock()
        return res.deleted_count

    def count(self) -> int:
        return self.collection.count_documents({})
