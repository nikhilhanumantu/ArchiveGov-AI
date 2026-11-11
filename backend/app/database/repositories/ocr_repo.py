from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from backend.app.database.mongodb import get_db, sync_disk_if_mock

class OCRRepository:
    @property
    def collection(self):
        return get_db().ocr_results

    def save_result(self, ocr_data: Dict[str, Any]) -> Dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        ocr_data["created_at"] = ocr_data.get("created_at", now)
        ocr_data["updated_at"] = now
        ocr_data["corrected_text"] = ocr_data.get("corrected_text", ocr_data.get("extracted_text", ""))
        
        # Replace existing or insert
        doc_id = ocr_data["document_id"]
        existing = self.collection.find_one({"document_id": doc_id})
        if existing:
            self.collection.update_one({"_id": existing["_id"]}, {"$set": ocr_data})
            ocr_data["_id"] = str(existing["_id"])
        else:
            result = self.collection.insert_one(ocr_data)
            ocr_data["_id"] = str(result.inserted_id)
            
        sync_disk_if_mock()
        return ocr_data

    def find_by_document_id(self, document_id: str) -> Optional[Dict[str, Any]]:
        res = self.collection.find_one({"document_id": document_id})
        if res and "_id" in res:
            res["_id"] = str(res["_id"])
        return res

    def update_corrected_text(self, document_id: str, corrected_text: str) -> Optional[Dict[str, Any]]:
        now = datetime.now(timezone.utc).isoformat()
        self.collection.update_one(
            {"document_id": document_id},
            {"$set": {"corrected_text": corrected_text, "updated_at": now}}
        )
        sync_disk_if_mock()
        return self.find_by_document_id(document_id)

