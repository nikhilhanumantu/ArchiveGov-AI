from datetime import datetime, timezone
from typing import Optional, Dict, Any
from pymongo import ReturnDocument
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
        
        key = {"document_id": ocr_data["document_id"], "version_id": ocr_data.get("version_id")}
        saved = self.collection.find_one_and_update(
            key, {"$set": ocr_data}, upsert=True, return_document=ReturnDocument.AFTER
        )
        ocr_data["_id"] = str(saved["_id"])
        sync_disk_if_mock()
        return ocr_data

    def find_by_document_id(self, document_id: str, version_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        doc = get_db().documents.find_one({"document_id": document_id}) or {}
        if version_id is None:
            version_id = doc.get("current_version_id")
        res = self.collection.find_one({"document_id": document_id, "version_id": version_id})
        if res is None:
            # Legacy OCR rows predate per-run IDs. Only use rows without a version
            # or the matching legacy release label on an older master.
            version = get_db().document_versions.find_one({"document_id": document_id, "version_id": version_id}) or {}
            legacy_labels = [None]
            legacy_master = version.get("file_key") == "restored_" + (doc.get("original_file_key") or "")
            if version.get("version_number") == "v2.0" or (legacy_master and version.get("version_number") != "v1.1"):
                legacy_labels.append("v2.0")
            res = self.collection.find_one({"document_id": document_id, "version_id": {"$in": legacy_labels}})
        if res and "_id" in res:
            res["_id"] = str(res["_id"])
        return res

    def update_corrected_text(self, document_id: str, corrected_text: str) -> Optional[Dict[str, Any]]:
        now = datetime.now(timezone.utc).isoformat()
        record = self.find_by_document_id(document_id)
        if record is None:
            return None
        self.collection.update_one(
            {"document_id": document_id, "version_id": record.get("version_id")},
            {"$set": {"corrected_text": corrected_text, "updated_at": now}}
        )
        sync_disk_if_mock()
        return self.find_by_document_id(document_id)

