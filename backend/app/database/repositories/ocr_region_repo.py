from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from backend.app.database.mongodb import get_db, sync_disk_if_mock

class OCRRegionRepository:
    """Repository for granular OCR bounding boxes and layout regions (Module 4 Lens)."""
    @property
    def collection(self):
        return get_db().ocr_regions

    def insert_many(self, regions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not regions:
            return []
        now = datetime.now(timezone.utc).isoformat()
        for r in regions:
            r["created_at"] = r.get("created_at", now)
        res = self.collection.insert_many(regions)
        for i, inserted_id in enumerate(res.inserted_ids):
            regions[i]["_id"] = str(inserted_id)
        sync_disk_if_mock()
        return regions

    def find_by_doc(
        self,
        doc_id: str,
        page_number: Optional[int] = None,
        granularity: Optional[str] = None,
        version_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        query: Dict[str, Any] = {"doc_id": doc_id}
        if version_id is not None:
            query["version_id"] = version_id
        if page_number is not None:
            query["page_number"] = page_number
        if granularity:
            query["granularity"] = granularity
        cursor = self.collection.find(query).sort([("page_number", 1), ("bbox_pixels.1", 1), ("bbox_pixels.0", 1)])
        results = []
        for doc in cursor:
            doc["_id"] = str(doc["_id"])
            results.append(doc)
        return results

    def find_by_region_id(self, doc_id: str, region_id: str, version_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        query = {"doc_id": doc_id, "region_id": region_id}
        if version_id is not None:
            query["version_id"] = version_id
        doc = self.collection.find_one(query)
        if doc and "_id" in doc:
            doc["_id"] = str(doc["_id"])
        return doc

    def delete_by_doc(self, doc_id: str) -> int:
        res = self.collection.delete_many({"doc_id": doc_id})
        sync_disk_if_mock()
        return res.deleted_count
