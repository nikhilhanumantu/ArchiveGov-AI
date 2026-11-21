from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from backend.app.database.mongodb import get_db, sync_disk_if_mock

class RestorationStageRepository:
    """Repository for managing granular restoration pipeline stages (Module 3 Time Machine)."""
    @property
    def collection(self):
        return get_db().restoration_stages

    def create(self, stage_data: Dict[str, Any]) -> Dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        stage_data["created_at"] = stage_data.get("created_at", now)
        result = self.collection.insert_one(stage_data)
        stage_data["_id"] = str(result.inserted_id)
        sync_disk_if_mock()
        return stage_data

    def find_by_doc(self, doc_id: str, version_id: Optional[str] = None) -> List[Dict[str, Any]]:
        query: Dict[str, Any] = {"doc_id": doc_id}
        if version_id:
            query["version_id"] = version_id
        cursor = self.collection.find(query).sort("stage_order", 1)
        stages = []
        for s in cursor:
            s["_id"] = str(s["_id"])
            stages.append(s)
        return stages

    def find_stage(self, doc_id: str, stage_name: str) -> Optional[Dict[str, Any]]:
        stage = self.collection.find_one({"doc_id": doc_id, "stage_name": stage_name})
        if stage and "_id" in stage:
            stage["_id"] = str(stage["_id"])
        return stage

    def delete_by_doc(self, doc_id: str) -> int:
        res = self.collection.delete_many({"doc_id": doc_id})
        sync_disk_if_mock()
        return res.deleted_count
