from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from backend.app.database.mongodb import get_db, sync_disk_if_mock

class TimelineRepository:
    """Repository for historical chronological event extraction (Module 2 Timeline)."""
    @property
    def collection(self):
        return get_db().timeline_events

    def create_or_update_events(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not events:
            return []
        now = datetime.now(timezone.utc).isoformat()
        for ev in events:
            ev["updated_at"] = now
            if not ev.get("created_at"):
                ev["created_at"] = now
            self.collection.update_one(
                {"event_id": ev["event_id"]},
                {"$set": ev},
                upsert=True
            )
        sync_disk_if_mock()
        return events

    def find_events(
        self,
        doc_id: Optional[str] = None,
        department: Optional[str] = None,
        certainty: Optional[str] = None,
        entity: Optional[str] = None,
        limit: int = 150
    ) -> List[Dict[str, Any]]:
        query: Dict[str, Any] = {}
        if doc_id:
            query["doc_id"] = doc_id
        if department and department != "all":
            query["department_id"] = department
        if certainty and certainty != "all":
            query["certainty"] = certainty
        if entity:
            query["entities"] = {"$regex": entity, "$options": "i"}

        cursor = self.collection.find(query).sort("year", 1).limit(limit)
        results = []
        for ev in cursor:
            ev["_id"] = str(ev["_id"])
            results.append(ev)
        return results

    def find_by_timeline_id(self, timeline_id: str) -> List[Dict[str, Any]]:
        cursor = self.collection.find({"timeline_id": timeline_id}).sort("year", 1)
        results = []
        for ev in cursor:
            ev["_id"] = str(ev["_id"])
            results.append(ev)
        return results

    def find_by_id(self, event_id: str) -> Optional[Dict[str, Any]]:
        ev = self.collection.find_one({"event_id": event_id})
        if ev and "_id" in ev:
            ev["_id"] = str(ev["_id"])
        return ev
