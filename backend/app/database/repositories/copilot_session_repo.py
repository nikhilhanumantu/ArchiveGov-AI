import uuid
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from backend.app.database.mongodb import get_db, sync_disk_if_mock

class CopilotSessionRepository:
    """Repository for tracking multi-turn Copilot conversations."""
    @property
    def collection(self):
        return get_db().copilot_sessions

    def get_or_create(self, session_id: Optional[str] = None, user_id: Optional[str] = None) -> Dict[str, Any]:
        if session_id:
            session = self.collection.find_one({"session_id": session_id})
            if session:
                session["_id"] = str(session["_id"])
                return session

        new_session_id = session_id or f"session_{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc).isoformat()
        session_data = {
            "session_id": new_session_id,
            "user_id": user_id or "anonymous",
            "messages": [],
            "created_at": now,
            "updated_at": now
        }
        res = self.collection.insert_one(session_data)
        session_data["_id"] = str(res.inserted_id)
        sync_disk_if_mock()
        return session_data

    def append_message(
        self,
        session_id: str,
        role: str,
        content: str,
        citations: Optional[List[Dict[str, Any]]] = None,
        engine_used: Optional[str] = None,
        provenance: Optional[str] = None
    ) -> Dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        msg = {
            "id": f"msg_{uuid.uuid4().hex[:8]}",
            "role": role,
            "content": content,
            "citations": citations or [],
            "engine_used": engine_used,
            "provenance": provenance or ("ai_interpretation" if role == "assistant" else "source_fact"),
            "timestamp": now
        }
        self.collection.update_one(
            {"session_id": session_id},
            {
                "$push": {"messages": msg},
                "$set": {"updated_at": now}
            },
            upsert=True
        )
        sync_disk_if_mock()
        return msg

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        session = self.collection.find_one({"session_id": session_id})
        if session and "_id" in session:
            session["_id"] = str(session["_id"])
        return session
