from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from backend.app.database.mongodb import get_db, sync_disk_if_mock

class SettingsRepository:
    def __init__(self):
        self.db = get_db()
        self.collection = self.db.system_settings

    def get_all(self) -> Dict[str, Any]:
        cursor = self.collection.find()
        settings_dict = {}
        for s in cursor:
            settings_dict[s["setting_key"]] = s.get("setting_value")
        return settings_dict

    def get_value(self, key: str, default: Any = None) -> Any:
        doc = self.collection.find_one({"setting_key": key})
        if doc and "setting_value" in doc:
            return doc["setting_value"]
        return default

    def set_value(self, key: str, value: Any, updated_by: str = "system") -> Dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        doc = {
            "setting_key": key,
            "setting_value": value,
            "updated_by": updated_by,
            "updated_at": now
        }
        self.collection.update_one(
            {"setting_key": key},
            {"$set": doc},
            upsert=True
        )
        sync_disk_if_mock()
        return doc

