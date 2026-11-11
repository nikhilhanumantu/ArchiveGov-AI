from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from bson import ObjectId
from backend.app.database.mongodb import get_db, sync_disk_if_mock

class UserRepository:
    @property
    def collection(self):
        return get_db().users

    def create(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        user_data["created_at"] = user_data.get("created_at", now)
        user_data["updated_at"] = now
        user_data["account_status"] = user_data.get("account_status", "active")
        
        result = self.collection.insert_one(user_data)
        user_data["_id"] = str(result.inserted_id)
        sync_disk_if_mock()
        return user_data

    def find_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        user = self.collection.find_one({"email": email.strip().lower()})
        if user and "_id" in user:
            user["_id"] = str(user["_id"])
        return user

    def find_by_employee_id(self, employee_id: str) -> Optional[Dict[str, Any]]:
        user = self.collection.find_one({"employee_id": employee_id.strip()})
        if user and "_id" in user:
            user["_id"] = str(user["_id"])
        return user

    def find_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        try:
            oid = ObjectId(user_id) if ObjectId.is_valid(user_id) else user_id
            user = self.collection.find_one({"_id": oid})
            if not user and isinstance(oid, ObjectId):
                user = self.collection.find_one({"_id": str(user_id)})
        except Exception:
            user = self.collection.find_one({"_id": user_id})
            
        if user and "_id" in user:
            user["_id"] = str(user["_id"])
        return user

    def update(self, user_id: str, update_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        update_data["updated_at"] = datetime.now(timezone.utc).isoformat()
        try:
            oid = ObjectId(user_id) if ObjectId.is_valid(user_id) else user_id
        except Exception:
            oid = user_id
            
        self.collection.update_one({"_id": oid}, {"$set": update_data})
        sync_disk_if_mock()
        return self.find_by_id(user_id)

    def list_users(self, skip: int = 0, limit: int = 100) -> List[Dict[str, Any]]:
        cursor = self.collection.find().skip(skip).limit(limit)
        users = []
        for u in cursor:
            u["_id"] = str(u["_id"])
            users.append(u)
        return users

    def count(self) -> int:
        return self.collection.count_documents({})

