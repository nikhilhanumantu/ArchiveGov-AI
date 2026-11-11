import uuid
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from bson import ObjectId
from backend.app.database.mongodb import get_db, sync_disk_if_mock

class RestorationJobRepository:
    @property
    def collection(self):
        return get_db().restoration_jobs

    def create(self, job_data: Dict[str, Any]) -> Dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        if "job_id" not in job_data or not job_data["job_id"]:
            job_data["job_id"] = f"JOB-{uuid.uuid4().hex[:8].upper()}"
        job_data["created_at"] = job_data.get("created_at", now)
        job_data["status"] = job_data.get("status", "pending")
        job_data["current_stage"] = job_data.get("current_stage", "damage_analysis")
        job_data["progress"] = job_data.get("progress", 0)
        job_data["retry_count"] = job_data.get("retry_count", 0)
        
        # Initialize default stages status
        if "stages" not in job_data:
            job_data["stages"] = {
                "damage_analysis": {"status": "pending", "progress": 0},
                "opencv_preprocessing": {"status": "pending", "progress": 0},
                "realesrgan_enhancement": {"status": "pending", "progress": 0},
                "lama_inpainting": {"status": "pending", "progress": 0},
                "paddleocr_extraction": {"status": "pending", "progress": 0},
                "quality_evaluation": {"status": "pending", "progress": 0}
            }
            
        result = self.collection.insert_one(job_data)
        job_data["_id"] = str(result.inserted_id)
        sync_disk_if_mock()
        return job_data

    def find_by_id(self, job_id: str) -> Optional[Dict[str, Any]]:
        job = self.collection.find_one({"job_id": job_id})
        if job and "_id" in job:
            job["_id"] = str(job["_id"])
        return job

    def find_by_document_id(self, document_id: str) -> List[Dict[str, Any]]:
        cursor = self.collection.find({"document_id": document_id}).sort("created_at", -1)
        jobs = []
        for j in cursor:
            j["_id"] = str(j["_id"])
            jobs.append(j)
        return jobs

    def update_stage(self, job_id: str, stage_name: str, stage_status: str, progress: int = 100, details: Optional[Dict[str, Any]] = None):
        update_fields = {
            f"stages.{stage_name}.status": stage_status,
            f"stages.{stage_name}.progress": progress,
            "current_stage": stage_name
        }
        if details:
            update_fields[f"stages.{stage_name}.details"] = details
            
        self.collection.update_one({"job_id": job_id}, {"$set": update_fields})
        sync_disk_if_mock()

    def update_status(self, job_id: str, status: str, progress: int, error_message: Optional[str] = None, output_file_key: Optional[str] = None):
        now = datetime.now(timezone.utc).isoformat()
        update_fields: Dict[str, Any] = {
            "status": status,
            "progress": progress
        }
        if status == "running" and not self.collection.find_one({"job_id": job_id, "started_at": {"$exists": True}}):
            update_fields["started_at"] = now
        elif status in ["completed", "failed", "cancelled"]:
            update_fields["completed_at"] = now
            
        if error_message:
            update_fields["error_message"] = error_message
        if output_file_key:
            update_fields["output_file_key"] = output_file_key
            
        self.collection.update_one({"job_id": job_id}, {"$set": update_fields})
        sync_disk_if_mock()

    def list_jobs(self, limit: int = 20) -> List[Dict[str, Any]]:
        cursor = self.collection.find().sort("created_at", -1).limit(limit)
        jobs = []
        for j in cursor:
            j["_id"] = str(j["_id"])
            jobs.append(j)
        return jobs

