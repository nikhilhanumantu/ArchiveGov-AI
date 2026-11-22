import os
import shutil
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import cv2
import numpy as np

from backend.app.core.config import settings
from backend.app.storage.local_storage import storage
from backend.app.database.repositories.document_repo import DocumentRepository
from backend.app.database.repositories.version_repo import DocumentVersionRepository
from backend.app.database.repositories.stage_repo import RestorationStageRepository

logger = logging.getLogger("archivegov.services.time_machine")

class TimeMachineService:
    """
    Restoration Time Machine Service (Module 3).
    Tracks granular pipeline execution stages, provides before/after split comparisons,
    and computes pixel-level difference masks / heatmaps showing reconstructed visual regions.
    """
    def __init__(self):
        self.stage_repo = RestorationStageRepository()
        self.doc_repo = DocumentRepository()
        self.version_repo = DocumentVersionRepository()

    def get_stages(self, doc_id: str, version_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Return sequential restoration stages ordered by stage_order.
        Falls back to baseline version lineage if stage records were not yet persisted for legacy scans.
        """
        doc = self.doc_repo.find_by_id(doc_id)
        if not doc:
            return {"stages": [], "error": "Document not found"}

        target_version = version_id or doc.get("current_version_id")
        stages = self.stage_repo.find_by_doc(doc_id, version_id=target_version)
        if not stages and version_id and version_id != doc.get("current_version_id"):
            return {"stages": [], "error": "No stages recorded for this version"}

        if stages:
            resolved_version = stages[0].get("version_id") or target_version or "v2.0"
            return {
                "document_id": doc_id,
                "version_id": resolved_version,
                "title": doc.get("title", ""),
                "total_stages": len(stages),
                "stages": stages
            }

        # Baseline auto-synthesis for documents processed before granular stage logging
        synthesized_stages = []
        original_key = doc.get("original_file_key")
        orig_path = storage.resolve_path(original_key) if original_key else None
        
        target_version = target_version or "v2.0"
        stages_dir = storage.stages / doc_id / target_version
        stages_dir.mkdir(parents=True, exist_ok=True)

        if orig_path and orig_path.exists():
            stage_0_path = stages_dir / "00_original.png"
            if not stage_0_path.exists():
                shutil.copy2(orig_path, stage_0_path)
            stage_0 = self.stage_repo.create({
                "doc_id": doc_id,
                "version_id": target_version,
                "stage_order": 0,
                "stage_name": "original",
                "label": "Original Archival Ingest",
                "file_path": str(stage_0_path),
                "file_url": f"/uploads/stages/{doc_id}/{target_version}/00_original.png",
                "model_used": "Raw Archival Sensor Scan",
                "fallback_used": False,
                "reconstructed_mask_path": None,
                "metrics": None
            })
            synthesized_stages.append(stage_0)

        restored_key = doc.get("restored_file_key")
        rest_path = storage.resolve_path(restored_key) if restored_key else None
        if rest_path and rest_path.exists():
            stage_final_path = stages_dir / "05_final.png"
            if not stage_final_path.exists():
                shutil.copy2(rest_path, stage_final_path)
            stage_final = self.stage_repo.create({
                "doc_id": doc_id,
                "version_id": target_version,
                "stage_order": 5,
                "stage_name": "final",
                "label": "Certified Restored Master",
                "file_path": str(stage_final_path),
                "file_url": f"/uploads/stages/{doc_id}/{target_version}/05_final.png",
                "model_used": "Real-ESRGAN + LaMa + OpenCV",
                "fallback_used": False,
                "reconstructed_mask_path": None,
                "metrics": doc.get("quality_metrics", {})
            })
            synthesized_stages.append(stage_final)

        if synthesized_stages:
            return {
                "document_id": doc_id,
                "version_id": version_id or doc.get("current_version_id", "v2.0"),
                "title": doc.get("title", ""),
                "total_stages": len(synthesized_stages),
                "stages": synthesized_stages
            }

        return {
            "document_id": doc_id,
            "stages": [],
            "action_needed": "Re-run pipeline to generate stage history"
        }

    def get_diff_mask(
        self,
        doc_id: str,
        from_stage: str = "original",
        to_stage: str = "final"
    ) -> Dict[str, Any]:
        """
        Compute or return the difference heatmap mask between two stages showing reconstructed pixels.
        """
        stages = self.get_stages(doc_id).get("stages", [])
        stage_map = {s.get("stage_name"): s for s in stages}

        # Resolve paths
        from_path = None
        to_path = None

        if from_stage in stage_map:
            from_path = stage_map[from_stage].get("file_path")
        if to_stage in stage_map:
            to_path = stage_map[to_stage].get("file_path")

        # Fallback to document file keys if stages not recorded
        if not from_path or not to_path:
            doc = self.doc_repo.find_by_id(doc_id)
            if doc:
                if not from_path and doc.get("original_file_key"):
                    p = storage.resolve_path(doc["original_file_key"])
                    from_path = str(p) if p else None
                if not to_path and doc.get("restored_file_key"):
                    p = storage.resolve_path(doc["restored_file_key"])
                    to_path = str(p) if p else None

        if not from_path or not to_path or not os.path.exists(from_path) or not os.path.exists(to_path):
            return {
                "success": False,
                "error": f"Cannot compute diff between '{from_stage}' and '{to_stage}': image assets missing on disk."
            }

        out_dir = Path(settings.UPLOAD_DIR) / "stages" / doc_id / "diffs"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"diff_{from_stage}_{to_stage}.png"

        try:
            img_a = cv2.imread(from_path)
            img_b = cv2.imread(to_path)
            if img_a is None or img_b is None:
                raise ValueError("Could not read image files for diff mask.")

            h_b, w_b = img_b.shape[:2]
            if img_a.shape[:2] != (h_b, w_b):
                img_a = cv2.resize(img_a, (w_b, h_b), interpolation=cv2.INTER_LINEAR)

            diff = cv2.absdiff(img_a, img_b)
            gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            norm = cv2.normalize(gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            heatmap = cv2.applyColorMap(norm, cv2.COLORMAP_JET)

            # Generate overlay blend
            overlay = cv2.addWeighted(img_b, 0.65, heatmap, 0.35, 0)
            cv2.imwrite(str(out_file), overlay)

            # Compute reconstructed percentage
            _, thresh = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
            altered_pixels = int(np.count_nonzero(thresh))
            total_pixels = int(h_b * w_b)
            reconstructed_ratio = round((altered_pixels / max(1, total_pixels)) * 100, 2)

            return {
                "success": True,
                "doc_id": doc_id,
                "from_stage": from_stage,
                "to_stage": to_stage,
                "mask_url": f"/uploads/stages/{doc_id}/diffs/diff_{from_stage}_{to_stage}.png",
                "file_path": str(out_file),
                "reconstructed_percentage": reconstructed_ratio,
                "altered_pixels": altered_pixels,
                "provenance": "reconstructed_visual"
            }
        except Exception as e:
            logger.error(f"Error computing diff mask: {e}", exc_info=True)
            return {
                "success": False,
                "error": f"Failed to compute diff mask: {str(e)}"
            }

time_machine_service = TimeMachineService()
