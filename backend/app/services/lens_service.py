import re
import uuid
import logging
from typing import List, Dict, Any, Optional, Set
import requests

from backend.app.core.config import settings
from backend.app.database.repositories.document_repo import DocumentRepository
from backend.app.database.repositories.ocr_repo import OCRRepository
from backend.app.database.repositories.ocr_region_repo import OCRRegionRepository

logger = logging.getLogger("archivegov.services.lens")

STOPWORDS: Set[str] = {
    "a", "an", "the", "and", "or", "in", "on", "at", "to", "for", "with",
    "by", "of", "from", "is", "was", "are", "were", "it", "this", "that",
    "these", "those", "be", "been", "has", "had", "have", "do", "does",
    "did", "as", "about", "what", "which", "who", "when", "where", "why",
    "how", "can", "could", "would", "should", "all", "any", "some"
}

class LensService:
    """
    Interactive Document Intelligence Lens Service (Module 4).
    Provides granular OCR box exploration, epistemically-labeled explanations (source_fact,
    ocr_extraction, ai_interpretation, reconstructed_visual), and cross-vault record linkage.
    """
    def __init__(self):
        self.doc_repo = DocumentRepository()
        self.ocr_repo = OCRRepository()
        self.region_repo = OCRRegionRepository()

    def get_regions(
        self,
        doc_id: str,
        page_number: int = 1,
        granularity: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Fetch normalized and pixel bounding boxes for a document.
        Automatically converts legacy OCR results into persistent multi-granularity regions.
        """
        doc = self.doc_repo.find_by_id(doc_id)
        if not doc:
            return {"document_id": doc_id, "regions": [], "error": "Document not found"}

        regions = self.region_repo.find_by_doc(doc_id, page_number=page_number, granularity=granularity)
        if regions:
            return {
                "document_id": doc_id,
                "title": doc.get("title", ""),
                "page_number": page_number,
                "total_regions": len(regions),
                "regions": regions
            }

        # Check if OCR record exists to synthesize granular regions
        ocr_record = self.ocr_repo.find_by_document_id(doc_id)
        raw_lines = []
        if ocr_record:
            raw_lines = ocr_record.get("text_regions", [])

        # Fallback to lines in document ocr_text if raw_lines is empty
        if not raw_lines and doc.get("ocr_text"):
            text_lines = [l.strip() for l in doc["ocr_text"].split("\n") if l.strip()]
            for idx, tl in enumerate(text_lines):
                raw_lines.append({
                    "line_number": idx + 1,
                    "text": tl,
                    "confidence": 95.0,
                    "bbox": [50, 100 + idx * 45, 700, 35],
                    "bbox_normalized": [0.06, round((100 + idx * 45) / 1200.0, 3), 0.88, 0.035]
                })

        if not raw_lines:
            return {
                "document_id": doc_id,
                "title": doc.get("title", ""),
                "regions": [],
                "action_needed": "Re-run pipeline to generate stage history / OCR regions"
            }

        # Convert raw_lines to persistent multi-granularity regions (line, word, block)
        damage_defects = doc.get("damage_analysis", {}).get("defects", [])
        synthesized_regions: List[Dict[str, Any]] = []

        block_lines = []
        block_text = []

        for line_idx, rl in enumerate(raw_lines):
            line_text = str(rl.get("text", "")).strip()
            if not line_text:
                continue

            conf_raw = float(rl.get("confidence", 95.0))
            conf = round(conf_raw / 100.0, 3) if conf_raw > 1.0 else round(conf_raw, 3)

            bbox = rl.get("bbox", [50, 50 + line_idx * 40, 600, 30])
            norm = rl.get("bbox_normalized", [0.05, 0.05 + line_idx * 0.04, 0.75, 0.035])

            # Compute overlap with damage defects
            damage_overlap = 0.0
            rx, ry, rw, rh = bbox[0], bbox[1], bbox[2], bbox[3]
            for defect in damage_defects:
                dbbox = defect.get("bbox", [0, 0, 0, 0])
                dx, dy, dw, dh = dbbox[0], dbbox[1], dbbox[2], dbbox[3]
                ix = max(0, min(rx + rw, dx + dw) - max(rx, dx))
                iy = max(0, min(ry + rh, dy + dh) - max(ry, dy))
                if ix > 0 and iy > 0:
                    damage_overlap = round((ix * iy) / max(1, rw * rh), 3)
                    break

            line_region = {
                "region_id": f"reg_{doc_id[:6]}_l_{line_idx+1}",
                "doc_id": doc_id,
                "version_id": doc.get("current_version_id", "v2.0"),
                "page_number": page_number,
                "granularity": "line",
                "bbox_pixels": bbox,
                "bbox_normalized": norm,
                "text": line_text,
                "confidence": conf,
                "damage_overlap": damage_overlap
            }
            synthesized_regions.append(line_region)

            # Generate word-level regions
            words = line_text.split()
            if words:
                word_w = max(10, int(rw / len(words)))
                for widx, w in enumerate(words):
                    wx = rx + widx * word_w
                    synthesized_regions.append({
                        "region_id": f"reg_{doc_id[:6]}_w_{line_idx+1}_{widx+1}",
                        "doc_id": doc_id,
                        "version_id": doc.get("current_version_id", "v2.0"),
                        "page_number": page_number,
                        "granularity": "word",
                        "bbox_pixels": [wx, ry, word_w - 4, rh],
                        "bbox_normalized": [
                            round(norm[0] + widx * (norm[2] / len(words)), 3),
                            norm[1],
                            round(norm[2] / len(words), 3),
                            norm[3]
                        ],
                        "text": w,
                        "confidence": conf,
                        "damage_overlap": damage_overlap
                    })

            # Accumulate block
            block_lines.append(line_region)
            block_text.append(line_text)
            if len(block_lines) >= 4 or line_idx == len(raw_lines) - 1:
                b_rx = min(l["bbox_pixels"][0] for l in block_lines)
                b_ry = min(l["bbox_pixels"][1] for l in block_lines)
                b_rw = max(l["bbox_pixels"][0] + l["bbox_pixels"][2] for l in block_lines) - b_rx
                b_rh = max(l["bbox_pixels"][1] + l["bbox_pixels"][3] for l in block_lines) - b_ry
                block_id = f"reg_{doc_id[:6]}_b_{len(synthesized_regions)}"
                synthesized_regions.append({
                    "region_id": block_id,
                    "doc_id": doc_id,
                    "version_id": doc.get("current_version_id", "v2.0"),
                    "page_number": page_number,
                    "granularity": "block",
                    "bbox_pixels": [b_rx, b_ry, b_rw, b_rh],
                    "bbox_normalized": [
                        round(b_rx / 1000.0, 3),
                        round(b_ry / 1400.0, 3),
                        round(b_rw / 1000.0, 3),
                        round(b_rh / 1400.0, 3)
                    ],
                    "text": " ".join(block_text),
                    "confidence": round(sum(l["confidence"] for l in block_lines) / len(block_lines), 3),
                    "damage_overlap": round(max(l["damage_overlap"] for l in block_lines), 3)
                })
                block_lines = []
                block_text = []

        if synthesized_regions:
            self.region_repo.insert_many(synthesized_regions)

        # Filter by granularity if requested
        ret_regions = synthesized_regions
        if granularity:
            ret_regions = [r for r in synthesized_regions if r.get("granularity") == granularity]

        return {
            "document_id": doc_id,
            "title": doc.get("title", ""),
            "page_number": page_number,
            "total_regions": len(ret_regions),
            "regions": ret_regions
        }

    def explain_region(
        self,
        doc_id: str,
        region_id: Optional[str] = None,
        text: Optional[str] = None,
        query: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Explain selected text or bounding box with strict epistemic provenance:
        - source_fact (verbatim text)
        - ocr_extraction (OCR confidence & engine)
        - ai_interpretation (contextual analysis)
        - reconstructed_visual (pixel modification status)
        """
        doc = self.doc_repo.find_by_id(doc_id)
        if not doc:
            return {"error": "Document not found"}

        target_text = text or ""
        confidence = 0.95
        damage_overlap = 0.0

        if region_id:
            reg = self.region_repo.find_by_region_id(doc_id, region_id)
            if reg:
                target_text = reg.get("text", "")
                confidence = reg.get("confidence", 0.95)
                damage_overlap = reg.get("damage_overlap", 0.0)

        if not target_text and doc.get("ocr_text"):
            target_text = doc["ocr_text"][:300]

        # Epistemic Provenance Components
        source_fact = target_text
        ocr_extraction = {
            "confidence": confidence,
            "engine": "PaddleOCR v3.7 PP-OCRv6 Deep Neural Engine",
            "damage_overlap": damage_overlap
        }
        reconstructed_visual = {
            "altered_pixels": bool(damage_overlap > 0.05),
            "note": "Reconstructed via neural inpainting / super-resolution" if damage_overlap > 0.05 else "Authentic substrate visual"
        }

        # Synthesize AI interpretation (Cloud LLM or sovereign offline archival heuristic)
        ai_interpretation = self._synthesize_explanation(doc, target_text, query)

        return {
            "document_id": doc_id,
            "region_id": region_id,
            "title": doc.get("title"),
            "source_fact": source_fact,
            "ocr_extraction": ocr_extraction,
            "reconstructed_visual": reconstructed_visual,
            "ai_interpretation": ai_interpretation,
            "provenance": "ai_interpretation"
        }

    def _synthesize_explanation(self, doc: Dict[str, Any], text: str, user_query: Optional[str] = None) -> str:
        """Query external LLM if configured; otherwise use offline archival knowledge base."""
        # 1. Try Google AI Studio (Gemini 2.5 Flash)
        if getattr(settings, "GOOGLE_AI_STUDIO_API_KEY", None):
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={settings.GOOGLE_AI_STUDIO_API_KEY}"
                prompt = (
                    f"You are a sovereign archival records scholar. Analyze this historical excerpt from document "
                    f"'{doc.get('title')}' (Date: {doc.get('original_date')}, Department: {doc.get('department_id')}).\n"
                    f"Text: \"{text}\"\n"
                    f"User question: {user_query or 'Explain the administrative, legal, or fiscal meaning of this passage.'}\n"
                    f"Provide an objective, historical interpretation citing fiscal denominations (rupees/annas/pies), land tenures (ryotwari/patta), or public orders accurately."
                )
                payload = {"contents": [{"parts": [{"text": prompt}]}]}
                resp = requests.post(url, json=payload, timeout=8)
                if resp.status_code == 200:
                    data = resp.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        return candidates[0]["content"]["parts"][0]["text"].strip()
            except Exception as e:
                logger.warning(f"Google AI Studio explanation error: {e}")

        # 2. Try OpenRouter
        if getattr(settings, "OPENROUTER_API_KEY", None):
            try:
                url = "https://openrouter.ai/api/v1/chat/completions"
                headers = {"Authorization": f"Bearer {settings.OPENROUTER_API_KEY}", "Content-Type": "application/json"}
                prompt = f"Historical record analysis of '{doc.get('title')}' ({doc.get('original_date')}):\nText: \"{text}\""
                payload = {
                    "model": "google/gemini-2.5-flash",
                    "messages": [{"role": "user", "content": prompt}]
                }
                resp = requests.post(url, headers=headers, json=payload, timeout=8)
                if resp.status_code == 200:
                    data = resp.json()
                    return data["choices"][0]["message"]["content"].strip()
            except Exception as e:
                logger.warning(f"OpenRouter explanation error: {e}")

        # 3. Sovereign Offline Archival Rule-Based Explanation Fallback
        return self._offline_archival_explanation(doc, text)

    def _offline_archival_explanation(self, doc: Dict[str, Any], text: str) -> str:
        """Deterministic, rich archival interpretation for offline environments."""
        lower_t = text.lower()
        insights = []

        if any(w in lower_t for w in ["acres", "cents", "survey", "fasli", "kuddalur", "arcot", "patta", "ryotwari"]):
            insights.append(
                "**Land Revenue & Settlement Context**: This entry specifies cadastral survey demarcations and annual assessments. "
                "In British India and provincial presidencies (such as Madras), ryotwari settlements conferred individual proprietary deeds (patta) "
                "subject to quinquennial or decennial revenue remeasurements."
            )
        if any(w in lower_t for w in ["rupees", "rs", "annas", "pies", "assessment", "cess"]):
            insights.append(
                "**Fiscal Denominations**: The monetary amounts reflect the classic currency system (1 Rupee = 16 Annas = 192 Pies). "
                "Cesses and water rates were levied based on whether land was categorized as 'wet' (nanjai, canal-irrigated) or 'dry' (punjai, rainfed)."
            )
        if any(w in lower_t for w in ["epidemic", "cholera", "plague", "quarantine", "sanitary", "fever", "influenza"]):
            insights.append(
                "**Public Health Order**: This excerpt documents sovereign epidemiological containment protocols. Under the Epidemic Diseases Act "
                "and Sanitary Commissioner circulars, emergency travel quarantines and fumigation cordons were strictly enforced."
            )
        if any(w in lower_t for w in ["court", "decree", "sheristadar", "collector", "magistrate", "gazette", "sealed"]):
            insights.append(
                "**Administrative Certification**: Marked by the District Collectorate or Judicial Sheristadar seal, certifying this entry "
                "as an admissible official government instrument."
            )

        if not insights:
            insights.append(
                f"**Archival Folio Context**: Official record maintained by the {doc.get('department_id', 'Government Archive')} "
                f"under reference identifier **{doc.get('archive_reference_id', doc.get('document_id'))}** ({doc.get('original_date', 'Historical Period')}). "
                f"The transcribed text reflects certified government record keeping."
            )

        return "\n\n".join(insights)

    def get_related_records(self, doc_id: str, region_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Find related archival folios based on department, temporal proximity, and shared entities.
        """
        doc = self.doc_repo.find_by_id(doc_id)
        if not doc:
            return []

        all_docs = self.doc_repo.list_all(limit=50)
        related = []

        dept = doc.get("department_id", "")
        year_match = re.search(r"\b(18\d{2}|19\d{2})\b", str(doc.get("original_date", "")))
        doc_year = int(year_match.group(1)) if year_match else None

        for other in all_docs:
            o_id = other.get("document_id")
            if o_id == doc_id or other.get("is_deleted"):
                continue

            reasons = []
            if dept and other.get("department_id") == dept:
                reasons.append(f"Same department ({dept})")

            other_year_match = re.search(r"\b(18\d{2}|19\d{2})\b", str(other.get("original_date", "")))
            if doc_year and other_year_match:
                other_year = int(other_year_match.group(1))
                if abs(doc_year - other_year) <= 6:
                    reasons.append(f"Contemporary era ({other_year} vs {doc_year})")

            # Check shared keywords
            d_words = set(re.findall(r"\w{4,}", (doc.get("title", "") + " " + doc.get("description", "")).lower()))
            o_words = set(re.findall(r"\w{4,}", (other.get("title", "") + " " + other.get("description", "")).lower()))
            shared = d_words.intersection(o_words) - STOPWORDS
            if shared:
                reasons.append(f"Shared archival terms: {', '.join(list(shared)[:3])}")

            if reasons:
                related.append({
                    "document_id": o_id,
                    "title": other.get("title", "Untitled Record"),
                    "archive_reference_id": other.get("archive_reference_id", o_id),
                    "department_id": other.get("department_id"),
                    "original_date": other.get("original_date"),
                    "similarity_reason": " · ".join(reasons),
                    "lens_url": f"/lens/{o_id}",
                    "archive_url": f"/app/archive/{o_id}"
                })

        return related[:5]

lens_service = LensService()
