import re
import uuid
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

from backend.app.database.repositories.document_repo import DocumentRepository
from backend.app.database.repositories.ocr_repo import OCRRepository
from backend.app.database.repositories.timeline_repo import TimelineRepository

logger = logging.getLogger("archivegov.ai.timeline")

KNOWN_LOCATIONS = [
    "South Arcot", "Kuddalur", "Madras Presidency", "Bengal", "Bombay",
    "Calcutta", "Fort St. George", "Vellar River", "Chidambaram",
    "Tindivanam", "Cuddalore", "Poona", "Allahabad", "Lahore"
]

KNOWN_OFFICIALS = [
    "District Collector", "Sheristadar", "Sanitary Commissioner",
    "Chief Secretary", "High Court Judge", "Revenue Board Member",
    "Tahsildar", "Governor-in-Council", "Surgeon-General"
]

class TimelineEventExtractor:
    """
    Historical Chronological Event Extractor (Module 2 Timeline).
    Parses vault documents, extracts verified/inferred milestones,
    classifies date certainty, extracts entities, and maintains epistemic provenance.
    """
    def __init__(self):
        self.doc_repo = DocumentRepository()
        self.ocr_repo = OCRRepository()
        self.timeline_repo = TimelineRepository()

    def extract_events_from_doc(self, doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        doc_id = doc.get("document_id", "")
        title = doc.get("title", "Archival Document")
        dept = doc.get("department_id", "General Government Archives")
        ref_id = doc.get("archive_reference_id", doc_id)

        ocr_rec = self.ocr_repo.find_by_document_id(doc_id)
        ocr_text = ""
        if ocr_rec:
            ocr_text = ocr_rec.get("corrected_text") or ocr_rec.get("extracted_text") or ""
        if not ocr_text:
            ocr_text = doc.get("ocr_text", "")

        full_content = f"{title}\n{doc.get('description', '')}\n{ocr_text}"
        events: List[Dict[str, Any]] = []

        # 1. Primary Milestone from Document Certified Metadata
        meta_date = str(doc.get("original_date", ""))
        year_match = re.search(r"\b(1[789]\d{2}|20\d{2})\b", meta_date)
        meta_year = int(year_match.group(1)) if year_match else 1900

        # Extract entities for doc
        entities = self._extract_entities(full_content)

        primary_event_id = f"ev_{doc_id[:8]}_0"
        events.append({
            "event_id": primary_event_id,
            "timeline_id": "vault_master",
            "doc_id": doc_id,
            "archive_reference_id": ref_id,
            "title": f"Official Registration: {title}",
            "year": meta_year,
            "date_display": meta_date or str(meta_year),
            "certainty": "verified",
            "category": doc.get("category", "Government Document"),
            "department_id": dept,
            "entities": entities,
            "source_fact": ocr_text[:220] if ocr_text else title,
            "ai_interpretation": f"Certified archival folio promulgated under {dept} records, establishing official administrative standing.",
            "lens_url": f"/lens/{doc_id}",
            "archive_url": f"/app/archive/{doc_id}"
        })

        # 2. Extract In-Text Historical Milestones
        # Look for sentences or paragraphs with explicit years
        paragraphs = [p.strip() for p in ocr_text.split("\n") if p.strip()]
        seen_years = {meta_year}

        for idx, para in enumerate(paragraphs[:15]):
            found_years = re.findall(r"\b(1[789]\d{2}|20[0-2]\d)\b", para)
            for y_str in found_years:
                y_val = int(y_str)
                if y_val in seen_years:
                    continue
                seen_years.add(y_val)

                # Determine certainty
                certainty = "inferred"
                lower_p = para.lower()
                if any(w in lower_p for w in ["circa", "approx", "about", "around", "faded"]):
                    certainty = "uncertain"
                elif any(w in lower_p for w in ["certified", "ordered", "decreed", "enacted"]):
                    certainty = "verified"
                elif any(w in lower_p for w in ["disputed", "conflicting", "dissenting", "claimed"]):
                    certainty = "conflicting"

                # Sub-category
                category = "Administrative Action"
                if any(w in lower_p for w in ["revenue", "patta", "survey", "acres", "assessment"]):
                    category = "Land Settlement"
                elif any(w in lower_p for w in ["epidemic", "cholera", "plague", "health", "hospital"]):
                    category = "Public Health"
                elif any(w in lower_p for w in ["court", "decree", "suit", "appeal", "sheristadar"]):
                    category = "Judicial Decree"

                para_entities = self._extract_entities(para)

                events.append({
                    "event_id": f"ev_{doc_id[:8]}_{len(events)}",
                    "timeline_id": "vault_master",
                    "doc_id": doc_id,
                    "archive_reference_id": ref_id,
                    "title": f"{category}: {para[:60]}...",
                    "year": y_val,
                    "date_display": f"{y_val}",
                    "certainty": certainty,
                    "category": category,
                    "department_id": dept,
                    "entities": para_entities or entities,
                    "source_fact": para[:220],
                    "ai_interpretation": f"Historical record event regarding {category.lower()} noted in records of {dept}.",
                    "lens_url": f"/lens/{doc_id}",
                    "archive_url": f"/app/archive/{doc_id}"
                })

        return events

    def _extract_entities(self, text: str) -> List[str]:
        entities = []
        for loc in KNOWN_LOCATIONS:
            if loc.lower() in text.lower():
                entities.append(loc)
        for off in KNOWN_OFFICIALS:
            if off.lower() in text.lower():
                entities.append(off)
        # Regex capitalized phrases
        caps = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b", text)
        for c in caps[:4]:
            if c not in entities and len(c) > 4:
                entities.append(c)
        return list(dict.fromkeys(entities))[:6]

    def generate_timeline(
        self,
        timeline_id: str = "vault_master",
        department: Optional[str] = None,
        doc_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate or update chronological milestones across the vault or specific document.
        """
        if doc_id and doc_id != "all":
            doc = self.doc_repo.find_by_id(doc_id)
            docs = [doc] if doc else []
        else:
            docs, _ = self.doc_repo.list_documents(
                department=department if department and department != "all" else None,
                limit=150
            )

        all_events: List[Dict[str, Any]] = []
        for d in docs:
            if not d.get("is_deleted"):
                evs = self.extract_events_from_doc(d)
                all_events.extend(evs)

        # Persist to timeline_repo
        if all_events:
            self.timeline_repo.create_or_update_events(all_events)

        # Sort by year ASC
        all_events.sort(key=lambda x: x.get("year", 0))
        return all_events

event_extractor = TimelineEventExtractor()
