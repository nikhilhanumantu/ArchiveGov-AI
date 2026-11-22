import io
import csv
import pytest
import numpy as np
from pathlib import Path

from backend.app.ai.rag.index_service import IndexService
from backend.app.ai.ocr.paddleocr_adapter import PaddleOCRAdapter
from backend.app.database.repositories.document_repo import DocumentRepository
from backend.app.database.repositories.stage_repo import RestorationStageRepository
from backend.app.database.repositories.rag_chunk_repo import RAGChunkRepository
from backend.app.database.repositories.ocr_region_repo import OCRRegionRepository
from backend.app.database.repositories.timeline_repo import TimelineRepository
from backend.app.database.repositories.copilot_session_repo import CopilotSessionRepository
from backend.app.services.time_machine_service import TimeMachineService
from backend.app.routers.archive import export_archive_csv

def test_indic_tokenization_and_search():
    """Verify Indic scripts (Hindi, Tamil, Telugu) are preserved during tokenization and retrieval."""
    service = IndexService()

    hindi_query = "भूमि अभिलेख"
    tamil_query = "நிலம்"
    telugu_query = "భూమి రికార్డు"

    hindi_tokens = service.tokenize(hindi_query)
    tamil_tokens = service.tokenize(tamil_query)
    telugu_tokens = service.tokenize(telugu_query)

    assert hindi_tokens == ["भूमि", "अभिलेख"], f"Expected Hindi tokens, got {hindi_tokens}"
    assert tamil_tokens == ["நிலம்"], f"Expected Tamil tokens, got {tamil_tokens}"
    assert telugu_tokens == ["భూమి", "రికార్డు"], f"Expected Telugu tokens, got {telugu_tokens}"

    # Verify search_chunks does not immediately abort for Indic queries
    results_hindi = service.search_chunks(query=hindi_query)
    assert isinstance(results_hindi, list)

def test_ocr_fallback_emits_empty_transcription_and_proper_engine():
    """Verify OCR fallback emits empty transcription with 0.0 confidence, never saving placeholders."""
    adapter = PaddleOCRAdapter()
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)

    res = adapter._run_morphological_ocr(dummy_img, 100, 100, "en")

    assert res["extracted_text"] == "", "Fallback extracted_text must be empty string, not placeholder"
    assert res["average_confidence"] == 0.0, "Fallback confidence must be 0.0"
    assert res["engine"] == "Morphological Document Segmenter (Fallback)"
    assert res["is_fallback"] is True
    for region in res.get("regions", []):
        assert region.get("text") == "", "Fallback region text must be empty"
        assert region.get("confidence") == 0.0
        assert "[Text Region" not in region.get("text", "")

def test_restoration_rerun_preserves_version_history():
    """Verify restoration version isolation: cleaning one version does not erase other versions."""
    stage_repo = RestorationStageRepository()
    doc_id = "DOC-TEST-RERUN-VERSION"

    # Seed stages for v1.0 and v2.0
    stage_repo.delete_by_doc(doc_id)
    stage_repo.create({"doc_id": doc_id, "version_id": "v1.0", "stage_name": "stage_01", "file_key": "stage1_v1.png", "stage_order": 1})
    stage_repo.create({"doc_id": doc_id, "version_id": "v1.0", "stage_name": "stage_02", "file_key": "stage2_v1.png", "stage_order": 2})
    stage_repo.create({"doc_id": doc_id, "version_id": "v2.0", "stage_name": "stage_01", "file_key": "stage1_v2.png", "stage_order": 1})

    assert len(stage_repo.find_by_doc(doc_id, "v1.0")) == 2
    assert len(stage_repo.find_by_doc(doc_id, "v2.0")) == 1

    # Rerun version v2.0: only v2.0 stages should be deleted, preserving v1.0
    stage_repo.delete_by_doc_version(doc_id, "v2.0")

    v1_stages = stage_repo.find_by_doc(doc_id, "v1.0")
    v2_stages = stage_repo.find_by_doc(doc_id, "v2.0")

    assert len(v1_stages) == 2, "v1.0 stage history must remain intact"
    assert len(v2_stages) == 0, "Only v2.0 stages should be cleared for the rerun"

    # Cleanup
    stage_repo.delete_by_doc(doc_id)

def test_deep_document_deletion_cascades_all_materials():
    """Verify permanent deletion purges derived Copilot chunks, OCR regions, timeline events, and stages."""
    doc_repo = DocumentRepository()
    chunk_repo = RAGChunkRepository()
    region_repo = OCRRegionRepository()
    timeline_repo = TimelineRepository()
    stage_repo = RestorationStageRepository()
    copilot_repo = CopilotSessionRepository()

    doc_id = "DOC-TEST-CASCADE-PURGE"

    # Seed record across collections
    doc_repo.delete(doc_id)
    doc_repo.collection.insert_one({"document_id": doc_id, "title": "Purge Test Document"})
    chunk_repo.collection.insert_one({"doc_id": doc_id, "chunk_id": "CHK-1", "text": "Test text"})
    region_repo.collection.insert_one({"document_id": doc_id, "region_id": "REG-1"})
    timeline_repo.collection.insert_one({"doc_id": doc_id, "event_id": "EVT-1"})
    stage_repo.create({"doc_id": doc_id, "version_id": "v1.0", "stage_name": "stage_01", "file_key": "stg.png"})
    copilot_repo.collection.insert_one({"doc_id": doc_id, "session_id": "SESS-1"})

    # Ensure records exist
    assert chunk_repo.collection.count_documents({"doc_id": doc_id}) == 1
    assert region_repo.collection.count_documents({"document_id": doc_id}) == 1
    assert timeline_repo.collection.count_documents({"doc_id": doc_id}) == 1
    assert stage_repo.collection.count_documents({"doc_id": doc_id}) == 1
    assert copilot_repo.collection.count_documents({"doc_id": doc_id}) == 1

    # Execute cascade delete
    deleted = doc_repo.delete(doc_id)
    assert deleted is True

    # Confirm all derived collections are completely purged
    assert chunk_repo.collection.count_documents({"doc_id": doc_id}) == 0
    assert region_repo.collection.count_documents({"document_id": doc_id}) == 0
    assert timeline_repo.collection.count_documents({"doc_id": doc_id}) == 0
    assert stage_repo.collection.count_documents({"doc_id": doc_id}) == 0
    assert copilot_repo.collection.count_documents({"doc_id": doc_id}) == 0
    assert doc_repo.collection.count_documents({"document_id": doc_id}) == 0

def test_archive_csv_export_pagination_beyond_1000(client, monkeypatch):
    """Verify CSV export streams all records across pagination boundaries without stopping at 1,000."""
    total_records = 1001

    # Mock list_documents to paginate 1,001 simulated records
    def mock_list_documents(skip=0, limit=500, sort_by="created_at", sort_dir=-1, **kwargs):
        records = []
        end = min(skip + limit, total_records)
        for i in range(skip, end):
            records.append({
                "document_id": f"DOC-{i:04d}",
                "title": f"Archival Record {i}",
                "archive_reference_id": f"REF-{i:04d}",
                "department_id": "Dept of Revenue",
                "category": "Historical Deeds",
                "original_date": "1912-05-14",
                "language": "English",
                "status": "archived",
                "created_at": "2026-01-01T00:00:00Z"
            })
        return records, total_records

    from backend.app.routers import archive
    monkeypatch.setattr(archive.doc_repo, "list_documents", mock_list_documents)

    from tests.backend.test_api import get_auth_headers
    headers = get_auth_headers(client)

    response = client.get("/api/v1/archive/export-csv", headers=headers)
    assert response.status_code == 200

    reader = list(csv.reader(io.StringIO(response.text)))

    # Header row + 1,001 document rows = 1,002 lines
    assert len(reader) == 1002, f"Expected 1002 rows (1 header + 1001 records), got {len(reader)}"
    assert reader[0][0] == "Document ID"
    assert reader[1][0] == "DOC-0000"
    assert reader[-1][0] == "DOC-1000"
