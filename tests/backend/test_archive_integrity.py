from pathlib import Path
from unittest.mock import Mock

import mongomock
import pytest

from backend.app.database import mongodb
from backend.app.database.repositories.copilot_session_repo import CopilotSessionRepository
from backend.app.routers import documents, versions
from backend.app.workers import job_worker


@pytest.fixture
def archive(tmp_path, monkeypatch):
    db = mongomock.MongoClient().archive
    monkeypatch.setattr(mongodb, "_db", db)
    monkeypatch.setattr(mongodb, "_mock_storage_path", tmp_path / "db.json")
    storage = job_worker.storage
    monkeypatch.setattr(storage, "base_dir", tmp_path)
    for name in ("working", "restored", "originals", "stages", "exports", "masks"):
        folder = tmp_path / name
        folder.mkdir()
        monkeypatch.setattr(storage, name, folder)
    (storage.originals / "scan.png").write_bytes(b"source")
    documents.doc_repo.create({"document_id": "D", "original_file_key": "scan.png",
                               "title": "Record", "language": "Tamil", "current_version_id": "original"})
    return db


def test_worker_rerun_failure_and_rollback(archive, monkeypatch):
    worker = job_worker.RestorationJobWorker()
    pages = iter((b"first", b"second", b"failed"))

    def preprocess(input_path, output_path, **options):
        Path(output_path).write_bytes(next(pages))
        return {"operations_applied": {}, "deskew_angle": 0}

    def recognize(path, lang):
        assert lang == "Tamil"
        text = Path(path).read_text()
        return {"language": "ta", "engine": "test", "average_confidence": 100,
                "extracted_text": text, "regions": [{"text": text, "confidence": 100}]}

    monkeypatch.setattr(job_worker.opencv_processor, "process_pipeline", preprocess)
    monkeypatch.setattr(job_worker.paddleocr_adapter, "extract_text", recognize)
    ops = ["opencv_preprocessing", "paddleocr_extraction"]

    def run(operations):
        job = worker.job_repo.create({"document_id": "D", "requested_operations": operations})
        worker._execute_pipeline(job["job_id"])
        return worker.job_repo.find_by_id(job["job_id"])["status"]

    assert run(ops) == "completed"
    first = worker.doc_repo.find_by_id("D")
    assert run(ops) == "completed"
    second = worker.doc_repo.find_by_id("D")
    assert first["restored_file_key"] != second["restored_file_key"]
    monkeypatch.setattr(job_worker.quality_evaluator, "evaluate", Mock(side_effect=ValueError("failed")))
    assert run(ops + ["quality_evaluation"]) == "failed"
    assert worker.doc_repo.find_by_id("D")["current_version_id"] == second["current_version_id"]

    versions.rollback_version("D", first["current_version_id"], {"_id": "reviewer"})
    current = worker.doc_repo.find_by_id("D")
    assert job_worker.storage.resolve_path(current["restored_file_key"]).read_bytes() == b"first"
    assert current["ocr_text"] == worker.ocr_repo.find_by_document_id("D")["extracted_text"] == "first"
    for document, expected in [(first, "first"), (second, "second")]:
        regions = worker.region_repo.find_by_doc("D", granularity="line", version_id=document["current_version_id"])
        assert [region["text"] for region in regions] == [expected]


def test_delete_retries_and_redacts_copilot(archive, monkeypatch):
    stage = documents.storage.stages / "D"
    stage.mkdir()
    (stage / "page.png").write_bytes(b"restored")
    sessions = CopilotSessionRepository()
    session_id = sessions.get_or_create(user_id="reader")["session_id"]
    sessions.append_message(session_id, "assistant", "private excerpt", citations=[{"document_id": "D"}])
    sessions.append_message(session_id, "assistant", "keep", citations=[{"document_id": "other"}])
    with monkeypatch.context() as patch:
        patch.setattr(documents.shutil, "rmtree", Mock(side_effect=OSError("busy")))
        with pytest.raises(OSError):
            documents.delete_document("D", {"_id": "admin"})
    assert documents.doc_repo.find_by_id("D")
    documents.delete_document("D", {"_id": "admin"})
    assert not stage.exists()
    assert documents.doc_repo.find_by_id("D") is None
    assert [m["content"] for m in sessions.get_session(session_id)["messages"]] == ["keep"]


def test_upload_language_label_selects_tamil_reader(monkeypatch):
    from backend.app.ai.ocr.paddleocr_adapter import PaddleOCRAdapter

    reader = Mock()
    factory = Mock(return_value=reader)
    monkeypatch.setattr(PaddleOCRAdapter, "_for_language", staticmethod(factory))
    adapter = object.__new__(PaddleOCRAdapter)
    adapter.language = "en"
    adapter.extract_text("scan.png", "English & Tamil Vernacular")
    factory.assert_called_once_with("ta")
    reader.extract_text.assert_called_once_with("scan.png", "ta")
