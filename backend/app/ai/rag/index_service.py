import re
import math
import uuid
import logging
from typing import List, Dict, Any, Optional, Set
from pathlib import Path
from bson import json_util

from backend.app.database.repositories.document_repo import DocumentRepository
from backend.app.database.repositories.ocr_repo import OCRRepository
from backend.app.database.repositories.ocr_region_repo import OCRRegionRepository
from backend.app.database.repositories.rag_chunk_repo import RAGChunkRepository

logger = logging.getLogger("archivegov.ai.rag.index")

ARCHIVAL_BOOST_TERMS: Set[str] = {
    "acres", "rupees", "annas", "pies", "survey", "collectorate",
    "epidemic", "kuddalur", "south", "arcot", "sanad", "ryotwari",
    "patta", "deed", "tenancy", "settlement", "assessment", "alluvial",
    "quarantine", "influenza", "cholera", "plague", "bengal", "bombay",
    "high", "court", "decree", "sheristadar", "vellar", "embankment",
    "kari", "man", "ryot", "collector", "gazette", "subordinate"
}

STOPWORDS: Set[str] = {
    "a", "an", "the", "and", "or", "in", "on", "at", "to", "for", "with",
    "by", "of", "from", "is", "was", "are", "were", "it", "this", "that",
    "these", "those", "be", "been", "has", "had", "have", "do", "does",
    "did", "as", "about", "what", "which", "who", "when", "where", "why",
    "how", "can", "could", "would", "should", "all", "any", "some"
}

class IndexService:
    """
    Hybrid Indexing & Retrieval Service for Sovereign Archival Records (Module 1 Copilot).
    Maintains chunked index in `rag_chunks` collection with BM25 + TF-IDF scoring and metadata boosting.
    """
    def __init__(self):
        self.doc_repo = DocumentRepository()
        self.ocr_repo = OCRRepository()
        self.region_repo = OCRRegionRepository()
        self.chunk_repo = RAGChunkRepository()

    def tokenize(self, text: str) -> List[str]:
        if not text:
            return []
        raw_tokens = re.findall(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)?", text.lower())
        return [t for t in raw_tokens if len(t) > 1 and t not in STOPWORDS]

    def index_document(self, doc_id: str) -> int:
        """
        Split a document's OCR and metadata into overlapping ~120-word chunks and persist to rag_chunks.
        """
        doc = self.doc_repo.find_by_id(doc_id)
        if not doc:
            return 0

        # Remove existing chunks for this doc to keep index synchronized
        self.chunk_repo.delete_by_doc(doc_id)

        ocr_record = self.ocr_repo.find_by_document_id(doc_id)
        ocr_text = ""
        if ocr_record:
            ocr_text = ocr_record.get("corrected_text") or ocr_record.get("extracted_text") or ""
        if not ocr_text:
            ocr_text = doc.get("ocr_text", "")

        combined_text = f"{doc.get('title', '')}\nReference: {doc.get('archive_reference_id', '')}\n" \
                        f"Department: {doc.get('department_id', '')}\nDate: {doc.get('original_date', '')}\n" \
                        f"Description: {doc.get('description', '')}\n\n{ocr_text}"

        words = combined_text.split()
        chunk_size = 120
        overlap = 25
        step = max(1, chunk_size - overlap)

        chunks_to_insert = []
        chunk_order = 0

        if not words:
            words = [doc.get("title", "Untitled Archival Record")]

        for i in range(0, len(words), step):
            chunk_slice = words[i:i + chunk_size]
            chunk_text = " ".join(chunk_slice)
            if not chunk_text.strip():
                continue

            chunk_id = f"chk_{doc_id}_{chunk_order}"
            tokens = self.tokenize(chunk_text)

            # Link to first region if available
            region_id = None
            page_number = 1
            if chunk_order == 0:
                regions = self.region_repo.find_by_doc(doc_id, page_number=1)
                if regions:
                    region_id = regions[0].get("region_id")
                    page_number = regions[0].get("page_number", 1)

            chunks_to_insert.append({
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "archive_reference_id": doc.get("archive_reference_id", doc_id),
                "title": doc.get("title", "Untitled Archival Record"),
                "department_id": doc.get("department_id", "General Government Archives"),
                "original_date": str(doc.get("original_date", "")),
                "page_number": page_number,
                "region_id": region_id,
                "chunk_order": chunk_order,
                "chunk_text": chunk_text,
                "token_count": len(tokens),
                "keywords": list(set(tokens[:20])),
                "archive_url": f"/app/archive/{doc_id}",
                "ocr_url": f"/app/ocr?document_id={doc_id}",
                "lens_url": f"/lens/{doc_id}?page={page_number}" + (f"&region={region_id}" if region_id else "")
            })
            chunk_order += 1

        if chunks_to_insert:
            self.chunk_repo.insert_chunks(chunks_to_insert)
            logger.info(f"Indexed {len(chunks_to_insert)} chunks for document {doc_id}")

        return len(chunks_to_insert)

    def reindex_vault(self) -> int:
        """Re-indexes all active non-deleted documents in the vault."""
        all_docs = self.doc_repo.list_all(limit=500)
        total_chunks = 0
        for d in all_docs:
            if not d.get("is_deleted"):
                total_chunks += self.index_document(d["document_id"])
        return total_chunks

    def search_chunks(
        self,
        query: str,
        department: Optional[str] = None,
        doc_id: Optional[str] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Hybrid BM25 + TF-IDF cosine retrieval with metadata boosting.
        """
        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []

        # Ensure index has content; if empty, index all documents
        if self.chunk_repo.count() == 0:
            self.reindex_vault()

        # Retrieve candidate chunks
        if doc_id and doc_id != "all":
            chunks = self.chunk_repo.find_by_doc(doc_id)
        else:
            chunks = self.chunk_repo.search_all(department=department)

        # Fallback if no chunks found in DB (e.g. file-based fallback)
        if not chunks:
            all_docs = self.doc_repo.list_all(limit=100)
            for d in all_docs:
                self.index_document(d["document_id"])
            if doc_id and doc_id != "all":
                chunks = self.chunk_repo.find_by_doc(doc_id)
            else:
                chunks = self.chunk_repo.search_all(department=department)

        if not chunks:
            return []

        # Precompute document frequencies for BM25/TF-IDF
        N = len(chunks)
        df: Dict[str, int] = {}
        chunk_token_lists = []
        for c in chunks:
            tokens = self.tokenize(c.get("chunk_text", ""))
            chunk_token_lists.append(tokens)
            unique_tokens = set(tokens)
            for t in unique_tokens:
                df[t] = df.get(t, 0) + 1

        avg_doc_len = sum(len(tl) for tl in chunk_token_lists) / max(1, N)
        k1 = 1.5
        b = 0.75

        scored_chunks = []
        for idx, chunk in enumerate(chunks):
            tokens = chunk_token_lists[idx]
            doc_len = len(tokens)
            score = 0.0

            # Term frequencies in chunk
            tf: Dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1

            for qt in query_tokens:
                if qt in tf:
                    # BM25 component
                    n_q = df.get(qt, 1)
                    idf = math.log(1.0 + (N - n_q + 0.5) / (n_q + 0.5))
                    term_freq = tf[qt]
                    bm25_part = idf * (term_freq * (k1 + 1.0)) / (term_freq + k1 * (1.0 - b + b * (doc_len / max(1.0, avg_doc_len))))
                    score += max(0.0, bm25_part)

                    # Boost archival terms
                    if qt in ARCHIVAL_BOOST_TERMS:
                        score += 0.35

            # Metadata boosts
            metadata_str = f"{chunk.get('title', '')} {chunk.get('archive_reference_id', '')} {chunk.get('original_date', '')} {chunk.get('department_id', '')}".lower()
            for qt in query_tokens:
                if qt in metadata_str:
                    score += 0.5
                # Exact date match boost (e.g. 1912, 1914, 1918)
                if len(qt) == 4 and qt.isdigit() and qt in metadata_str:
                    score += 1.2

            if score > 0:
                # Normalize score roughly between 0.0 and 1.0
                norm_score = round(min(1.0, score / 8.0), 3)
                snippet = chunk.get("chunk_text", "")
                if len(snippet) > 280:
                    snippet = snippet[:280] + "..."

                scored_chunks.append({
                    "chunk_id": chunk.get("chunk_id"),
                    "doc_id": chunk.get("doc_id"),
                    "archive_reference_id": chunk.get("archive_reference_id"),
                    "title": chunk.get("title"),
                    "original_date": chunk.get("original_date"),
                    "department_id": chunk.get("department_id"),
                    "page_number": chunk.get("page_number", 1),
                    "region_id": chunk.get("region_id"),
                    "relevance_score": norm_score,
                    "snippet": snippet,
                    "archive_url": chunk.get("archive_url"),
                    "ocr_url": chunk.get("ocr_url"),
                    "lens_url": chunk.get("lens_url") or f"/lens/{chunk.get('doc_id')}?page={chunk.get('page_number', 1)}"
                })

        scored_chunks.sort(key=lambda x: x["relevance_score"], reverse=True)
        return scored_chunks[:top_k]

index_service = IndexService()
