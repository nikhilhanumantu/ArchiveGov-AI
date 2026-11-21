import logging
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from backend.app.core.dependencies import get_current_user, require_permission
from backend.app.core.permissions import Permission
from backend.app.database.repositories.audit_repo import AuditRepository
from backend.app.database.repositories.copilot_session_repo import CopilotSessionRepository
from backend.app.ai.rag.archive_copilot import archive_copilot
from backend.app.ai.rag.index_service import index_service

logger = logging.getLogger("archivegov.copilot_router")

router = APIRouter(tags=["Archival RAG Copilot"])
audit_repo = AuditRepository()
session_repo = CopilotSessionRepository()

class CopilotQueryRequest(BaseModel):
    query: str = Field(..., min_length=2, description="Natural language question or query regarding archival records")
    department: Optional[str] = Field("all", description="Filter by department, or 'all'")
    document_id: Optional[str] = Field(None, description="Optional specific document context identifier")
    session_id: Optional[str] = Field(None, description="Conversational session identifier for multi-turn thread")
    chat_history: Optional[List[Dict[str, str]]] = Field(None, description="Optional conversational history")

class CopilotQueryResponse(BaseModel):
    success: bool
    session_id: str
    query: str
    answer: str
    engine_used: str
    citations: List[Dict[str, Any]]
    provenance: str = "ai_interpretation"

class CopilotSuggestionsResponse(BaseModel):
    suggestions: List[str]

@router.post("/query", response_model=CopilotQueryResponse)
@router.post("/ask", response_model=CopilotQueryResponse)
def query_copilot(
    request: CopilotQueryRequest,
    user: dict = Depends(require_permission(Permission.DOCUMENT_VIEW))
):
    """
    RAG-powered conversational question-answering over sovereign historical vault.
    Multi-turn session management + hybrid retrieval + inline citations linking to Document Lens.
    """
    try:
        session = session_repo.get_or_create(session_id=request.session_id, user_id=user.get("_id"))
        session_id = session["session_id"]

        # Reconstruct chat history from session if not explicitly provided
        history = request.chat_history
        if not history and session.get("messages"):
            history = [{"role": m["role"], "content": m["content"]} for m in session["messages"][-6:]]

        # Log user message
        session_repo.append_message(
            session_id=session_id,
            role="user",
            content=request.query,
            provenance="source_fact"
        )

        res = archive_copilot.ask(
            query=request.query,
            department=request.department,
            document_id=request.document_id,
            chat_history=history
        )

        # Log assistant response
        session_repo.append_message(
            session_id=session_id,
            role="assistant",
            content=res["answer"],
            citations=res.get("citations", []),
            engine_used=res.get("engine_used"),
            provenance="ai_interpretation"
        )

        audit_repo.log(
            user_id=user["_id"],
            action="COPILOT_QUERY",
            resource_type="copilot",
            resource_id=request.document_id or "vault",
            metadata={
                "session_id": session_id,
                "query": request.query,
                "department": request.department,
                "engine_used": res.get("engine_used"),
                "citations_count": len(res.get("citations", []))
            }
        )

        return {
            "success": True,
            "session_id": session_id,
            "query": request.query,
            "answer": res["answer"],
            "engine_used": res["engine_used"],
            "citations": res["citations"],
            "provenance": "ai_interpretation"
        }
    except Exception as e:
        logger.error(f"Error in query_copilot: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Copilot intelligence service error: {str(e)}")

@router.get("/sessions/{session_id}")
def get_copilot_session(
    session_id: str,
    user: dict = Depends(require_permission(Permission.DOCUMENT_VIEW))
):
    """
    Retrieve message history for an active multi-turn copilot session.
    """
    session = session_repo.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Copilot session not found")
    return session

@router.post("/reindex")
def reindex_vault_corpus(
    user: dict = Depends(require_permission(Permission.DOCUMENT_EDIT))
):
    """
    Re-index all active vault documents into granular RAG chunks.
    """
    total = index_service.reindex_vault()
    audit_repo.log(
        user_id=user["_id"],
        action="COPILOT_REINDEX",
        resource_type="copilot",
        resource_id="vault",
        metadata={"total_chunks_indexed": total}
    )
    return {
        "success": True,
        "message": f"Successfully reindexed {total} archival chunks across vault.",
        "total_chunks": total
    }

@router.get("/suggestions", response_model=CopilotSuggestionsResponse)
def get_copilot_suggestions(
    document_id: Optional[str] = None,
    user: dict = Depends(require_permission(Permission.DOCUMENT_VIEW))
):
    """
    Get dynamic, context-aware prompt suggestions based on records in the repository.
    """
    suggestions = archive_copilot.get_suggestions(document_id=document_id)
    return {"suggestions": suggestions}
