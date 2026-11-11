import uuid
import json
import asyncio
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from backend.app.schemas.restoration import RestorationJobCreate, FinegrainEnhanceRequest, OpenRouterEnhanceRequest, GoogleAIStudioEnhanceRequest
from backend.app.core.dependencies import get_current_user
from backend.app.database.repositories.job_repo import RestorationJobRepository
from backend.app.database.repositories.document_repo import DocumentRepository
from backend.app.database.repositories.audit_repo import AuditRepository
from backend.app.database.repositories.version_repo import DocumentVersionRepository
from backend.app.storage.local_storage import storage
from backend.app.ai.enhancement.finegrain_adapter import finegrain_adapter
from backend.app.ai.enhancement.openrouter_gemini_adapter import openrouter_gemini_adapter
from backend.app.ai.enhancement.google_ai_studio_adapter import google_ai_studio_adapter
from backend.app.workers.job_worker import job_worker

router = APIRouter(prefix="/api/v1/restoration", tags=["Restoration Studio"])
job_repo = RestorationJobRepository()
doc_repo = DocumentRepository()
audit_repo = AuditRepository()
version_repo = DocumentVersionRepository()

@router.post("/jobs")
def create_restoration_job(request: RestorationJobCreate, user: dict = Depends(get_current_user)):
    doc = doc_repo.find_by_id(request.document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Target document not found.")

    job_id = f"JOB-{uuid.uuid4().hex[:8].upper()}"
    job_data = {
        "job_id": job_id,
        "document_id": request.document_id,
        "status": "pending",
        "current_stage": "damage_analysis",
        "progress": 0,
        "requested_operations": request.operations,
        "model_configuration": request.model_configuration or {},
        "processing_parameters": request.processing_parameters or {},
        "created_by": user["_id"]
    }
    created_job = job_repo.create(job_data)
    
    # Trigger background worker execution
    job_worker.process_job_async(job_id)

    audit_repo.log(
        user_id=user["_id"],
        action="RESTORATION_JOB_START",
        resource_type="job",
        resource_id=job_id,
        metadata={"document_id": request.document_id}
    )

    return {
        "success": True,
        "job_id": job_id,
        "message": "Neural restoration job initialized and queued.",
        "job": created_job
    }

@router.get("/jobs/{job_id}")
def get_job_status(job_id: str, user: dict = Depends(get_current_user)):
    job = job_repo.find_by_id(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Restoration job not found.")
    return job

@router.get("/jobs/{job_id}/events")
async def job_events_stream(job_id: str, request: Request):
    """Server-Sent Events (SSE) stream providing real-time pipeline telemetry."""
    async def event_generator():
        last_progress = -1
        last_stage = None
        last_status = None
        
        for _ in range(60):  # Stream up to 60 seconds
            if await request.is_disconnected():
                break
                
            job = job_repo.find_by_id(job_id)
            if not job:
                yield f"data: {json.dumps({'error': 'Job not found'})}\n\n"
                break
                
            current_progress = job.get("progress", 0)
            current_stage = job.get("current_stage")
            current_status = job.get("status")
            
            # Send event when state changes
            if current_progress != last_progress or current_stage != last_stage or current_status != last_status:
                last_progress = current_progress
                last_stage = current_stage
                last_status = current_status
                
                payload = {
                    "job_id": job_id,
                    "status": current_status,
                    "progress": current_progress,
                    "current_stage": current_stage,
                    "stages": job.get("stages", {}),
                    "output_file_key": job.get("output_file_key")
                }
                yield f"data: {json.dumps(payload)}\n\n"
                
            if current_status in ["completed", "failed", "cancelled"]:
                break
                
            await asyncio.sleep(0.8)
            
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str, user: dict = Depends(get_current_user)):
    job = job_repo.find_by_id(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    job_repo.update_status(job_id, "cancelled", job.get("progress", 0), error_message="Cancelled by officer request.")
    return {"message": "Job cancelled."}

@router.get("/finegrain-status")
def get_finegrain_status(user: dict = Depends(get_current_user)):
    """Check connectivity and operational status of finegrain/finegrain-image-enhancer."""
    return finegrain_adapter.get_status()

@router.post("/finegrain-enhance")
def finegrain_enhance_document(request: FinegrainEnhanceRequest, user: dict = Depends(get_current_user)):
    """
    On-demand Finegrain Diffusion Super-Resolution & Paper Restoration.
    Accepts all 13 Finegrain parameters, enhances the document image,
    persists the output in storage, and updates the document record.
    """
    doc = doc_repo.find_by_id(request.document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Target document not found.")

    orig_key = doc.get("original_file_key")
    if not orig_key:
        raise HTTPException(status_code=400, detail="Target document has no original file associated.")

    input_path = storage.get_path("originals", orig_key)
    if not input_path.exists():
        input_path = storage.get_path("working", orig_key)
    if not input_path.exists():
        input_path = storage.get_path("restored", orig_key)
    if not input_path.exists():
        raise HTTPException(status_code=404, detail=f"Source file for {orig_key} not found in archive storage.")

    # Generate persistent restored output file key
    ext = input_path.suffix if input_path.suffix else ".png"
    enhanced_key = f"finegrain_{request.document_id}_{uuid.uuid4().hex[:6]}{ext}"
    output_path = storage.restored / enhanced_key

    # Execute Finegrain enhancement with all 13 parameters
    result = finegrain_adapter.enhance(
        input_path=str(input_path),
        output_path=str(output_path),
        prompt=request.prompt,
        negative_prompt=request.negative_prompt,
        seed=request.seed,
        upscale_factor=request.upscale_factor,
        controlnet_scale=request.controlnet_scale,
        controlnet_decay=request.controlnet_decay,
        condition_scale=request.condition_scale,
        tile_width=request.tile_width,
        tile_height=request.tile_height,
        denoise_strength=request.denoise_strength,
        num_inference_steps=request.num_inference_steps,
        solver=request.solver,  # type: ignore
        hf_token=request.hf_token,
    )

    # Register as archival version
    ver_id = f"VER-FG-{uuid.uuid4().hex[:6].upper()}"
    version_repo.create({
        "version_id": ver_id,
        "document_id": request.document_id,
        "version_number": "v2.1-finegrain",
        "parent_version_id": doc.get("current_version_id"),
        "version_type": "restored",
        "file_key": enhanced_key,
        "model_used": "finegrain/finegrain-image-enhancer",
        "parameters": result.get("parameters_applied", {}),
        "notes": f"Neural diffusion enhancement (scale={request.upscale_factor}x, denoise={request.denoise_strength})"
    })

    # Update document record
    doc_repo.update(request.document_id, {
        "restored_file_key": enhanced_key,
        "current_version_id": ver_id,
        "status": "pending_review"
    })

    audit_repo.log(
        user_id=user["_id"],
        action="FINEGRAIN_ENHANCEMENT",
        resource_type="document",
        resource_id=request.document_id,
        metadata={
            "output_key": enhanced_key,
            "scale": request.upscale_factor,
            "engine": result.get("engine")
        }
    )

    return {
        "success": True,
        "document_id": request.document_id,
        "output_file_key": enhanced_key,
        "file_url": f"/api/v1/files/{enhanced_key}",
        "engine": result.get("engine"),
        "metrics": result
    }

@router.get("/openrouter-status")
def get_openrouter_status(user: dict = Depends(get_current_user)):
    """Check configuration and model parameters for google/gemini-nano-banana-2.1."""
    return openrouter_gemini_adapter.get_status()

@router.post("/openrouter-enhance")
def openrouter_enhance_document(request: OpenRouterEnhanceRequest, user: dict = Depends(get_current_user)):
    """
    Restore document image using Google Gemini Nano Banana 2.1 via OpenRouter Image API.
    Sends source document as input reference and generates master archival restoration.
    """
    doc = doc_repo.find_by_id(request.document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Target document not found.")

    orig_key = doc.get("original_file_key")
    if not orig_key:
        raise HTTPException(status_code=400, detail="Target document has no original file associated.")

    input_path = storage.get_path("originals", orig_key)
    if not input_path.exists():
        input_path = storage.get_path("working", orig_key)
    if not input_path.exists():
        input_path = storage.get_path("restored", orig_key)
    if not input_path.exists():
        raise HTTPException(status_code=404, detail=f"Source file for {orig_key} not found in archive storage.")

    # Target output key in restored storage
    ext = input_path.suffix if input_path.suffix else ".png"
    enhanced_key = f"gemini_nano_{request.document_id}_{uuid.uuid4().hex[:6]}{ext}"
    output_path = storage.restored / enhanced_key

    # Execute enhancement
    result = openrouter_gemini_adapter.enhance_document_image(
        input_path=str(input_path),
        output_path=str(output_path),
        prompt=request.prompt,
        resolution=request.resolution or "1K",
        aspect_ratio=request.aspect_ratio,
        api_key=request.api_key
    )

    # Register document version
    ver_id = f"VER-GN-{uuid.uuid4().hex[:6].upper()}"
    version_repo.create({
        "version_id": ver_id,
        "document_id": request.document_id,
        "version_number": "v2.2-gemini-nano",
        "parent_version_id": doc.get("current_version_id"),
        "version_type": "restored",
        "file_key": enhanced_key,
        "model_used": "OpenRouter: google/gemini-nano-banana-2.1",
        "parameters": {
            "resolution": request.resolution or "1K",
            "aspect_ratio": request.aspect_ratio
        },
        "notes": f"Restoration via Google Gemini Nano Banana 2.1 ({result.get('engine')})"
    })

    # Update document record
    doc_repo.update(request.document_id, {
        "restored_file_key": enhanced_key,
        "current_version_id": ver_id,
        "status": "pending_review"
    })

    audit_repo.log(
        user_id=user["_id"],
        action="OPENROUTER_GEMINI_ENHANCEMENT",
        resource_type="document",
        resource_id=request.document_id,
        metadata={
            "output_key": enhanced_key,
            "engine": result.get("engine"),
            "model": "google/gemini-nano-banana-2.1"
        }
    )

    return {
        "success": True,
        "document_id": request.document_id,
        "output_file_key": enhanced_key,
        "file_url": f"/api/v1/files/{enhanced_key}",
        "engine": result.get("engine"),
        "metrics": result,
        "credit_notice": result.get("credit_notice")
    }

@router.get("/google-ai-studio-status")
def get_google_ai_studio_status(user: dict = Depends(get_current_user)):
    """Check connectivity and available models directly on Google AI Studio."""
    return google_ai_studio_adapter.get_status()

@router.post("/google-ai-studio-enhance")
def google_ai_studio_enhance_document(request: GoogleAIStudioEnhanceRequest, user: dict = Depends(get_current_user)):
    """
    Direct restoration via Google AI Studio API (Gemini Nano Banana 2.1 / Gemini 2.5 Flash).
    Communicates directly with Google Generative Language endpoints.
    """
    doc = doc_repo.find_by_id(request.document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Target document not found.")

    orig_key = doc.get("original_file_key")
    if not orig_key:
        raise HTTPException(status_code=400, detail="Target document has no original file associated.")

    input_path = storage.get_path("originals", orig_key)
    if not input_path.exists():
        input_path = storage.get_path("working", orig_key)
    if not input_path.exists():
        input_path = storage.get_path("restored", orig_key)
    if not input_path.exists():
        raise HTTPException(status_code=404, detail=f"Source file for {orig_key} not found in archive storage.")

    ext = input_path.suffix if input_path.suffix else ".png"
    enhanced_key = f"google_studio_{request.document_id}_{uuid.uuid4().hex[:6]}{ext}"
    output_path = storage.restored / enhanced_key

    result = google_ai_studio_adapter.enhance(
        input_path=str(input_path),
        output_path=str(output_path),
        prompt=request.prompt,
        model=request.model or "gemini-nano-banana-2.1",
        api_key=request.api_key
    )

    ver_id = f"VER-GS-{uuid.uuid4().hex[:6].upper()}"
    version_repo.create({
        "version_id": ver_id,
        "document_id": request.document_id,
        "version_number": "v2.3-google-studio",
        "parent_version_id": doc.get("current_version_id"),
        "version_type": "restored",
        "file_key": enhanced_key,
        "model_used": f"Google AI Studio: {result.get('model', 'gemini-nano-banana-2.1')}",
        "parameters": {
            "model": request.model or "gemini-nano-banana-2.1"
        },
        "notes": f"Restoration via Google AI Studio ({result.get('engine')})"
    })

    doc_repo.update(request.document_id, {
        "restored_file_key": enhanced_key,
        "current_version_id": ver_id,
        "status": "pending_review"
    })

    audit_repo.log(
        user_id=user["_id"],
        action="GOOGLE_AI_STUDIO_ENHANCEMENT",
        resource_type="document",
        resource_id=request.document_id,
        metadata={
            "output_key": enhanced_key,
            "engine": result.get("engine"),
            "model": request.model or "gemini-nano-banana-2.1"
        }
    )

    return {
        "success": True,
        "document_id": request.document_id,
        "output_file_key": enhanced_key,
        "file_url": f"/api/v1/files/{enhanced_key}",
        "engine": result.get("engine"),
        "metrics": result,
        "rate_limit_notice": result.get("rate_limit_notice")
    }




