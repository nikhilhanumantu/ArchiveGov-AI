from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class RestorationJobCreate(BaseModel):
    document_id: str
    operations: Optional[List[str]] = [
        "damage_analysis",
        "opencv_preprocessing",
        "realesrgan_enhancement",
        "lama_inpainting",
        "paddleocr_extraction",
        "quality_evaluation"
    ]
    model_configuration: Optional[Dict[str, Any]] = None
    processing_parameters: Optional[Dict[str, Any]] = None

class InpaintingRequest(BaseModel):
    document_id: str
    mask_data_base64: Optional[str] = None
    mask_svg: Optional[str] = None
    brush_feather: int = 5
    notes: Optional[str] = "Manual mask selective inpainting"

class ReviewDecisionInput(BaseModel):
    decision: str  # approve, reject, needs_correction
    comments: str
    related_version_id: Optional[str] = None

class FinegrainEnhanceRequest(BaseModel):
    document_id: str
    prompt: Optional[str] = None
    negative_prompt: Optional[str] = None
    seed: float = 42
    upscale_factor: float = 2.0
    controlnet_scale: float = 0.6
    controlnet_decay: float = 1.0
    condition_scale: float = 6.0
    tile_width: float = 112
    tile_height: float = 144
    denoise_strength: float = 0.35
    num_inference_steps: float = 18
    solver: str = "DDIM"
    hf_token: Optional[str] = None

class OpenRouterEnhanceRequest(BaseModel):
    document_id: str
    prompt: Optional[str] = None
    resolution: Optional[str] = "1K"
    aspect_ratio: Optional[str] = None
    api_key: Optional[str] = None

class GoogleAIStudioEnhanceRequest(BaseModel):
    document_id: str
    prompt: Optional[str] = None
    model: Optional[str] = "gemini-nano-banana-2.1"
    api_key: Optional[str] = None




