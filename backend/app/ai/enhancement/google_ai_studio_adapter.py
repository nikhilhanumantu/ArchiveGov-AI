import os
import json
import base64
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import requests
import cv2

from backend.app.core.config import settings
from backend.app.ai.preprocessing.opencv_processor import opencv_processor
from backend.app.ai.enhancement.realesrgan_adapter import realesrgan_adapter

logger = logging.getLogger("archivegov.google_ai_studio")


class GoogleAIStudioAdapter:
    """
    Direct Native Adapter for Google AI Studio & Gemini API.
    Interacts directly with https://generativelanguage.googleapis.com/v1beta
    using official Google AI Studio API credentials.
    Supports gemini-nano-banana-2.1, gemini-2.5-flash-image, and gemini-2.5-flash.
    """

    DEFAULT_PROMPT = (
        "Master Archival Restoration: Clean uniform ivory parchment substrate, "
        "eliminate water wash stains, paper aging marks, blotches, and foxing. "
        "Enhance printed typography stroke definition and legibility."
    )

    def __init__(self):
        self.api_key = settings.GOOGLE_AI_STUDIO_API_KEY
        self.base_url = settings.GOOGLE_AI_STUDIO_BASE_URL
        self.default_model = "gemini-nano-banana-2.1"

    def get_status(self) -> Dict[str, Any]:
        """Verify API key connectivity with Google AI Studio."""
        if not self.api_key:
            return {
                "name": "Google AI Studio",
                "connected": False,
                "error": "No Google AI Studio API key configured."
            }

        try:
            url = f"{self.base_url}/models?key={self.api_key}"
            resp = requests.get(url, timeout=8)
            if resp.status_code == 200:
                models = [m.get("name", "").replace("models/", "") for m in resp.json().get("models", [])]
                supported = [m for m in models if any(k in m for k in ["gemini", "nano", "banana", "image"])]
                return {
                    "name": "Google AI Studio",
                    "status": "ready",
                    "connected": True,
                    "default_model": self.default_model,
                    "available_models_count": len(models),
                    "supported_models": supported[:10]
                }
            return {
                "name": "Google AI Studio",
                "status": "error",
                "connected": False,
                "http_status": resp.status_code,
                "note": resp.text[:200]
            }
        except Exception as e:
            return {
                "name": "Google AI Studio",
                "status": "unreachable",
                "connected": False,
                "error": str(e)
            }

    def _encode_image_b64(self, image_path: str) -> tuple[str, str]:
        with open(image_path, "rb") as f:
            b64_str = base64.b64encode(f.read()).decode("utf-8")
        ext = Path(image_path).suffix.lower()
        mime = "image/png" if ext == ".png" else "image/jpeg"
        return b64_str, mime

    def enhance(
        self,
        input_path: str,
        output_path: str,
        prompt: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute document enhancement using Google AI Studio.
        Sends image parts with archival restoration prompt.
        Falls back smoothly to local neural substrate restoration if rate limits occur.
        """
        used_key = api_key.strip() if api_key and api_key.strip() else self.api_key
        used_model = model.strip() if model and model.strip() else self.default_model
        used_prompt = prompt.strip() if prompt and prompt.strip() else self.DEFAULT_PROMPT

        src_img = opencv_processor.load_image(input_path)
        (h, w) = src_img.shape[:2]

        try:
            b64_data, mime_type = self._encode_image_b64(input_path)
            url = f"{self.base_url}/models/{used_model}:generateContent?key={used_key}"

            payload = {
                "contents": [
                    {
                        "parts": [
                            {
                                "inline_data": {
                                    "mime_type": mime_type,
                                    "data": b64_data
                                }
                            },
                            {
                                "text": used_prompt
                            }
                        ]
                    }
                ]
            }

            logger.info(f"Invoking Google AI Studio {used_model} on {input_path}")
            resp = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=30)
            res_data = resp.json()

            # Check if image data returned in parts
            image_saved = False
            if resp.status_code == 200:
                candidates = res_data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    for part in parts:
                        inline = part.get("inline_data") or part.get("inlineData")
                        if inline and "data" in inline:
                            img_bytes = base64.b64decode(inline["data"])
                            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                            with open(output_path, "wb") as f:
                                f.write(img_bytes)
                            image_saved = True
                            break

            if image_saved:
                out_img = cv2.imread(output_path)
                (new_h, new_w) = out_img.shape[:2] if out_img is not None else (h, w)
                return {
                    "success": True,
                    "engine": f"Google AI Studio ({used_model})",
                    "model": used_model,
                    "original_dimensions": {"width": w, "height": h},
                    "enhanced_dimensions": {"width": new_w, "height": new_h},
                    "output_path": output_path
                }

            # If quota limit (429) or text-only analysis returned, execute high-fidelity paper cleaner
            fallback = realesrgan_adapter.enhance(input_path, output_path, outscale=2.0)
            fallback["engine"] = f"Google AI Studio Fallback -> {fallback.get('engine')}"
            fallback["google_ai_studio_status"] = resp.status_code
            if resp.status_code == 429:
                fallback["rate_limit_notice"] = (
                    "Google AI Studio free-tier rate limit active. Applied local substrate mark-removal "
                    "and super-resolution automatically."
                )
            return fallback

        except Exception as e:
            logger.exception(f"Google AI Studio error: {e}. Executing fallback.")
            fallback = realesrgan_adapter.enhance(input_path, output_path, outscale=2.0)
            fallback["engine"] = f"Google AI Studio Fallback -> {fallback.get('engine')}"
            fallback["warning"] = str(e)
            return fallback


google_ai_studio_adapter = GoogleAIStudioAdapter()

