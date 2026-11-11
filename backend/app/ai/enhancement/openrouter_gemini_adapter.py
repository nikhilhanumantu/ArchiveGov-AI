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

logger = logging.getLogger("archivegov.openrouter_gemini")


class OpenRouterGeminiAdapter:
    """
    Adapter for Google Gemini Nano Banana 2.1 (google/gemini-nano-banana-2.1)
    via OpenRouter Image API & Chat Completions API.
    Supports Image-to-Image document restoration and Text-to-Image generation.
    """

    DEFAULT_PROMPT = (
        "Restore this historical government document image to master archival preservation quality: "
        "clean uniform ivory parchment paper, completely remove all water stains, blotches, tears, "
        "and discoloration marks on the paper, make all printed and written typography sharp, crisp, "
        "high-contrast dark black ink, and perfectly legible. Museum preservation grade."
    )

    def __init__(self):
        self.api_key = settings.OPENROUTER_API_KEY
        self.base_url = settings.OPENROUTER_BASE_URL
        self.image_model = settings.OPENROUTER_IMAGE_MODEL

    def get_status(self) -> Dict[str, Any]:
        """Check API configuration and connectivity."""
        has_key = bool(self.api_key and self.api_key.startswith("sk-or-v1-"))
        return {
            "name": "Google Gemini Nano Banana 2.1",
            "model_id": self.image_model,
            "provider": "OpenRouter",
            "api_key_configured": has_key,
            "api_endpoint": f"{self.base_url}/images",
            "supported_resolutions": ["1K", "2K", "4K"],
            "supported_aspect_ratios": [
                "1:1", "1:4", "1:8", "2:3", "3:2", "3:4", "4:1", "4:3",
                "4:5", "5:4", "8:1", "9:16", "16:9", "21:9"
            ]
        }

    def _encode_image_to_data_url(self, image_path: str) -> str:
        """Encode local image file to base64 data URL."""
        with open(image_path, "rb") as f:
            b64_data = base64.b64encode(f.read()).decode("utf-8")
        
        ext = Path(image_path).suffix.lower()
        mime_type = "image/png" if ext == ".png" else "image/jpeg"
        return f"data:{mime_type};base64,{b64_data}"

    def enhance_document_image(
        self,
        input_path: str,
        output_path: str,
        prompt: Optional[str] = None,
        resolution: str = "1K",
        aspect_ratio: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Enhance a degraded archival document image using google/gemini-nano-banana-2.1
        via OpenRouter's Image-to-Image API.
        Falls back smoothly to local neural substrate restoration if credits or quota are unavailable.
        """
        used_key = api_key.strip() if api_key and api_key.strip() else self.api_key
        used_prompt = prompt.strip() if prompt and prompt.strip() else self.DEFAULT_PROMPT

        headers = {
            "Authorization": f"Bearer {used_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://archivegov.ai",
            "X-Title": "ArchiveGov AI Historical Document Restorer"
        }

        # Load input dimensions for telemetry
        src_img = opencv_processor.load_image(input_path)
        (h, w) = src_img.shape[:2]

        try:
            # 1. Prepare base64 data URL for Image-to-Image reference
            data_url = self._encode_image_to_data_url(input_path)

            payload: Dict[str, Any] = {
                "model": self.image_model,
                "prompt": used_prompt,
                "n": 1,
                "resolution": resolution if resolution in ["1K", "2K", "4K"] else "1K",
                "input_references": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_url
                        }
                    }
                ]
            }

            if aspect_ratio:
                payload["aspect_ratio"] = aspect_ratio

            logger.info(f"Calling OpenRouter {self.image_model} for image enhancement on {input_path}")
            response = requests.post(
                f"{self.base_url}/images",
                headers=headers,
                json=payload,
                timeout=45
            )

            res_json = response.json()

            # If success and image data received
            if response.status_code == 200 and "data" in res_json and len(res_json["data"]) > 0:
                first_img = res_json["data"][0]
                b64_bytes = base64.b64decode(first_img["b64_json"])

                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "wb") as f:
                    f.write(b64_bytes)

                saved_img = cv2.imread(output_path)
                (new_h, new_w) = saved_img.shape[:2] if saved_img is not None else (h, w)

                return {
                    "success": True,
                    "engine": f"OpenRouter ({self.image_model})",
                    "model": self.image_model,
                    "resolution": resolution,
                    "prompt_used": used_prompt,
                    "usage": res_json.get("usage", {}),
                    "original_dimensions": {"width": w, "height": h},
                    "enhanced_dimensions": {"width": new_w, "height": new_h},
                    "output_path": output_path
                }

            # Handle API errors (e.g. 402 Insufficient credits)
            err_msg = res_json.get("error", {}).get("message", response.text)
            err_code = res_json.get("error", {}).get("code", response.status_code)
            logger.warning(
                f"OpenRouter {self.image_model} call returned status {response.status_code} ({err_msg}). "
                f"Activating high-fidelity substrate restoration fallback."
            )

            # Robust fallback to local substrate normalization & stain suppression
            fallback_res = realesrgan_adapter.enhance(input_path, output_path, outscale=2.0)
            fallback_res["engine"] = f"OpenRouter Gemini Nano Fallback -> {fallback_res.get('engine')}"
            fallback_res["openrouter_status"] = err_code
            fallback_res["openrouter_notice"] = err_msg
            if err_code == 402:
                fallback_res["credit_notice"] = (
                    "OpenRouter account requires credits ($1.00 minimum for image generation). "
                    "Add credits at https://openrouter.ai/settings/credits to unlock cloud GPU generation."
                )

            return fallback_res

        except Exception as e:
            logger.exception(f"OpenRouter image generation exception: {e}. Executing fallback.")
            fallback_res = realesrgan_adapter.enhance(input_path, output_path, outscale=2.0)
            fallback_res["engine"] = f"OpenRouter Fallback -> {fallback_res.get('engine')}"
            fallback_res["warning"] = str(e)
            return fallback_res


openrouter_gemini_adapter = OpenRouterGeminiAdapter()

