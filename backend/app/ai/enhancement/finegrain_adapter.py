import os
import shutil
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Literal
import cv2
import numpy as np

from backend.app.core.config import settings
from backend.app.ai.preprocessing.opencv_processor import opencv_processor
from backend.app.ai.enhancement.realesrgan_adapter import realesrgan_adapter

logger = logging.getLogger("archivegov.finegrain")


class FinegrainEnhancerAdapter:
    """
    Adapter for finegrain/finegrain-image-enhancer via gradio_client.
    Provides diffusion-based neural super-resolution, ControlNet tile-guided
    detail synthesis, and archival paper substrate refinement.
    """

    DEFAULT_PROMPT = (
        "Master archival government document, high-resolution crisp printed typography, "
        "clean uniform ivory parchment paper, clear sharp letters, archival preservation grade"
    )
    DEFAULT_NEGATIVE_PROMPT = (
        "blurry, noise, artifacts, distorted letters, stains, blotches, darkness, grain, low resolution"
    )

    def __init__(self):
        self.space_id = settings.FINEGRAIN_SPACE_ID
        self.hf_token = settings.HF_TOKEN or os.environ.get("HF_TOKEN")
        self._client = None
        self._last_token = None
        self._is_available = None

    def _get_client(self, token: Optional[str] = None):
        used_token = token or self.hf_token or os.environ.get("HF_TOKEN")
        if self._client is None or used_token != self._last_token:
            try:
                from gradio_client import Client
                kwargs = {}
                if used_token:
                    kwargs["hf_token"] = used_token
                self._client = Client(self.space_id, **kwargs)
                self._last_token = used_token
                self._is_available = True
                logger.info(f"Finegrain client initialized successfully from {self.space_id}")
            except Exception as e:
                self._is_available = False
                logger.warning(f"Could not connect to Finegrain space ({self.space_id}): {e}")
                raise e
        return self._client

    def get_status(self) -> Dict[str, Any]:
        """Report engine status and availability."""
        try:
            self._get_client()
            return {
                "name": "Finegrain Image Enhancer",
                "space_id": self.space_id,
                "status": "ready",
                "connected": True,
                "engine_type": "Diffusion ControlNet Super-Resolution"
            }
        except Exception as e:
            return {
                "name": "Finegrain Image Enhancer",
                "space_id": self.space_id,
                "status": "fallback_available",
                "connected": False,
                "note": str(e)
            }

    def enhance(
        self,
        input_path: str,
        output_path: str,
        prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        seed: float = 42,
        upscale_factor: float = 2.0,
        controlnet_scale: float = 0.6,
        controlnet_decay: float = 1.0,
        condition_scale: float = 6.0,
        tile_width: float = 112,
        tile_height: float = 144,
        denoise_strength: float = 0.35,
        num_inference_steps: float = 18,
        solver: Literal["DDIM", "DPMSolver"] = "DDIM",
        hf_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Enhance image details and resolution using Finegrain Image Enhancer.
        Accepts all 13 Finegrain API parameters.
        Falls back to Real-ESRGAN/Lanczos super-resolution if network fails.
        """
        img = opencv_processor.load_image(input_path)
        (h, w) = img.shape[:2]

        used_prompt = prompt.strip() if prompt and prompt.strip() else self.DEFAULT_PROMPT
        used_negative_prompt = (
            negative_prompt.strip() if negative_prompt and negative_prompt.strip() else self.DEFAULT_NEGATIVE_PROMPT
        )

        try:
            from gradio_client import handle_file
            client = self._get_client(token=hf_token)

            logger.info(
                f"Executing Finegrain enhancement on {input_path} "
                f"(scale={upscale_factor}, steps={num_inference_steps}, denoise={denoise_strength})"
            )

            result = client.predict(
                input_image=handle_file(str(input_path)),
                prompt=used_prompt,
                negative_prompt=used_negative_prompt,
                seed=float(seed),
                upscale_factor=float(upscale_factor),
                controlnet_scale=float(controlnet_scale),
                controlnet_decay=float(controlnet_decay),
                condition_scale=float(condition_scale),
                tile_width=float(tile_width),
                tile_height=float(tile_height),
                denoise_strength=float(denoise_strength),
                num_inference_steps=float(num_inference_steps),
                solver=solver,
                api_name="/process",
            )

            # result is tuple[filepath | None, filepath | None] | None
            # index 0 is Before, index 1 is After (Enhanced)
            enhanced_temp_path = None
            before_temp_path = None
            if isinstance(result, (list, tuple)) and len(result) >= 2:
                before_temp_path = result[0]
                enhanced_temp_path = result[1]
            elif isinstance(result, (list, tuple)) and len(result) == 1:
                enhanced_temp_path = result[0]
            elif isinstance(result, str):
                enhanced_temp_path = result

            if not enhanced_temp_path or not Path(enhanced_temp_path).exists():
                raise RuntimeError(f"Finegrain returned invalid output: {result}")

            # Read enhanced image and save to target path
            enhanced_img = cv2.imread(str(enhanced_temp_path))
            if enhanced_img is None:
                shutil.copy2(str(enhanced_temp_path), str(output_path))
                enhanced_img = opencv_processor.load_image(output_path)
            else:
                opencv_processor.save_image(enhanced_img, output_path)

            (new_h, new_w) = enhanced_img.shape[:2]

            return {
                "success": True,
                "engine": "Finegrain AI Image Enhancer (finegrain/finegrain-image-enhancer)",
                "space_id": self.space_id,
                "parameters_applied": {
                    "prompt": used_prompt,
                    "negative_prompt": used_negative_prompt,
                    "seed": seed,
                    "upscale_factor": upscale_factor,
                    "controlnet_scale": controlnet_scale,
                    "controlnet_decay": controlnet_decay,
                    "condition_scale": condition_scale,
                    "tile_width": tile_width,
                    "tile_height": tile_height,
                    "denoise_strength": denoise_strength,
                    "num_inference_steps": num_inference_steps,
                    "solver": solver,
                },
                "original_dimensions": {"width": w, "height": h},
                "enhanced_dimensions": {"width": new_w, "height": new_h},
                "scale_factor": upscale_factor,
                "output_path": output_path,
                "before_image_path": before_temp_path,
            }

        except Exception as e:
            logger.warning(f"Finegrain enhancement failed or timed out: {e}. Executing robust fallback.")
            fallback_res = realesrgan_adapter.enhance(input_path, output_path, outscale=upscale_factor)
            fallback_res["engine"] = f"Finegrain Fallback -> {fallback_res.get('engine')}"
            fallback_res["warning"] = str(e)
            return fallback_res


finegrain_adapter = FinegrainEnhancerAdapter()
