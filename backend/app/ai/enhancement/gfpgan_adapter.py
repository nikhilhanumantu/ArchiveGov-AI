import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import cv2
import numpy as np

# Ensure torchvision compatibility shim for basicsr / gfpgan
try:
    import torchvision.transforms.functional
    if 'torchvision.transforms.functional_tensor' not in sys.modules:
        sys.modules['torchvision.transforms.functional_tensor'] = torchvision.transforms.functional
except Exception:
    pass

from backend.app.core.config import settings, MODELS_DIR
from backend.app.ai.preprocessing.opencv_processor import opencv_processor

logger = logging.getLogger("archivegov.gfpgan")

class GFPGANAdapter:
    """
    Adapter for GFPGAN (Generative Facial Prior GAN) face restoration and image upscaling.
    Restores degraded historical portraits, ID cards, royal portraiture, and document photography.
    """
    def __init__(self):
        self.weights_dir = MODELS_DIR / "gfpgan"
        model_path = getattr(settings, "GFPGAN_MODEL_PATH", None)
        if model_path and Path(model_path).exists():
            self.weights_file = Path(model_path)
        elif (MODELS_DIR / "gfpgan" / "GFPGANv1.4.pth").exists():
            self.weights_file = MODELS_DIR / "gfpgan" / "GFPGANv1.4.pth"
        elif Path("experiments/pretrained_models/GFPGANv1.4.pth").exists():
            self.weights_file = Path("experiments/pretrained_models/GFPGANv1.4.pth")
        else:
            self.weights_file = self.weights_dir / "GFPGANv1.4.pth"

        self.device = "cuda" if getattr(settings, "DEVICE", "cpu") == "cuda" else "cpu"
        self._restorer = None

    def _get_restorer(self, upscale: int = 2):
        if self._restorer is None:
            try:
                from gfpgan import GFPGANer
                self._restorer = GFPGANer(
                    model_path=str(self.weights_file),
                    upscale=upscale,
                    arch="clean",
                    channel_multiplier=2,
                    bg_upsampler=None,
                    device=self.device
                )
            except Exception as e:
                logger.warning(f"Could not initialize GFPGANer: {e}")
                return None
        return self._restorer

    def get_status(self) -> Dict[str, Any]:
        has_weights = self.weights_file.exists()
        is_installed = False
        try:
            import gfpgan
            is_installed = True
        except ImportError:
            is_installed = False

        return {
            "name": "GFPGAN v1.4 Facial & Portrait Prior Restorer",
            "model_version": "v1.4",
            "device": self.device,
            "weights_present": has_weights,
            "weights_path": str(self.weights_file),
            "library_installed": is_installed,
            "status": "ready" if (has_weights and is_installed) else "available_fallback",
            "capabilities": [
                "Generative Facial Prior restoration",
                "Historical portraiture de-blurring & scratch repair",
                "Passport & deed photograph enhancement",
                "2x / 4x Portrait Super-Resolution"
            ]
        }

    def enhance(
        self,
        input_path: str,
        output_path: str,
        upscale: int = 2,
        only_center_face: bool = False,
        weight: float = 0.5
    ) -> Dict[str, Any]:
        """
        Execute facial restoration and upscaling on the provided archival image.
        """
        img = opencv_processor.load_image(input_path)
        h, w = img.shape[:2]

        restored_img = None
        engine_used = "GFPGAN v1.4 Neural Face Restoration & Upscaling"
        faces_detected = 0

        # Attempt neural GFPGAN inference
        if self.weights_file.exists():
            try:
                restorer = self._get_restorer(upscale=upscale)
                if restorer is not None:
                    # Execute face restoration
                    cropped_faces, restored_faces, restored = restorer.enhance(
                        img,
                        has_aligned=False,
                        only_center_face=only_center_face,
                        paste_back=True,
                        weight=weight
                    )
                    faces_detected = len(cropped_faces) if cropped_faces else 0
                    restored_img = restored
            except Exception as e:
                logger.warning(f"GFPGAN neural inference exception: {e}, using high-fidelity fallback.")

        # Fallback to high-fidelity portrait enhancement if model unavailable or failed
        if restored_img is None:
            engine_used = "Archival High-Fidelity Portrait & Feature Contrast Kernel"
            restored_img = self._run_portrait_fallback(img, upscale=upscale)

        opencv_processor.save_image(restored_img, output_path)
        new_h, new_w = restored_img.shape[:2]

        return {
            "engine": engine_used,
            "original_dimensions": {"width": w, "height": h},
            "enhanced_dimensions": {"width": new_w, "height": new_h},
            "upscale_factor": upscale,
            "faces_detected": faces_detected,
            "output_path": output_path
        }

    def _run_portrait_fallback(self, img: np.ndarray, upscale: int = 2) -> np.ndarray:
        """
        High-precision fallback for historical portraiture and documents:
        Skin-tone preserving bilateral filter, contrast enhancement, unsharp mask, and Lanczos-4 upscaling.
        """
        h, w = img.shape[:2]
        target_w = max(1, int(w * upscale))
        target_h = max(1, int(h * upscale))

        # 1. Lanczos-4 upscaling
        upscaled = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

        # 2. Skin tone & substrate bilateral smoothing
        bilateral = cv2.bilateralFilter(upscaled, d=7, sigmaColor=35, sigmaSpace=35)

        # 3. CLAHE on luminance channel for facial depth
        lab = cv2.cvtColor(bilateral, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)
        enhanced_lab = cv2.merge((l_clahe, a, b))
        enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

        # 4. Unsharp facial sharpening
        gaussian = cv2.GaussianBlur(enhanced_bgr, (0, 0), sigmaX=1.5)
        sharpened = cv2.addWeighted(enhanced_bgr, 1.4, gaussian, -0.4, 0)
        return np.clip(sharpened, 0, 255).astype(np.uint8)

gfpgan_adapter = GFPGANAdapter()

