import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple

class OpenCVProcessor:
    """Archival document image preprocessor using OpenCV."""

    @staticmethod
    def load_image(image_path: str) -> np.ndarray:
        # Load preserving UTF-8 Windows paths
        data = np.fromfile(image_path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Could not decode image at {image_path}")
        return img

    @staticmethod
    def save_image(img: np.ndarray, output_path: str):
        ext = Path(output_path).suffix.lower()
        success, encoded = cv2.imencode(ext if ext else ".png", img)
        if success:
            encoded.tofile(output_path)
        else:
            raise ValueError(f"Failed to encode image to {output_path}")

    @staticmethod
    def deskew(img: np.ndarray) -> Tuple[np.ndarray, float]:
        """Detect document skew angle and rotate to align."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        # Invert colors so text is white
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        
        # Find coordinates of all text pixels
        coords = np.column_stack(np.where(thresh > 0))
        if len(coords) < 100:
            return img, 0.0
            
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        elif angle > 45:
            angle = 90 - angle
        else:
            angle = -angle
            
        # Avoid rotating if skew is minimal
        if abs(angle) < 0.2 or abs(angle) > 40:
            return img, 0.0

        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated, round(float(angle), 2)

    @staticmethod
    def apply_clahe(img: np.ndarray, clip_limit: float = 2.5, tile_grid_size: int = 8) -> np.ndarray:
        """Apply Contrast Limited Adaptive Histogram Equalization to restore faded text."""
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
        if len(img.shape) == 3:
            # Convert to LAB color space
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l_enhanced = clahe.apply(l)
            enhanced_lab = cv2.merge((l_enhanced, a, b))
            return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        else:
            return clahe.apply(img)

    @staticmethod
    def denoise(img: np.ndarray, strength: int = 10) -> np.ndarray:
        """Archival edge-preserving denoising using Bilateral Filter."""
        if len(img.shape) == 3:
            return cv2.bilateralFilter(img, d=9, sigmaColor=strength * 5, sigmaSpace=strength * 5)
        else:
            return cv2.fastNlMeansDenoising(img, h=strength)

    @staticmethod
    def clean_paper_substrate(img: np.ndarray, target_brightness: float = 248.0) -> np.ndarray:
        """
        Removes paper stains, water marks, yellowing, bleed-through, dirt, and uneven illumination
        via morphological substrate estimation and background shading division, while preserving
        and sharpening all text strokes, signatures, and colored archival seals.
        """
        h, w = img.shape[:2]
        is_color = len(img.shape) == 3 and not (
            np.allclose(img[:, :, 0], img[:, :, 1], atol=3) and np.allclose(img[:, :, 1], img[:, :, 2], atol=3)
        )

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()

        # Step 1: Morphological dilation removes dark text strokes to expose the paper substrate
        k_dilate = max(7, int(min(w, h) * 0.012))
        if k_dilate % 2 == 0:
            k_dilate += 1
        dilated = cv2.dilate(gray, cv2.getStructuringElement(cv2.MORPH_RECT, (k_dilate, k_dilate)))

        # Step 2: Multi-scale median & Gaussian smoothing to construct paper illumination surface
        k_blur = max(31, int(min(w, h) * 0.05))
        if k_blur % 2 == 0:
            k_blur += 1
        bg_surface = cv2.medianBlur(dilated, k_blur)
        bg_smooth = cv2.GaussianBlur(bg_surface, (k_blur, k_blur), 0)

        # Step 3: Background shading division - cancels stains, marks, and illumination gradients
        bg_float = np.maximum(bg_smooth.astype(np.float32), 20.0)
        gray_float = gray.astype(np.float32)

        normalized = (gray_float / bg_float) * target_brightness
        normalized = np.clip(normalized, 0, 255).astype(np.uint8)

        # Step 4: Identify text strokes vs paper background
        text_diff = bg_float - gray_float
        text_mask = (text_diff > 18).astype(np.uint8) * 255

        # Paper background whitening: wipe residual stains on non-text substrate
        cleaned_gray = normalized.copy()
        paper_pixels = (~(text_mask > 0)) & (cleaned_gray > 195)
        cleaned_gray[paper_pixels] = 252

        # Step 5: Enhance contrast and sharpness on ink strokes
        text_indices = text_mask > 0
        if np.any(text_indices):
            # Rich, deep ink density
            cleaned_gray[text_indices] = np.clip(
                cleaned_gray[text_indices].astype(float) * 0.88, 0, 255
            ).astype(np.uint8)

        # Unsharp mask for high-definition letter edge definition
        gaussian = cv2.GaussianBlur(cleaned_gray, (0, 0), sigmaX=1.3)
        sharpened = np.clip(cv2.addWeighted(cleaned_gray, 1.35, gaussian, -0.35, 0), 0, 255).astype(np.uint8)

        if is_color:
            # Preserve colored seals, stamps, and signatures in their authentic color
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)

            # Paper substrate gets clean neutral ivory (a*=128, b*=132)
            a_clean = a.copy()
            b_clean = b.copy()
            paper_non_colored = paper_pixels
            a_clean[paper_non_colored] = 128
            b_clean[paper_non_colored] = 132

            clean_lab = cv2.merge([sharpened, a_clean, b_clean])
            return cv2.cvtColor(clean_lab, cv2.COLOR_LAB2BGR)
        else:
            return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def adaptive_binarize(img: np.ndarray) -> np.ndarray:
        """Produce crisp monochrome document for high-contrast archival reading."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        # Background illumination removal
        dilated = cv2.dilate(gray, np.ones((7, 7), np.uint8))
        bg = cv2.medianBlur(dilated, 21)
        diff = 255 - cv2.absdiff(gray, bg)
        norm = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        
        # Adaptive Gaussian thresholding
        binary = cv2.adaptiveThreshold(
            norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10
        )
        return binary

    def process_pipeline(
        self,
        input_path: str,
        output_path: str,
        do_deskew: bool = True,
        do_clahe: bool = True,
        do_denoise: bool = True,
        do_clean_paper: bool = True
    ) -> Dict[str, Any]:
        """Execute full OpenCV document restoration pipeline."""
        img = self.load_image(input_path)
        angle = 0.0
        
        if do_deskew:
            img, angle = self.deskew(img)
        if do_clean_paper:
            img = self.clean_paper_substrate(img)
        if do_denoise:
            img = self.denoise(img)
        if do_clahe:
            img = self.apply_clahe(img)
            
        self.save_image(img, output_path)
        (h, w) = img.shape[:2]
        
        return {
            "output_path": output_path,
            "dimensions": {"width": w, "height": h},
            "deskew_angle": angle,
            "operations_applied": {
                "deskew": do_deskew,
                "clean_paper": do_clean_paper,
                "denoise": do_denoise,
                "clahe": do_clahe
            }
        }

opencv_processor = OpenCVProcessor()

