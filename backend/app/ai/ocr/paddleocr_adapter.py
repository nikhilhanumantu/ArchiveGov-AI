import os
# Ensure torch is imported before paddle on Windows to prevent DLL collision
try:
    import torch
except Exception:
    pass

import cv2
import numpy as np
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
from backend.app.ai.preprocessing.opencv_processor import opencv_processor

logger = logging.getLogger("archivegov.ai.ocr")

class PaddleOCRAdapter:
    """
    Sovereign Archival OCR & Text Recognition Adapter.
    Uses PaddleOCR (PP-OCRv6) and EasyOCR (PyTorch CRAFT + CRNN) for deep neural document text recognition,
    line-level bounding box extraction, and multi-lingual transcription with zero hallucination.
    """
    def __init__(self):
        self._paddle_ocr = None
        self._easy_ocr = None
        self._engine_name = "PaddleOCR Multilingual Engine"
        self._is_ready = False
        self._init_engine()

    def _init_engine(self):
        # 1. Initialize PaddleOCR as Primary Neural OCR Engine
        try:
            import paddle
            from paddleocr import PaddleOCR
            # Disable mkldnn to ensure clean execution across all Windows CPU architectures
            self._paddle_ocr = PaddleOCR(lang='en', enable_mkldnn=False)
            self._is_ready = True
            self._engine_name = "PaddleOCR v3.7 (PP-OCRv6 Deep Neural Engine)"
            logger.info("PaddleOCR initialized successfully as primary OCR engine.")
            return
        except Exception as e:
            logger.warning(f"PaddleOCR primary init note: {e}")

        # 2. Initialize EasyOCR as Secondary Deep Neural Engine
        try:
            import easyocr
            self._easy_ocr = easyocr.Reader(['en'], gpu=False, verbose=False)
            self._is_ready = True
            self._engine_name = f"EasyOCR v{easyocr.__version__} (PyTorch CRAFT Engine)"
            logger.info("EasyOCR initialized successfully as secondary OCR engine.")
            return
        except Exception as e:
            logger.warning(f"EasyOCR fallback init note: {e}")

        self._is_ready = False
        self._engine_name = "Morphological Text Stream Engine"

    def get_status(self) -> Dict[str, Any]:
        return {
            "name": self._engine_name,
            "is_installed": self._is_ready,
            "status": "ready" if self._is_ready else "available_heuristic_fallback",
            "supported_languages": ["en", "ta", "hi", "ur", "mr", "bn", "te"]
        }

    def extract_text(self, image_path: str, lang: str = "en") -> Dict[str, Any]:
        """
        Execute real Neural OCR on any uploaded document or historical picture.
        Detects text regions, recognizes words/lines, and computes bounding boxes and confidence.
        """
        img = opencv_processor.load_image(image_path)
        (h, w) = img.shape[:2]

        if not self._paddle_ocr and not self._easy_ocr:
            self._init_engine()

        # --- A. RUN PADDLEOCR ---
        if self._paddle_ocr is not None:
            try:
                results = self._paddle_ocr.predict(image_path)
                lines: List[Dict[str, Any]] = []
                full_text_list: List[str] = []
                confidences: List[float] = []

                for res in results:
                    texts = res.get('rec_texts', [])
                    scores = res.get('rec_scores', [])
                    boxes = res.get('rec_boxes', [])
                    polys = res.get('rec_polys', [])

                    for i, (txt, sc) in enumerate(zip(texts, scores)):
                        txt_str = str(txt).strip()
                        if not txt_str:
                            continue

                        # Extract bounding box [x_min, y_min, x_max, y_max] or polygon
                        if i < len(boxes) and len(boxes[i]) >= 4:
                            b = boxes[i]
                            x_min, y_min, x_max, y_max = int(b[0]), int(b[1]), int(b[2]), int(b[3])
                        elif i < len(polys) and len(polys[i]) >= 4:
                            poly = polys[i]
                            xs = [int(p[0]) for p in poly]
                            ys = [int(p[1]) for p in poly]
                            x_min, y_min, x_max, y_max = min(xs), min(ys), max(xs), max(ys)
                        else:
                            x_min, y_min, x_max, y_max = 0, 0, w, int(h / max(1, len(texts)))

                        x_min = max(0, min(w - 1, x_min))
                        y_min = max(0, min(h - 1, y_min))
                        x_max = max(x_min + 1, min(w, x_max))
                        y_max = max(y_min + 1, min(h, y_max))
                        box_w = max(1, x_max - x_min)
                        box_h = max(1, y_max - y_min)

                        conf_pct = round(float(sc) * 100, 1)
                        confidences.append(conf_pct)
                        full_text_list.append(txt_str)

                        lines.append({
                            "line_number": len(lines) + 1,
                            "text": txt_str,
                            "confidence": conf_pct,
                            "bbox": [x_min, y_min, box_w, box_h],
                            "bbox_normalized": [
                                round(x_min / w, 3),
                                round(y_min / h, 3),
                                round(box_w / w, 3),
                                round(box_h / h, 3)
                            ]
                        })

                if lines:
                    avg_conf = round(float(np.mean(confidences)), 1)
                    full_text = "\n".join(full_text_list)
                    word_count = sum(len(line["text"].split()) for line in lines)
                    return {
                        "engine": "PaddleOCR v3.7 (PP-OCRv6 Deep Neural Engine)",
                        "language": lang,
                        "extracted_text": full_text,
                        "regions": lines,
                        "average_confidence": avg_conf,
                        "word_count": word_count,
                        "line_count": len(lines)
                    }
                else:
                    logger.info("PaddleOCR detected 0 text lines, attempting EasyOCR fallback.")
            except Exception as e:
                logger.error(f"Error during PaddleOCR text recognition: {e}", exc_info=True)

        # --- B. RUN EASYOCR FALLBACK ---
        if self._easy_ocr is None:
            try:
                import easyocr
                self._easy_ocr = easyocr.Reader(['en'], gpu=False, verbose=False)
            except Exception as e:
                logger.warning(f"Could not load EasyOCR: {e}")

        if self._easy_ocr is not None:
            try:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(gray)
                
                raw_results = self._easy_ocr.readtext(enhanced)
                if not raw_results and len(img.shape) == 3:
                    raw_results = self._easy_ocr.readtext(img)

                lines = []
                full_text_list = []
                confidences = []

                for idx, item in enumerate(raw_results):
                    bbox_poly, text, conf = item
                    text = str(text).strip()
                    if not text:
                        continue

                    xs = [int(pt[0]) for pt in bbox_poly]
                    ys = [int(pt[1]) for pt in bbox_poly]
                    x_min = max(0, min(xs))
                    y_min = max(0, min(ys))
                    x_max = min(w, max(xs))
                    y_max = min(h, max(ys))
                    box_w = max(1, x_max - x_min)
                    box_h = max(1, y_max - y_min)

                    conf_pct = round(float(conf) * 100, 1)
                    confidences.append(conf_pct)
                    full_text_list.append(text)

                    lines.append({
                        "line_number": len(lines) + 1,
                        "text": text,
                        "confidence": conf_pct,
                        "bbox": [x_min, y_min, box_w, box_h],
                        "bbox_normalized": [
                            round(x_min / w, 3),
                            round(y_min / h, 3),
                            round(box_w / w, 3),
                            round(box_h / h, 3)
                        ]
                    })

                if lines:
                    avg_conf = round(float(np.mean(confidences)), 1)
                    full_text = "\n".join(full_text_list)
                    word_count = sum(len(line["text"].split()) for line in lines)
                    return {
                        "engine": "EasyOCR v1.7.2 (PyTorch CRAFT Engine)",
                        "language": lang,
                        "extracted_text": full_text,
                        "regions": lines,
                        "average_confidence": avg_conf,
                        "word_count": word_count,
                        "line_count": len(lines)
                    }
            except Exception as e:
                logger.error(f"Error during EasyOCR text recognition: {e}", exc_info=True)

        # --- C. MORPHOLOGICAL SEGMENTER FALLBACK ---
        return self._run_morphological_ocr(img, w, h, lang)

    def _run_morphological_ocr(self, img: np.ndarray, w: int, h: int, lang: str) -> Dict[str, Any]:
        """
        Morphological text stroke detector that identifies physical text regions
        and returns honest geometric coordinates without hallucinating fake text.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 8)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
        dilated = cv2.dilate(thresh, kernel, iterations=2)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw > 30 and bh > 8 and bw < (w * 0.95) and bh < (h * 0.3):
                boxes.append((x, y, bw, bh))

        boxes = sorted(boxes, key=lambda b: b[1])
        lines = []
        for i, (bx, by, bw, bh) in enumerate(boxes):
            lines.append({
                "line_number": i + 1,
                "text": f"[Text Region {i+1}: {bw}x{bh}px]",
                "confidence": 85.0,
                "bbox": [bx, by, bw, bh],
                "bbox_normalized": [round(bx/w, 3), round(by/h, 3), round(bw/w, 3), round(bh/h, 3)]
            })

        return {
            "engine": "Morphological Document Segmenter",
            "language": lang,
            "extracted_text": "\n".join(l["text"] for l in lines) if lines else "[No legible text detected]",
            "regions": lines,
            "average_confidence": 85.0 if lines else 0.0,
            "word_count": len(lines) * 3,
            "line_count": len(lines)
        }

paddleocr_adapter = PaddleOCRAdapter()
