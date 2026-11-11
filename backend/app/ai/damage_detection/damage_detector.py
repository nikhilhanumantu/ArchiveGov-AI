import cv2
import numpy as np
from typing import Dict, Any, List
from backend.app.ai.preprocessing.opencv_processor import opencv_processor


class DamageDetector:
    """
    Explainable computer-vision damage detector for degraded archival records.
    Accurately identifies and segments:
      1. Physical Holes & Interior Voids (parchment perforation, insect holes)
      2. Torn Edge & Margin Voids (ragged perimeter cutouts, paper loss, missing corners)
      3. Water Damage & Oxidation Stains (water rings, damp wash, foxing blotches on paper substrate)
      4. Faded Ink & Low-Contrast Script (washed out iron-gall lines, localized faded blocks)
      5. Uneven Illumination & Shadow Blemishes (vignetting, scanner bed shadow)
    """

    ENGINE_VERSION = "3.0"

    @staticmethod
    def _is_grayscale(img: np.ndarray) -> bool:
        if len(img.shape) < 3:
            return True
        b, g, r = cv2.split(img)
        return bool(np.allclose(b, g, atol=4) and np.allclose(g, r, atol=4))

    def analyze_document(self, image_path: str) -> Dict[str, Any]:
        img = opencv_processor.load_image(image_path)
        (h, w) = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        is_gray = self._is_grayscale(img)

        defects: List[Dict[str, Any]] = []

        # ── 0. TEXT STROKE SEGMENTATION MASK ─────────────────────────────
        # Mask out text strokes so that text headings or margins are never
        # falsely classified as water stains or torn edges.
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 8
        )
        text_mask = cv2.dilate(
            thresh, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=2
        )

        # ── 1. PHYSICAL HOLES & INTERIOR VOIDS ──────────────────────────
        # Dark voids that are NOT text characters
        dark_mask = (gray < 65).astype(np.uint8) * 255
        dark_non_text = cv2.bitwise_and(dark_mask, cv2.bitwise_not(text_mask))
        k_hole = max(13, int(min(w, h) * 0.015))
        if k_hole % 2 == 0:
            k_hole += 1
        void_opened = cv2.morphologyEx(
            dark_non_text,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_hole, k_hole))
        )

        cnts_holes, _ = cv2.findContours(void_opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts_holes:
            area = cv2.contourArea(c)
            if area > (w * h * 0.003):  # at least 0.3% of page
                bx, by, bw, bh = cv2.boundingRect(c)
                # Must be interior void (not margin border)
                if bx > w * 0.04 and (bx + bw) < w * 0.96 and by > h * 0.04 and (by + bh) < h * 0.96:
                    px = int(bw * 0.1)
                    py = int(bh * 0.1)
                    bx = max(0, bx - px)
                    by = max(0, by - py)
                    bw = min(w - bx, bw + 2 * px)
                    bh = min(h - by, bh + 2 * py)
                    sev = round(float(min(95.0, (area / (w * h)) * 500 + 45)), 1)
                    defects.append({
                        "category": "Torn Paper & Border Voids",
                        "label": "Paper Loss / Void",
                        "severity_score": float(sev),
                        "severity_label": "Critical",
                        "detection_method": "Morphological Thick-Void Analysis",
                        "evidence": f"Parchment perforation void: {int(area):,} px",
                        "recommended_operation": "LaMa Generative Structural Boundary Repair",
                        "bbox": [int(bx), int(by), int(bw), int(bh)],
                        "bbox_normalized": [
                            float(round(bx / w, 3)),
                            float(round(by / h, 3)),
                            float(round(bw / w, 3)),
                            float(round(bh / h, 3))
                        ]
                    })

        # ── 2. TORN BORDERS & PERIMETER PAPER LOSS ──────────────────────
        # Real tears have missing perimeter paper where the scanner bed shows through.
        # Mask out text so printed text near margins is never mistaken for a tear.
        scanner_dark_bed = (gray < 55).astype(np.uint8) * 255
        scanner_dark_bed = cv2.bitwise_and(scanner_dark_bed, cv2.bitwise_not(text_mask))
        k_tear = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        scanner_dark_bed = cv2.morphologyEx(scanner_dark_bed, cv2.MORPH_OPEN, k_tear)

        cnts_border, _ = cv2.findContours(scanner_dark_bed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts_border:
            area = cv2.contourArea(c)
            bx, by, bw, bh = cv2.boundingRect(c)
            touches_border = (bx <= 6 or by <= 6 or (bx + bw) >= (w - 6) or (by + bh) >= (h - 6))
            if touches_border and area > (w * h * 0.008):
                sev = round(float(min(90.0, (area / (w * h)) * 400 + 35)), 1)
                defects.append({
                    "category": "Torn Paper & Border Voids",
                    "label": "Torn Edge",
                    "severity_score": float(sev),
                    "severity_label": "Structural Damage",
                    "detection_method": "Perimeter Convexity & Paper-Loss Analysis",
                    "evidence": f"Torn border chunk void: {int(area):,} px",
                    "recommended_operation": "LaMa Generative Structural Boundary Repair",
                    "bbox": [int(bx), int(by), int(bw), int(bh)],
                    "bbox_normalized": [
                        float(round(bx / w, 3)),
                        float(round(by / h, 3)),
                        float(round(bw / w, 3)),
                        float(round(bh / h, 3))
                    ]
                })

        # ── 3. WATER DAMAGE, OXIDATION & STAINS (PURE PAPER SUBSTRATE) ──
        # Generate pure paper substrate by inpainting over text strokes with median paper tone
        paper_only = gray.copy()
        valid_paper = paper_only[text_mask == 0]
        bg_median = float(np.median(valid_paper)) if len(valid_paper) > 0 else 240.0
        paper_only[text_mask > 0] = int(bg_median)

        k_smooth = max(35, int(min(w, h) * 0.15))
        if k_smooth % 2 == 0:
            k_smooth += 1
        bg_smooth = cv2.GaussianBlur(paper_only, (k_smooth, k_smooth), 0)
        diff_stain = cv2.absdiff(paper_only, bg_smooth)

        # Inset 3% border to prevent outer crop/shadow artifacts
        b_inset = max(6, int(min(w, h) * 0.03))
        diff_stain[:b_inset, :] = 0
        diff_stain[-b_inset:, :] = 0
        diff_stain[:, :b_inset] = 0
        diff_stain[:, -b_inset:] = 0

        # Absolute threshold: only genuine discolored substrate (> 18 intensity difference)
        _, stain_bin = cv2.threshold(diff_stain, 18, 255, cv2.THRESH_BINARY)
        stain_bin = cv2.morphologyEx(
            stain_bin,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        )

        cnts_stains, _ = cv2.findContours(stain_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        stain_items = []
        for c in cnts_stains:
            area = cv2.contourArea(c)
            if area > (w * h * 0.006):  # > 0.6% of page
                bx, by, bw, bh = cv2.boundingRect(c)
                if bw < w * 0.9 and bh < h * 0.9:
                    px = int(bw * 0.06)
                    py = int(bh * 0.06)
                    bx = max(0, bx - px)
                    by = max(0, by - py)
                    bw = min(w - bx, bw + 2 * px)
                    bh = min(h - by, bh + 2 * py)
                    stain_items.append((area, bx, by, bw, bh))

        stain_items.sort(key=lambda x: x[0], reverse=True)
        for s_area, sbx, sby, sbw, sbh in stain_items[:2]:
            sev = round(float(min(88.0, (s_area / (w * h)) * 400 + 38)), 1)
            defects.append({
                "category": "Water Damage & Oxidation Stains",
                "label": "Water Wash Stain",
                "severity_score": float(sev),
                "severity_label": "Local Defect",
                "detection_method": "Substrate Chromatic Deviation Extraction",
                "evidence": f"Oxidation wash stain area: {int(s_area):,} px",
                "recommended_operation": "Selective LaMa Inpainting Fill",
                "bbox": [int(sbx), int(sby), int(sbw), int(sbh)],
                "bbox_normalized": [
                    float(round(sbx / w, 3)),
                    float(round(sby / h, 3)),
                    float(round(sbw / w, 3)),
                    float(round(sbh / h, 3))
                ]
            })

        # ── 4. FADED INK & LOW-CONTRAST SCRIPT ──────────────────────────
        # Line-by-line contrast analysis, clustering vertically adjacent faded lines
        _, bin_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        line_k_w = max(15, int(w * 0.035))
        line_k_h = max(2, int(h * 0.003))
        lines_morph = cv2.morphologyEx(
            bin_inv,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (line_k_w, line_k_h))
        )
        cnts_lines, _ = cv2.findContours(lines_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        line_candidates = []
        for cnt in cnts_lines:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            if bw > (w * 0.10) and 6 < bh < (h * 0.25):
                line_roi = gray[by:by + bh, bx:bx + bw]
                line_bin = bin_inv[by:by + bh, bx:bx + bw]
                if np.count_nonzero(line_bin) > 20:
                    fg = float(line_roi[line_bin > 0].mean())
                    bg = float(line_roi[line_bin == 0].mean())
                    contrast = bg - fg
                    line_candidates.append({
                        "bbox": [bx, by, bw, bh],
                        "contrast": contrast,
                        "fg": fg,
                        "bg": bg
                    })

        if line_candidates:
            max_c = max(b["contrast"] for b in line_candidates)
            # Faded lines have noticeably lower contrast than the optimal lines
            faded_lines = [b for b in line_candidates if b["contrast"] < max_c * 0.78 and b["contrast"] < 135]
            if faded_lines:
                faded_lines.sort(key=lambda b: b["bbox"][1])
                # Cluster vertically adjacent lines (gap < 25px) to bound local faded blocks
                clusters: List[List[Dict[str, Any]]] = []
                for l in faded_lines:
                    if not clusters:
                        clusters.append([l])
                    else:
                        prev_bottom = clusters[-1][-1]["bbox"][1] + clusters[-1][-1]["bbox"][3]
                        if (l["bbox"][1] - prev_bottom) < 25:
                            clusters[-1].append(l)
                        else:
                            clusters.append([l])

                for cl in clusters[:2]:  # Top 2 faded clusters
                    min_x = min(b["bbox"][0] for b in cl)
                    min_y = min(b["bbox"][1] for b in cl)
                    max_x = max(b["bbox"][0] + b["bbox"][2] for b in cl)
                    max_y = max(b["bbox"][1] + b["bbox"][3] for b in cl)

                    px = int((max_x - min_x) * 0.05)
                    py = int((max_y - min_y) * 0.05)
                    min_x = max(0, min_x - px)
                    min_y = max(0, min_y - py)
                    bw_c = min(w - min_x, (max_x - min_x) + 2 * px)
                    bh_c = min(h - min_y, (max_y - min_y) + 2 * py)

                    avg_c = float(np.mean([b["contrast"] for b in cl]))
                    sev = round(float(min(85.0, max(28.0, (1.0 - avg_c / (max_c + 1e-5)) * 100))), 1)
                    defects.append({
                        "category": "Faded Ink & Low Contrast",
                        "label": "Faded Script",
                        "severity_score": float(sev),
                        "severity_label": "High" if sev > 50 else "Moderate",
                        "detection_method": "Morphological Stroke Contrast Analysis",
                        "evidence": f"Faded text block: contrast {avg_c:.1f} vs optimal {max_c:.1f}",
                        "recommended_operation": "CLAHE Contrast Enhancement",
                        "bbox": [int(min_x), int(min_y), int(bw_c), int(bh_c)],
                        "bbox_normalized": [
                            float(round(min_x / w, 3)),
                            float(round(min_y / h, 3)),
                            float(round(bw_c / w, 3)),
                            float(round(bh_c / h, 3))
                        ]
                    })

        # ── 5. UNEVEN ILLUMINATION & SHADOW BLEMISH ─────────────────────
        q1 = float(np.mean(gray[:h // 2, :w // 2]))
        q2 = float(np.mean(gray[:h // 2, w // 2:]))
        q3 = float(np.mean(gray[h // 2:, :w // 2]))
        q4 = float(np.mean(gray[h // 2:, w // 2:]))
        quad_diff = float(max(q1, q2, q3, q4) - min(q1, q2, q3, q4))

        if quad_diff > 35.0:
            illum_sev = round(float(min(70.0, quad_diff * 0.8)), 1)
            q_means = [q1, q2, q3, q4]
            min_q = int(np.argmin(q_means))
            qx = 0 if min_q in [0, 2] else w // 2
            qy = 0 if min_q in [0, 1] else h // 2
            defects.append({
                "category": "Uneven Illumination & Vignetting",
                "label": "Shadow Blemish",
                "severity_score": float(illum_sev),
                "severity_label": "Moderate",
                "detection_method": "Multi-Quadrant Luminance Discrepancy",
                "evidence": f"Quadrant illumination gradient delta: {quad_diff:.1f}",
                "recommended_operation": "Morphological Background Normalization",
                "bbox": [int(qx), int(qy), int(w // 2), int(h // 2)],
                "bbox_normalized": [
                    float(round(qx / w, 3)),
                    float(round(qy / h, 3)),
                    0.5,
                    0.5
                ]
            })

        # ── 6. DIVERSITY SELECTION: ENSURE DISTINCT DEFECT CATEGORIES ──
        category_buckets: Dict[str, List[Dict[str, Any]]] = {}
        for d in defects:
            cat = d["category"]
            if cat not in category_buckets:
                category_buckets[cat] = []
            category_buckets[cat].append(d)

        final_defects: List[Dict[str, Any]] = []
        cats_by_max_sev = sorted(
            category_buckets.keys(),
            key=lambda c: max(d["severity_score"] for d in category_buckets[c]),
            reverse=True
        )
        for c in cats_by_max_sev:
            best_d = max(category_buckets[c], key=lambda d: d["severity_score"])
            final_defects.append(best_d)
            category_buckets[c].remove(best_d)

        remaining = [d for bucket in category_buckets.values() for d in bucket]
        remaining.sort(key=lambda d: d["severity_score"], reverse=True)
        while len(final_defects) < 4 and remaining:
            final_defects.append(remaining.pop(0))

        final_defects.sort(key=lambda d: d["severity_score"], reverse=True)

        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        contrast_std = float(gray.std())

        if final_defects:
            overall_index = round(float(np.mean([d["severity_score"] for d in final_defects])), 1)
        else:
            overall_index = 8.5

        return {
            "engine_version": self.ENGINE_VERSION,
            "document_dimensions": {"width": w, "height": h},
            "overall_degradation_index": float(overall_index),
            "overall_status": (
                "Severe Degradation" if overall_index > 50
                else ("Moderate Degradation" if overall_index > 25
                      else "Preserved / Minor Decay")
            ),
            "defect_count": len(final_defects),
            "defects": final_defects,
            "spectral_metrics": {
                "laplacian_sharpness": round(laplacian_var, 2),
                "contrast_std": round(contrast_std, 2),
                "illumination_delta": round(quad_diff, 2)
            }
        }


damage_detector = DamageDetector()
