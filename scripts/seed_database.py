import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

from backend.app.core.config import settings, UPLOADS_DIR
from backend.app.core.security import hash_password
from backend.app.database.mongodb import get_db, sync_disk_if_mock

def generate_sample_parchment_image(output_path: str, is_restored: bool = False, text_title: str = "LAND SETTLEMENT RECORD 1912"):
    """
    Synthesizes authentic-looking historical government document textures on disk.
    If degraded: aged sepia, darker water stains, foxing spots, frayed borders, faded ink.
    If restored: crisp high-contrast parchment, sharpened black/dark blue ink, cleaned stains.
    """
    w, h = 1000, 1300
    if is_restored:
        # Clean cream/ivory parchment
        base_color = np.full((h, w, 3), [248, 246, 238], dtype=np.uint8)
        # Subtle texture noise
        noise = np.random.normal(0, 3, (h, w, 3)).astype(np.int16)
        img = np.clip(base_color.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    else:
        # Degraded aged yellow/amber sepia parchment
        base_color = np.full((h, w, 3), [168, 198, 218], dtype=np.uint8) # BGR sepia tint
        noise = np.random.normal(0, 12, (h, w, 3)).astype(np.int16)
        img = np.clip(base_color.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Add water stain gradient circles
        cv2.circle(img, (w // 2, h // 2 - 100), 220, (110, 140, 175), -1)
        cv2.circle(img, (w - 180, 260), 140, (100, 130, 165), -1)
        # Blur the stains for realistic bleeding
        img = cv2.GaussianBlur(img, (51, 51), 0)
        
        # Add vignette on edges
        y, x = np.ogrid[:h, :w]
        center_x, center_y = w / 2, h / 2
        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        vignette = 1 - np.clip((dist - 350) / 450, 0, 0.45)
        for c in range(3):
            img[:, :, c] = (img[:, :, c] * vignette).astype(np.uint8)

    # Convert to PIL to draw text
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    
    ink_color = (25, 25, 35) if is_restored else (95, 75, 60)
    
    # Draw ornate border
    border_inset = 40
    border_color = (60, 50, 40) if is_restored else (120, 100, 80)
    draw.rectangle([border_inset, border_inset, w - border_inset, h - border_inset], outline=border_color, width=3)
    draw.rectangle([border_inset + 8, border_inset + 8, w - border_inset - 8, h - border_inset - 8], outline=border_color, width=1)
    
    # Official Header
    draw.text((w // 2 - 240, 70), "GOVERNMENT OF BRITISH INDIA", fill=ink_color)
    draw.text((w // 2 - 280, 110), "MADRAS PRESIDENCY — LAND REVENUE DEPARTMENT", fill=ink_color)
    draw.text((w // 2 - 210, 150), text_title, fill=ink_color)
    draw.line([(border_inset + 30, 185), (w - border_inset - 30, 185)], fill=border_color, width=2)
    
    # Official seal circle
    seal_color = (30, 60, 130) if is_restored else (80, 95, 130)
    draw.ellipse([w - 240, h - 260, w - 80, h - 100], outline=seal_color, width=3)
    draw.text((w - 210, h - 190), "OFFICIAL SEAL\nCOLLECTORATE", fill=seal_color)
    
    # Document content lines
    sample_text = [
        "Registry Reference No. 1912/KDL-Folio-IX",
        "Village: Kuddalur • Taluk: Vridhachalam • District: South Arcot",
        "Settlement Holder: Thiru M. Annasamy Mudaliar & Hereditary Co-sharers",
        "Total Wet Land Extent: 18 Acres, 3 Rods, 14 Perches",
        "Soil Classification: First Class Alluvial Clay (Kari Man)",
        "Annual Permanent Assessment: Rs. 64, Annas 8, Pies 0",
        "Bounded on the North by River Vellar embankment conduit;",
        "Bounded on the East by Revenue Cart-track to Virudhachalam town;",
        "Bounded on the South by Village Common Pasture lands No. 84;",
        "Bounded on the West by Ryotwari Holding of Govinda Pillai.",
        "",
        "The said tenure is hereby ratified under Crown Settlement Sanad.",
        "Entered and compared by Head Sheristadar, Collectorate Office.",
        "Given under my hand and the seal of the Collectorate at Cuddalore,",
        "this Twenty-Fourth day of October in the Year Nineteen Hundred and Twelve.",
    ]
    
    y_offset = 220
    for line in sample_text:
        draw.text((border_inset + 45, y_offset), line, fill=ink_color)
        y_offset += 45
        
    draw.text((border_inset + 45, h - 140), "Transcribed & Verified via ArchiveGov AI Sovereign Ingestion", fill=(120, 120, 140))
    
    # Save image
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    pil_img.save(output_path, "PNG")

def seed_all(db=None):
    if db is None:
        db = get_db()
        
    # Check if already seeded
    if db.users.count_documents({}) > 0:
        return

    print("Seeding ArchiveGov AI database with official roles and historical records...")
    now = datetime.now(timezone.utc).isoformat()
    
    # 1. Users
    users = [
        {
            "name": "Dr. Rajeshwar Sharma",
            "email": "admin@archivegov.ai",
            "employee_id": "GOV-ADM-001",
            "password_hash": hash_password("Admin@12345"),
            "role": "admin",
            "department_id": "Central Government National Archives",
            "account_status": "active",
            "created_at": now
        },
        {
            "name": "Senior Officer Meenakshi Iyer",
            "email": "officer@archivegov.ai",
            "employee_id": "GOV-OFF-042",
            "password_hash": hash_password("Officer@12345"),
            "role": "archive_officer",
            "department_id": "Land Revenue & Settlement Archives",
            "account_status": "active",
            "created_at": now
        },
        {
            "name": "Chief Reviewer K. Ramanathan",
            "email": "reviewer@archivegov.ai",
            "employee_id": "GOV-REV-019",
            "password_hash": hash_password("Reviewer@12345"),
            "role": "reviewer",
            "department_id": "Historical Court & Legal Records",
            "account_status": "active",
            "created_at": now
        },
        {
            "name": "Operator Vikram Singh",
            "email": "operator@archivegov.ai",
            "employee_id": "GOV-DAT-108",
            "password_hash": hash_password("Operator@12345"),
            "role": "data_operator",
            "department_id": "Digital Preservation Ingestion Unit",
            "account_status": "active",
            "created_at": now
        }
    ]
    user_ids = {}
    for u in users:
        res = db.users.insert_one(u)
        user_ids[u["role"]] = str(res.inserted_id)

    # 2. Generate Real Sample Image Files
    orig_path_1 = settings.ORIGINALS_DIR + "/doc_1912_kdl_orig.png"
    rest_path_1 = settings.RESTORED_DIR + "/doc_1912_kdl_rest.png"
    generate_sample_parchment_image(orig_path_1, is_restored=False, text_title="LAND SETTLEMENT RECORD 1912 — KUDDALUR")
    generate_sample_parchment_image(rest_path_1, is_restored=True, text_title="LAND SETTLEMENT RECORD 1912 — KUDDALUR")

    orig_path_2 = settings.ORIGINALS_DIR + "/doc_1888_bng_orig.png"
    rest_path_2 = settings.RESTORED_DIR + "/doc_1888_bng_rest.png"
    generate_sample_parchment_image(orig_path_2, is_restored=False, text_title="BENGAL TENANCY REGISTRY FOLIO 1888")
    generate_sample_parchment_image(rest_path_2, is_restored=True, text_title="BENGAL TENANCY REGISTRY FOLIO 1888")

    orig_path_3 = settings.ORIGINALS_DIR + "/doc_1934_bom_orig.png"
    rest_path_3 = settings.RESTORED_DIR + "/doc_1934_bom_rest.png"
    generate_sample_parchment_image(orig_path_3, is_restored=False, text_title="BOMBAY HIGH COURT DECREE 1934")
    generate_sample_parchment_image(rest_path_3, is_restored=True, text_title="BOMBAY HIGH COURT DECREE 1934")

    # 3. Documents
    docs = [
        {
            "document_id": "DOC-1912-KDL",
            "title": "1912 Land Settlement Deed — Kuddalur Revenue District",
            "department_id": "Land Revenue & Settlement",
            "category": "Historical Land Records",
            "original_date": "1912",
            "language": "English & Tamil Vernacular",
            "archive_reference_id": "REC-1912-KDL-V09",
            "description": "Colonial land registry deed confirming perpetual ryotwari tenancy in South Arcot collectorate with official royal revenue seal.",
            "original_file_key": "doc_1912_kdl_orig.png",
            "restored_file_key": "doc_1912_kdl_rest.png",
            "file_size": 1824000,
            "checksum": "a7b8c9d0e1f21912kdl456789abcdef",
            "current_version_id": "VER-V20-KDL",
            "status": "pending_review",
            "uploaded_by": user_ids["archive_officer"],
            "uploader_name": "Senior Officer Meenakshi Iyer",
            "damage_analysis": {
                "overall_degradation_index": 38.4,
                "overall_status": "Moderate Degradation",
                "defect_count": 3,
                "spectral_metrics": {"laplacian_sharpness": 142.5, "contrast_std": 38.2, "illumination_delta": 42.1},
                "defects": [
                    {
                        "category": "Water Damage & Oxidation Stains",
                        "severity_score": 45.2,
                        "severity_label": "Local Defect",
                        "detection_method": "CIELAB Chromaticity Anomaly Clustering",
                        "evidence": "Central water wash stain area: 24,500 px",
                        "recommended_operation": "Selective LaMa Inpainting Mask Fill",
                        "bbox": [280, 320, 440, 360],
                        "bbox_normalized": [0.28, 0.24, 0.44, 0.27]
                    },
                    {
                        "category": "Faded Ink & Low Contrast",
                        "severity_score": 38.0,
                        "severity_label": "Moderate",
                        "detection_method": "RMS Contrast Distribution",
                        "evidence": "Iron gall ink degradation standard deviation: 38.2",
                        "recommended_operation": "CLAHE Contrast Enhancement",
                        "bbox": [100, 180, 800, 600],
                        "bbox_normalized": [0.1, 0.13, 0.8, 0.46]
                    },
                    {
                        "category": "Torn Paper & Border Voids",
                        "severity_score": 32.0,
                        "severity_label": "Structural Damage",
                        "detection_method": "Contour Convexity Defect Analysis",
                        "evidence": "Torn left jagged margin perimeter",
                        "recommended_operation": "LaMa Generative Structural Boundary Repair",
                        "bbox": [0, 0, 120, 1300],
                        "bbox_normalized": [0.0, 0.0, 0.12, 1.0]
                    }
                ]
            },
            "quality_metrics": {
                "non_reference_indicators": {
                    "sharpness_score": 324.5,
                    "rms_contrast": 58.4,
                    "mean_brightness": 242.0,
                    "information_entropy": 7.42,
                    "overall_restoration_fidelity_pct": 98.7
                },
                "reference_based_metrics": {
                    "has_reference": True,
                    "psnr_db": 29.1,
                    "ssim": 0.963
                }
            },
            "created_at": now
        },
        {
            "document_id": "DOC-1888-BNG",
            "title": "1888 Bengal Tenancy Registry Certificate",
            "department_id": "Agricultural & Tenancy Board",
            "category": "Revenue Records",
            "original_date": "1888",
            "language": "English & Bengali",
            "archive_reference_id": "REC-1888-BNG-V02",
            "description": "Historical tenancy record under Bengal Act VIII of 1885 establishing occupancy raiyat rights.",
            "original_file_key": "doc_1888_bng_orig.png",
            "restored_file_key": "doc_1888_bng_rest.png",
            "file_size": 1940000,
            "checksum": "b8c9d0e1f2a31888bng456789abcdef",
            "current_version_id": "VER-V20-BNG",
            "status": "restored",
            "uploaded_by": user_ids["archive_officer"],
            "uploader_name": "Senior Officer Meenakshi Iyer",
            "created_at": now
        },
        {
            "document_id": "DOC-1934-BOM",
            "title": "1934 Bombay High Court Archival Decree",
            "department_id": "Judicial Archives",
            "category": "Historical Court Records",
            "original_date": "1934",
            "language": "English & Marathi",
            "archive_reference_id": "REC-1934-BOM-J12",
            "description": "Appellate decree regarding sovereign municipal property easement in Fort Bombay.",
            "original_file_key": "doc_1934_bom_orig.png",
            "restored_file_key": "doc_1934_bom_rest.png",
            "file_size": 2100000,
            "checksum": "c9d0e1f2a3b41934bom456789abcdef",
            "current_version_id": "VER-V20-BOM",
            "status": "approved",
            "uploaded_by": user_ids["archive_officer"],
            "uploader_name": "Senior Officer Meenakshi Iyer",
            "assigned_reviewer": user_ids["reviewer"],
            "created_at": now
        }
    ]
    for d in docs:
        db.documents.insert_one(d)

    # 4. Versions for DOC-1912-KDL
    versions = [
        {
            "version_id": "VER-V10-KDL",
            "document_id": "DOC-1912-KDL",
            "version_number": "v1.0",
            "parent_version_id": None,
            "version_type": "original",
            "file_key": "doc_1912_kdl_orig.png",
            "model_used": "Raw Archival Scan (Unmodified Ingestion)",
            "parameters": {"resolution_dpi": 600, "color_space": "RGB"},
            "created_by": user_ids["archive_officer"],
            "created_at": now,
            "notes": "Immutable original master archival scan."
        },
        {
            "version_id": "VER-V11-KDL",
            "document_id": "DOC-1912-KDL",
            "version_number": "v1.1",
            "parent_version_id": "VER-V10-KDL",
            "version_type": "preprocessing",
            "file_key": "doc_1912_kdl_orig.png",
            "model_used": "OpenCV Archival Pipeline (CLAHE + Bilateral Denoise + Deskew)",
            "parameters": {"deskew_deg": -1.2, "clahe_clip": 2.5},
            "created_by": user_ids["archive_officer"],
            "created_at": now,
            "notes": "Deskewed by -1.2° and equalized contrast."
        },
        {
            "version_id": "VER-V20-KDL",
            "document_id": "DOC-1912-KDL",
            "version_number": "v2.0",
            "parent_version_id": "VER-V11-KDL",
            "version_type": "restored",
            "file_key": "doc_1912_kdl_rest.png",
            "model_used": "Real-ESRGAN Super-Resolution + LaMa Neural Inpainting + PaddleOCR",
            "parameters": {"psnr_db": 29.1, "ssim": 0.963, "fidelity_pct": 98.7},
            "created_by": user_ids["archive_officer"],
            "created_at": now,
            "notes": "Current Restored Master. Reconstructed margins and ink enhanced."
        }
    ]
    for v in versions:
        db.document_versions.insert_one(v)

    # 5. OCR Results for DOC-1912-KDL
    db.ocr_results.insert_one({
        "document_id": "DOC-1912-KDL",
        "version_id": "v2.0",
        "extracted_text": "GOVERNMENT OF BRITISH INDIA\nMADRAS PRESIDENCY — LAND REVENUE DEPARTMENT\nLAND SETTLEMENT RECORD 1912 — KUDDALUR\nRegistry Reference No. 1912/KDL-Folio-IX\nVillage: Kuddalur • Taluk: Vridhachalam • District: South Arcot\nSettlement Holder: Thiru M. Annasamy Mudaliar & Hereditary Co-sharers\nTotal Wet Land Extent: 18 Acres, 3 Rods, 14 Perches\nSoil Classification: First Class Alluvial Clay (Kari Man)\nAnnual Permanent Assessment: Rs. 64, Annas 8, Pies 0\nBounded on the North by River Vellar embankment conduit;\nBounded on the East by Revenue Cart-track to Virudhachalam town;\nBounded on the South by Village Common Pasture lands No. 84;\nBounded on the West by Ryotwari Holding of Govinda Pillai.\nThe said tenure is hereby ratified under Crown Settlement Sanad.\nEntered and compared by Head Sheristadar, Collectorate Office.\nGiven under my hand and the seal of the Collectorate at Cuddalore,\nthis Twenty-Fourth day of October in the Year Nineteen Hundred and Twelve.",
        "corrected_text": "GOVERNMENT OF BRITISH INDIA\nMADRAS PRESIDENCY — LAND REVENUE DEPARTMENT\nLAND SETTLEMENT RECORD 1912 — KUDDALUR\nRegistry Reference No. 1912/KDL-Folio-IX\nVillage: Kuddalur • Taluk: Vridhachalam • District: South Arcot\nSettlement Holder: Thiru M. Annasamy Mudaliar & Hereditary Co-sharers\nTotal Wet Land Extent: 18 Acres, 3 Rods, 14 Perches\nSoil Classification: First Class Alluvial Clay (Kari Man)\nAnnual Permanent Assessment: Rs. 64, Annas 8, Pies 0\nBounded on the North by River Vellar embankment conduit;\nBounded on the East by Revenue Cart-track to Virudhachalam town;\nBounded on the South by Village Common Pasture lands No. 84;\nBounded on the West by Ryotwari Holding of Govinda Pillai.\nThe said tenure is hereby ratified under Crown Settlement Sanad.\nEntered and compared by Head Sheristadar, Collectorate Office.\nGiven under my hand and the seal of the Collectorate at Cuddalore,\nthis Twenty-Fourth day of October in the Year Nineteen Hundred and Twelve.",
        "language": "en",
        "average_confidence": 98.4,
        "model_used": "PaddleOCR v2.8 Multilingual Dual-stream",
        "created_at": now
    })

    # 6. Notifications
    db.notifications.insert_many([
        {
            "user_id": "all",
            "type": "system",
            "title": "ArchiveGov AI Engine Online",
            "message": "All neural restoration clusters operating at peak fidelity under SEC-L4 protocol.",
            "is_read": False,
            "created_at": now
        },
        {
            "user_id": user_ids["archive_officer"],
            "type": "review_pending",
            "title": "Review Required: DOC-1912-KDL",
            "message": "1912 Land Settlement Deed awaiting archival sign-off.",
            "related_document_id": "DOC-1912-KDL",
            "is_read": False,
            "created_at": now
        },
        {
            "user_id": user_ids["archive_officer"],
            "type": "restoration_completed",
            "title": "Restoration Completed",
            "message": "1888 Bengal Tenancy Registry Certificate restored with 99.1% clarity.",
            "related_document_id": "DOC-1888-BNG",
            "is_read": True,
            "created_at": now
        }
    ])

    # 7. Audit Logs
    db.audit_logs.insert_many([
        {
            "user_id": user_ids["admin"],
            "action": "SYSTEM_INITIALIZED",
            "resource_type": "system",
            "resource_id": "core",
            "metadata": {"version": settings.APP_VERSION, "status": "SEC-L4 Active"},
            "timestamp": now
        },
        {
            "user_id": user_ids["archive_officer"],
            "action": "DOCUMENT_UPLOAD",
            "resource_type": "document",
            "resource_id": "DOC-1912-KDL",
            "metadata": {"title": "1912 Land Settlement Deed"},
            "timestamp": now
        }
    ])

    sync_disk_if_mock()
    print("Database seeding completed successfully.")

if __name__ == "__main__":
    seed_all()
