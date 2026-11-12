import io
import time
import pytest
import pypdf
from PIL import Image
from backend.app.core.config import settings

def get_auth_headers(client, email="admin@archivegov.ai", password="Admin@12345"):
    resp = client.post("/api/v1/auth/login", json={
        "username_or_email": email,
        "password": password
    })
    assert resp.status_code == 200
    token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}

# ==================== 1. PUBLIC ROUTES ====================

def test_landing_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "ArchiveGov AI" in response.text

def test_login_page(client):
    response = client.get("/login")
    assert response.status_code == 200
    assert "Sign In" in response.text
    # Ensure notifications script is NOT loaded for unauthenticated users
    assert 'notifications.js' not in response.text

def test_login_page_with_stale_cookie(client):
    # Set a bogus/expired token cookie
    client.cookies.set(settings.COOKIE_NAME, "invalid.jwt.token")
    response = client.get("/login")
    assert response.status_code == 200
    assert "Sign In" in response.text

def test_login_page_authenticated_redirect(client):
    # Log in first
    login_resp = client.post("/api/v1/auth/login", json={
        "username_or_email": "admin@archivegov.ai",
        "password": "Admin@12345"
    })
    token = login_resp.json()["access_token"]
    # Access login page with valid session
    client.cookies.set(settings.COOKIE_NAME, token)
    response = client.get("/login", follow_redirects=False)
    assert response.status_code == 307 or response.status_code == 302
    assert response.headers["location"] == "/app/dashboard"

def test_auth_guard_no_circular_redirect(client):
    # Access protected dashboard without auth
    client.cookies.clear()
    response = client.get("/app/dashboard", follow_redirects=False)
    assert response.status_code == 307 or response.status_code == 302
    assert response.headers["location"] == "/login?redirect=/app/dashboard"

def test_forgot_password_page(client):
    response = client.get("/forgot-password")
    assert response.status_code == 200
    assert "Password" in response.text

# ==================== 2. AUTHENTICATION & SESSIONS ====================

def test_login_success(client):
    response = client.post("/api/v1/auth/login", json={
        "username_or_email": "admin@archivegov.ai",
        "password": "Admin@12345"
    })
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert data["user"]["email"] == "admin@archivegov.ai"
    assert data["user"]["role"] == "admin"

def test_login_invalid_password(client):
    response = client.post("/api/v1/auth/login", json={
        "username_or_email": "admin@archivegov.ai",
        "password": "WrongPassword123!"
    })
    assert response.status_code == 401

def test_get_me(client):
    headers = get_auth_headers(client)
    response = client.get("/api/v1/auth/me", headers=headers)
    assert response.status_code == 200
    user = response.json()
    assert user["email"] == "admin@archivegov.ai"
    assert "password_hash" not in user

# ==================== 3. DOCUMENT MANAGEMENT & UPLOAD ====================

def test_list_documents(client):
    headers = get_auth_headers(client)
    response = client.get("/api/v1/documents", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert "documents" in data
    assert isinstance(data["documents"], list)

def test_upload_document(client):
    headers = get_auth_headers(client)
    
    # Create dummy in-memory archival image
    img = Image.new("RGB", (200, 200), color=(240, 230, 210))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    
    files = {
        "file": ("test_record.png", buf, "image/png")
    }
    data = {
        "title": "Test Historical Survey Deed 1905",
        "category": "Historical Land Records",
        "department_id": "Land Revenue & Settlement",
        "original_date": "1905",
        "language": "English & Tamil Vernacular"
    }

    response = client.post("/api/v1/documents/upload", headers=headers, files=files, data=data)
    assert response.status_code in (200, 201)
    res_json = response.json()
    assert res_json["success"] is True
    doc = res_json["document"]
    assert doc["title"] == "Test Historical Survey Deed 1905"
    
    # Store doc_id on client for subsequent tests
    client.test_doc_id = doc["_id"]

def test_get_document_by_id(client):
    headers = get_auth_headers(client)
    doc_id = getattr(client, "test_doc_id", None)
    if not doc_id:
        pytest.skip("No uploaded doc_id available")

    response = client.get(f"/api/v1/documents/{doc_id}", headers=headers)
    assert response.status_code == 200
    assert response.json()["_id"] == doc_id

def test_upload_pdf_document_rasterization(client):
    headers = get_auth_headers(client)
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=200, height=200)
    pdf_buf = io.BytesIO()
    writer.write(pdf_buf)
    pdf_buf.seek(0)

    resp = client.post(
        "/api/v1/documents/upload",
        files={"file": ("proclamation_deed.pdf", pdf_buf, "application/pdf")},
        data={"title": "1912 Royal Proclamation PDF", "category": "Legal Deeds"},
        headers=headers
    )
    assert resp.status_code in (200, 201)
    doc = resp.json()["document"]
    assert doc["original_file_key"].endswith(".png")
    assert "pdf_file_key" in doc
    assert doc["pdf_file_key"].endswith(".pdf")
    doc_id = doc["_id"]

    # Verify damage analysis succeeds on the rasterized PDF
    analysis_resp = client.get(f"/api/v1/damage-analysis/{doc_id}", headers=headers)
    assert analysis_resp.status_code == 200
    analysis_data = analysis_resp.json()
    assert "overall_degradation_index" in analysis_data or "overall_status" in analysis_data

# ==================== 4. DAMAGE ANALYSIS & HEURISTICS ====================

def test_damage_analysis(client):
    headers = get_auth_headers(client)
    doc_id = getattr(client, "test_doc_id", None)
    if not doc_id:
        pytest.skip("No uploaded doc_id available")

    response = client.get(f"/api/v1/damage-analysis/{doc_id}", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert "overall_degradation_index" in data or "defects" in data or "overall_status" in data

# ==================== 5. OCR & TEXT EXTRACTION ====================

def test_ocr_extraction(client):
    headers = get_auth_headers(client)
    doc_id = getattr(client, "test_doc_id", None)
    if not doc_id:
        pytest.skip("No uploaded doc_id available")

    response = client.get(f"/api/v1/ocr/{doc_id}", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert "extracted_text" in data

def test_ocr_export_txt(client):
    headers = get_auth_headers(client)
    doc_id = getattr(client, "test_doc_id", None)
    if not doc_id:
        pytest.skip("No uploaded doc_id available")

    response = client.get(f"/api/v1/ocr/{doc_id}/export/txt", headers=headers)
    assert response.status_code == 200
    assert response.headers.get("content-type").startswith("text/plain")

# ==================== 6. SETTINGS & HARDWARE TELEMETRY ====================

def test_system_settings(client):
    headers = get_auth_headers(client)
    response = client.get("/api/v1/settings", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert "ai_hardware" in data
    assert "models_status" in data
    assert "database" in data

def test_settings_smtp_test(client):
    headers = get_auth_headers(client)
    response = client.post("/api/v1/settings/test-smtp", headers=headers)
    assert response.status_code == 200
    assert "message" in response.json()

def test_settings_audit_logs(client):
    headers = get_auth_headers(client)
    response = client.get("/api/v1/settings/audit-logs", headers=headers)
    assert response.status_code == 200
    assert "logs" in response.json()

# ==================== 7. RBAC & PERMISSION BOUNDARIES ====================

def test_admin_can_access_users(client):
    admin_headers = get_auth_headers(client, "admin@archivegov.ai", "Admin@12345")
    response = client.get("/api/v1/users", headers=admin_headers)
    assert response.status_code == 200
    users = response.json()
    assert isinstance(users, list)
    assert len(users) >= 1

def test_operator_forbidden_from_users(client):
    op_headers = get_auth_headers(client, "operator@archivegov.ai", "Operator@12345")
    response = client.get("/api/v1/users", headers=op_headers)
    assert response.status_code == 403

def test_reviewer_forbidden_from_upload_and_delete(client):
    reviewer_headers = get_auth_headers(client, "reviewer@archivegov.ai", "Reviewer@12345")
    # Reviewer cannot upload documents
    img = Image.new("RGB", (30, 30), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    upload_resp = client.post(
        "/api/v1/documents/upload",
        files={"file": ("reviewer_test.png", buf, "image/png")},
        data={"title": "Reviewer Upload Attempt", "category": "Test"},
        headers=reviewer_headers
    )
    assert upload_resp.status_code == 403

    # Reviewer cannot delete documents
    del_resp = client.delete("/api/v1/documents/DOC-1912-KDL", headers=reviewer_headers)
    assert del_resp.status_code == 403

def test_operator_forbidden_from_delete_and_approve(client):
    operator_headers = get_auth_headers(client, "operator@archivegov.ai", "Operator@12345")
    # Operator cannot delete documents
    del_resp = client.delete("/api/v1/documents/DOC-1912-KDL", headers=operator_headers)
    assert del_resp.status_code == 403

    # Operator cannot approve restoration review
    approve_resp = client.post(
        "/api/v1/reviews/DOC-1912-KDL/decision",
        json={"decision": "approve", "comments": "Operator unauthorized sign-off attempt"},
        headers=operator_headers
    )
    assert approve_resp.status_code == 403

def test_delete_document_permanently_from_db(client):
    admin_headers = get_auth_headers(client, "admin@archivegov.ai", "Admin@12345")
    # 1. Upload a temporary test document
    img = Image.new("RGB", (32, 32), color=(240, 230, 210))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    upload_resp = client.post(
        "/api/v1/documents/upload",
        files={"file": ("doc_to_delete.png", buf, "image/png")},
        data={"title": "Document Scheduled For Deletion", "category": "Test"},
        headers=admin_headers
    )
    assert upload_resp.status_code in (200, 201)
    doc_id = upload_resp.json()["document"]["document_id"]

    # Verify it exists in DB
    from backend.app.database.mongodb import get_db
    db = get_db()
    assert db.documents.find_one({"document_id": doc_id}) is not None

    # 2. Call DELETE endpoint
    del_resp = client.delete(f"/api/v1/documents/{doc_id}", headers=admin_headers)
    assert del_resp.status_code == 200
    del_data = del_resp.json()
    assert del_data.get("success") is True

    # 3. Verify it is permanently purged from MongoDB collection
    assert db.documents.find_one({"document_id": doc_id}) is None

    # 4. Verify GET returns 404
    get_resp = client.get(f"/api/v1/documents/{doc_id}", headers=admin_headers)
    assert get_resp.status_code == 404

# ==================== 8. NOTIFICATIONS ====================

def test_notifications_list(client):
    headers = get_auth_headers(client)
    response = client.get("/api/v1/notifications", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert "notifications" in data
    assert "unread_count" in data

def test_notifications_read_all(client):
    headers = get_auth_headers(client)
    response = client.post("/api/v1/notifications/read-all", headers=headers)
    assert response.status_code == 200

# ==================== 9. FINEGRAIN IMAGE ENHANCER ====================

def test_finegrain_status(client):
    headers = get_auth_headers(client)
    response = client.get("/api/v1/restoration/finegrain-status", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "space_id" in data
    assert data["space_id"] == "finegrain/finegrain-image-enhancer"

def test_finegrain_enhance_endpoint(client):
    headers = get_auth_headers(client)
    # Upload a small test image
    img = Image.new("RGB", (40, 40), color=(240, 235, 220))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    
    upload_resp = client.post(
        "/api/v1/documents/upload",
        files={"file": ("finegrain_test_doc.png", buf, "image/png")},
        data={"title": "Finegrain Test Document", "category": "Legal Deeds"},
        headers=headers
    )
    assert upload_resp.status_code in (200, 201)
    doc_id = upload_resp.json()["document"]["_id"]
    
    # Request Finegrain enhancement with the 13 parameters
    fg_resp = client.post(
        "/api/v1/restoration/finegrain-enhance",
        json={
            "document_id": doc_id,
            "prompt": "Test archival enhancement",
            "negative_prompt": "blurry, noise",
            "seed": 42,
            "upscale_factor": 1.5,
            "controlnet_scale": 0.6,
            "controlnet_decay": 1.0,
            "condition_scale": 6.0,
            "tile_width": 112,
            "tile_height": 144,
            "denoise_strength": 0.35,
            "num_inference_steps": 18,
            "solver": "DDIM"
        },
        headers=headers
    )
    assert fg_resp.status_code == 200
    data = fg_resp.json()
    assert data["success"] is True
    assert "output_file_key" in data
    assert "file_url" in data
    assert "metrics" in data

# ==================== 10. OPENROUTER GEMINI NANO BANANA 2.1 ====================

def test_openrouter_status(client):
    headers = get_auth_headers(client)
    response = client.get("/api/v1/restoration/openrouter-status", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Google Gemini Nano Banana 2.1"
    assert data["model_id"] == "google/gemini-nano-banana-2.1"
    assert "api_key_configured" in data

def test_openrouter_enhance_endpoint(client):
    headers = get_auth_headers(client)
    # Upload a small test image
    img = Image.new("RGB", (40, 40), color=(245, 240, 225))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    
    upload_resp = client.post(
        "/api/v1/documents/upload",
        files={"file": ("openrouter_test_doc.png", buf, "image/png")},
        data={"title": "OpenRouter Test Document", "category": "Legal Deeds"},
        headers=headers
    )
    assert upload_resp.status_code in (200, 201)
    doc_id = upload_resp.json()["document"]["_id"]

    resp = client.post(
        "/api/v1/restoration/openrouter-enhance",
        json={
            "document_id": doc_id,
            "prompt": "Restore archival document",
            "resolution": "1K",
            "aspect_ratio": "1:1"
        },
        headers=headers
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert "output_file_key" in data
    assert "file_url" in data
    assert "engine" in data

# ==================== 11. GOOGLE AI STUDIO ====================

def test_google_ai_studio_status(client):
    headers = get_auth_headers(client)
    response = client.get("/api/v1/restoration/google-ai-studio-status", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Google AI Studio"
    assert "connected" in data

def test_google_ai_studio_enhance_endpoint(client):
    headers = get_auth_headers(client)
    # Upload a small test image
    img = Image.new("RGB", (40, 40), color=(242, 238, 222))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    
    upload_resp = client.post(
        "/api/v1/documents/upload",
        files={"file": ("google_studio_test_doc.png", buf, "image/png")},
        data={"title": "Google Studio Test Document", "category": "Legal Deeds"},
        headers=headers
    )
    assert upload_resp.status_code in (200, 201)
    doc_id = upload_resp.json()["document"]["_id"]

    resp = client.post(
        "/api/v1/restoration/google-ai-studio-enhance",
        json={
            "document_id": doc_id,
            "prompt": "Restore master archival document substrate",
            "model": "gemini-nano-banana-2.1"
        },
        headers=headers
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert "output_file_key" in data
    assert "file_url" in data
    assert "engine" in data

# ==================== 12. RESTORATION ASYNC JOB ====================

def test_create_and_run_restoration_job(client):
    headers = get_auth_headers(client)
    img = Image.new("RGB", (60, 60), color=(240, 235, 220))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    upload_resp = client.post(
        "/api/v1/documents/upload",
        files={"file": ("job_test_doc.png", buf, "image/png")},
        data={"title": "Job Test Document", "category": "Historical Records"},
        headers=headers
    )
    assert upload_resp.status_code in (200, 201)
    doc_id = upload_resp.json()["document"]["_id"]

    job_create_resp = client.post(
        "/api/v1/restoration/jobs",
        json={
            "document_id": doc_id,
            "operations": ["damage_analysis", "opencv_preprocessing", "realesrgan_enhancement", "lama_inpainting", "quality_evaluation"],
            "model_configuration": {
                "enhancer": "openrouter"
            },
            "processing_parameters": {
                "prompt": "Restore test document"
            }
        },
        headers=headers
    )
    assert job_create_resp.status_code == 200
    job_id = job_create_resp.json()["job_id"]

    final_status = None
    final_progress = 0
    for _ in range(40):
        time.sleep(0.5)
        status_resp = client.get(f"/api/v1/restoration/jobs/{job_id}", headers=headers)
        assert status_resp.status_code == 200
        job_data = status_resp.json()
        final_status = job_data.get("status")
        final_progress = job_data.get("progress", 0)
        if final_status in ("completed", "failed"):
            break

    assert final_status == "completed"
    assert final_progress == 100

# ==================== 13. GFPGAN FACE RESTORATION & UPSCALING ====================

def test_gfpgan_status(client):
    headers = get_auth_headers(client)
    response = client.get("/api/v1/restoration/gfpgan-status", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "GFPGAN" in data["name"]
    assert "status" in data

def test_gfpgan_enhance_endpoint(client):
    headers = get_auth_headers(client)
    img = Image.new("RGB", (64, 64), color=(220, 210, 190))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    upload_resp = client.post(
        "/api/v1/documents/upload",
        files={"file": ("gfpgan_test_portrait.png", buf, "image/png")},
        data={"title": "GFPGAN Test Portrait", "category": "Legal Deeds"},
        headers=headers
    )
    assert upload_resp.status_code in (200, 201)
    doc_id = upload_resp.json()["document"]["_id"]

    resp = client.post(
        "/api/v1/restoration/gfpgan-enhance",
        json={
            "document_id": doc_id,
            "upscale": 2,
            "weight": 0.5,
            "only_center_face": False
        },
        headers=headers
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert "output_file_key" in data
    assert "file_url" in data
    assert "engine" in data
    assert "metrics" in data





