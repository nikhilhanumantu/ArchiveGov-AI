import pytest
from backend.app.core.security import hash_password, verify_password

def get_auth_headers(client, email="admin@archivegov.ai", password="Admin@12345"):
    resp = client.post("/api/v1/auth/login", json={
        "username_or_email": email,
        "password": password
    })
    assert resp.status_code == 200
    token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}

# ==================== 1. PHASE 0 SECURITY: PASSWORD HASHING ====================

def test_security_rejects_plaintext_and_invalid_hashes():
    """Verify strictly that plaintext passwords and non-pbkdf2 hashes are rejected."""
    # Plain text equality must return False
    assert verify_password("Admin@12345", "Admin@12345") is False
    assert verify_password("secret", "secret") is False
    # Non-PBKDF2 prefixes must return False
    assert verify_password("secret", "sha256$randomsalt$hash") is False
    assert verify_password("secret", "md5$hash") is False
    assert verify_password("", "") is False
    assert verify_password("secret", None) is False

    # Valid PBKDF2 hash must pass
    pwd = "ArchivalSecure@2026"
    hashed = hash_password(pwd)
    assert hashed.startswith("pbkdf2:sha256:")
    assert verify_password(pwd, hashed) is True
    assert verify_password("WrongPassword", hashed) is False

# ==================== 2. MODULE 3: RESTORATION TIME MACHINE ====================

def test_time_machine_stages(client):
    headers = get_auth_headers(client)
    doc_id = "DOC-1912-KDL"

    # Both /api/v1/versions/... and /api/versions/... must resolve
    for prefix in ["/api/v1/versions", "/api/versions"]:
        resp = client.get(f"{prefix}/{doc_id}/stages", headers=headers)
        assert resp.status_code == 200, f"Failed for prefix {prefix}: {resp.text}"
        data = resp.json()
        assert "stages" in data
        assert len(data["stages"]) >= 1
        first_stage = data["stages"][0]
        assert "stage_order" in first_stage
        assert "stage_name" in first_stage

def test_time_machine_diff_mask(client):
    headers = get_auth_headers(client)
    doc_id = "DOC-1912-KDL"

    for prefix in ["/api/v1/versions", "/api/versions"]:
        resp = client.get(f"{prefix}/{doc_id}/diff-mask?from_stage=original&to_stage=final", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert "mask_url" in data
        assert data.get("provenance") == "reconstructed_visual"

# ==================== 3. MODULE 4: DOCUMENT INTELLIGENCE LENS ====================

def test_lens_regions_extraction(client):
    headers = get_auth_headers(client)
    doc_id = "DOC-1912-KDL"

    for prefix in ["/api/v1/lens", "/api/lens"]:
        resp = client.get(f"{prefix}/{doc_id}/regions", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "regions" in data
        assert len(data["regions"]) > 0
        r = data["regions"][0]
        assert "bbox_normalized" in r
        assert "bbox_pixels" in r
        assert "confidence" in r
        assert "damage_overlap" in r

def test_lens_granularity_filtering(client):
    headers = get_auth_headers(client)
    doc_id = "DOC-1912-KDL"

    resp = client.get(f"/api/v1/lens/{doc_id}/regions?granularity=line", headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    for reg in data["regions"]:
        assert reg["granularity"] == "line"

def test_lens_explain_with_epistemic_provenance(client):
    headers = get_auth_headers(client)
    doc_id = "DOC-1912-KDL"

    for prefix in ["/api/v1/lens", "/api/lens"]:
        resp = client.post(
            f"{prefix}/{doc_id}/explain",
            headers=headers,
            json={
                "text": "Ryotwari settlement survey deed with assessment of Rs 42 and 8 annas for nanjai wet land",
                "query": "Explain fiscal and land context"
            }
        )
        assert resp.status_code == 200
        data = resp.json()
        # Verify strict epistemic labels
        assert "source_fact" in data
        assert "ocr_extraction" in data
        assert "ai_interpretation" in data
        assert "reconstructed_visual" in data
        assert data["provenance"] == "ai_interpretation"
        assert len(data["ai_interpretation"]) > 20

def test_lens_related_records(client):
    headers = get_auth_headers(client)
    doc_id = "DOC-1912-KDL"

    resp = client.get(f"/api/v1/lens/{doc_id}/related", headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert "related_records" in data

# ==================== 4. MODULE 1: HYBRID RAG COPILOT ====================

def test_copilot_query_and_session(client):
    headers = get_auth_headers(client)

    # Test /api/v1/copilot/query and /api/copilot/query
    for path in ["/api/v1/copilot/query", "/api/copilot/query"]:
        resp = client.post(
            path,
            headers=headers,
            json={
                "query": "What are the survey details and assessment rates in South Arcot in 1912?",
                "department": "all"
            }
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert "session_id" in data
        assert "answer" in data
        assert "citations" in data
        assert data.get("provenance") == "ai_interpretation"

        # Verify citation structure has lens_url
        if data["citations"]:
            assert "lens_url" in data["citations"][0]

        # Verify session retrieval
        session_id = data["session_id"]
        sess_resp = client.get(f"/api/v1/copilot/sessions/{session_id}", headers=headers)
        assert sess_resp.status_code == 200
        sess_data = sess_resp.json()
        assert len(sess_data.get("messages", [])) >= 2

def test_copilot_reindex(client):
    headers = get_auth_headers(client)

    resp = client.post("/api/v1/copilot/reindex", headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert "total_chunks" in data

# ==================== 5. MODULE 2: HISTORICAL TIMELINE ====================

def test_timeline_generation_and_filtering(client):
    headers = get_auth_headers(client)

    for prefix in ["/api/v1/timeline", "/api/timeline"]:
        # Generate
        gen_resp = client.post(f"{prefix}/generate", headers=headers, json={"timeline_id": "vault_master"})
        assert gen_resp.status_code == 200
        gen_data = gen_resp.json()
        assert gen_data["success"] is True
        assert gen_data["total_events"] > 0

        # Retrieve
        get_resp = client.get(f"{prefix}?certainty=verified", headers=headers)
        assert get_resp.status_code == 200
        get_data = get_resp.json()
        assert "events" in get_data
        for ev in get_data["events"]:
            assert ev["certainty"] == "verified"
            assert "source_fact" in ev
            assert "ai_interpretation" in ev
            assert "lens_url" in ev

# ==================== 6. HTML WEB UI ROUTES ====================

def test_lens_and_timeline_html_pages(client):
    headers = get_auth_headers(client)
    # Log in and set session cookie
    token = headers["Authorization"].split(" ")[1]
    client.cookies.set("archivegov_session", token)

    # Document Lens Page
    resp_lens = client.get("/lens/DOC-1912-KDL")
    assert resp_lens.status_code == 200
    assert "Document Intelligence Lens" in resp_lens.text
    assert "document_lens.js" in resp_lens.text

    # Historical Timeline Page
    resp_tl = client.get("/timeline")
    assert resp_tl.status_code == 200
    assert "Historical Archival Timeline" in resp_tl.text
    assert "timeline.js" in resp_tl.text

# ==================== 7. RBAC & UNAUTHENTICATED REJECTION ====================

def test_unauthenticated_api_rejection(client):
    client.cookies.clear()
    resp1 = client.get("/api/v1/lens/DOC-1912-KDL/regions")
    assert resp1.status_code == 401

    resp2 = client.get("/api/v1/timeline")
    assert resp2.status_code == 401

    resp3 = client.post("/api/v1/copilot/query", json={"query": "test"})
    assert resp3.status_code == 401
