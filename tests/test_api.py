"""API 端点集成测试。"""

import hashlib
import hmac
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from config import AppConfig
from core.task_manager import TaskManager
from main import _credential_hash, app


@pytest.fixture(autouse=True)
def reset_config():
    """每个测试使用独立的临时配置，不修改真实 config.json。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = AppConfig(
            jellyfin_url="http://test:8096",
            jellyfin_api_key="test-key",
            asr_mode="local",
            asr_model="Qwen/Qwen3-ASR-0.6B",
            asr_api_url="",
            asr_api_key="",
            asr_model_online="",
            translate_mode="local",
            translate_api_url="https://api.test.com/v1",
            translate_api_key="api-key",
            translate_model="test-model",
            translate_model_local="Qwen/Qwen3-0.6B",
            translate_prompt_format="json",
            translate_thinking=False,
            path_mappings={"/media": "/mnt/data"},
            temp_dir=tmpdir,
        )
        with patch("main.get_config", return_value=cfg):
            with patch("config.get_config", return_value=cfg):
                with patch("main.save_config", side_effect=lambda c: None):
                    with patch("config.save_config", side_effect=lambda c: None):
                        yield cfg


def _authenticated_client(client: TestClient) -> TestClient:
    """给 TestClient 设置认证 session。"""
    username = os.getenv("ADMIN_USERNAME", "admin")
    password = os.getenv("ADMIN_PASSWORD", "admin")
    client.post("/login", json={
        "username": username,
        "password": _credential_hash(username, password),
        "totp_code": "",  # 无 TOTP_SECRET 时跳过
    })
    return client


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, base_url="https://testserver")


def test_login_accepts_browser_credential_hash(client):
    """Login accepts the hash produced by the browser login client."""
    response = client.post("/login", json={
        "username": "admin",
        "password": _credential_hash("admin", "admin"),
        "totp_code": "",
    })

    assert response.json() == {"status": "ok"}


def test_index_returns_html():
    """GET / 应返回 HTML 页面。"""
    client = TestClient(app, base_url="https://testserver")
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "JellySub-AI" in resp.text


def test_get_config():
    """GET /api/config 应返回当前配置。"""
    client = TestClient(app, base_url="https://testserver")
    _authenticated_client(client)
    resp = client.get("/api/config")
    assert resp.status_code == 200
    data = resp.json()
    assert data["jellyfin_url"] == "http://test:8096"
    assert data["target_language"] == "zh-CN"
    assert data["translate_model_local"] == "Qwen/Qwen3-0.6B"


def test_get_config_unauthenticated():
    """未认证时 GET /api/config 应返回 401。"""
    client = TestClient(app, base_url="https://testserver")
    resp = client.get("/api/config", follow_redirects=False)
    assert resp.status_code == 401


def test_put_config():
    """PUT /api/config 应保存新配置。"""
    client = TestClient(app, base_url="https://testserver")
    _authenticated_client(client)
    new_cfg = {
        "jellyfin_url": "http://new:9096",
        "jellyfin_api_key": "new-key",
        "asr_mode": "online",
        "asr_model": "custom/model",
        "asr_api_url": "https://api.openai.com/v1",
        "asr_api_key": "asr-key",
        "asr_model_online": "whisper-1",
        "translate_mode": "local",
        "translate_api_url": "https://new-api.com/v1",
        "translate_api_key": "new-api-key",
        "translate_model": "",
        "translate_model_local": "gpt-4o-mini",
        "translate_prompt_format": "numbered",
        "translate_thinking": False,
        "target_language": "zh-TW",
        "path_mappings": {"/old": "/new"},
        "temp_dir": "/tmp/test",
        "video_dirs": [],
    }
    resp = client.put("/api/config", json=new_cfg)
    assert resp.status_code == 200
    assert resp.json() == {"status": "saved"}


def _signed_webhook(client: TestClient, payload: dict, signature_body: bytes | None = None):
    raw_body = json.dumps(payload, separators=(",", ":")).encode()
    body_to_sign = signature_body if signature_body is not None else raw_body
    signature = hmac.new(b"test-secret", body_to_sign, hashlib.sha256).hexdigest()
    return client.post(
        "/webhook",
        content=raw_body,
        headers={"Content-Type": "application/json", "X-Jellyfin-Signature": signature},
    )


@pytest.fixture
def webhook_payload(reset_config, tmp_path, monkeypatch):
    video_root = tmp_path / "videos"
    video_root.mkdir()
    video_path = video_root / "test.mp4"
    video_path.touch()
    reset_config.video_dirs = [str(video_root)]
    reset_config.path_mappings = {"/media": str(video_root)}

    async def no_internal_subtitle(_path):
        return False

    monkeypatch.setattr("core.audio.has_internal_subtitle", no_internal_subtitle)
    monkeypatch.setattr("core.subtitle_checker.find_existing_subtitle", lambda *_args: None)
    monkeypatch.setattr("main.task_manager", TaskManager(str(tmp_path / "tasks.db")))
    return {
        "Name": "Test Movie",
        "ItemId": "abc123",
        "Path": "/media/test.mp4",
        "ItemType": "Movie",
    }


def test_webhook_rejects_when_secret_is_unset(client, webhook_payload):
    """A missing secret must not leave the webhook endpoint open."""
    with patch("main.WEBHOOK_SECRET", ""):
        assert client.post("/webhook", json=webhook_payload).status_code == 503


def test_webhook_accepts_only_body_hmac(client, webhook_payload):
    """A valid HMAC for the exact raw request body accepts a mapped video."""
    with patch("main.WEBHOOK_SECRET", "test-secret"):
        response = _signed_webhook(client, webhook_payload)
    assert response.status_code == 200
    assert response.json() == {"status": "accepted", "item": "Test Movie"}


def test_webhook_rejects_signature_for_changed_body(client, webhook_payload):
    """Changing a signed request body invalidates its signature."""
    original = json.dumps(webhook_payload, separators=(",", ":")).encode()
    changed_payload = {**webhook_payload, "Name": "Tampered Movie"}
    with patch("main.WEBHOOK_SECRET", "test-secret"):
        response = _signed_webhook(client, changed_payload, signature_body=original)
    assert response.status_code == 401


def test_webhook_rejects_invalid_signature_before_parsing_malformed_json(client):
    """Unauthenticated malformed bodies receive the authentication response."""
    with patch("main.WEBHOOK_SECRET", "test-secret"):
        response = client.post(
            "/webhook",
            content=b"not json",
            headers={"Content-Type": "application/json", "X-Jellyfin-Signature": "wrong"},
        )
    assert response.status_code == 401


def test_webhook_rejects_mapped_path_outside_video_roots(client, webhook_payload):
    """Mapped paths outside configured video roots never reach media probing."""
    webhook_payload["Path"] = "/media/../outside.mp4"
    with patch("main.WEBHOOK_SECRET", "test-secret"):
        response = _signed_webhook(client, webhook_payload)
    assert response.status_code == 403


def test_webhook_rejects_nonexistent_mapped_file(client, webhook_payload):
    """A mapped video must exist before a task can be created."""
    webhook_payload["Path"] = "/media/missing.mp4"
    with patch("main.WEBHOOK_SECRET", "test-secret"):
        response = _signed_webhook(client, webhook_payload)
    assert response.status_code == 404


def test_webhook_skips_duplicate_active_task(client, webhook_payload):
    """A pending task for the mapped video prevents a duplicate webhook task."""
    from main import task_manager

    mapped_path = str(Path(task_manager.db_path).parent / "videos" / "test.mp4")
    task_manager.create_task(mapped_path)
    with patch("main.WEBHOOK_SECRET", "test-secret"):
        response = _signed_webhook(client, webhook_payload)
    assert response.status_code == 200
    assert response.json()["status"] == "already_running"


def test_static_files_served():
    """静态文件应可通过 /static/ 访问。"""
    client = TestClient(app, base_url="https://testserver")
    resp = client.get("/static/style.css")
    assert resp.status_code == 200



def test_login_redirect_when_authenticated():
    """已登录用户访问 /login 应重定向到 /admin。"""
    client = TestClient(app, base_url="https://testserver")
    _authenticated_client(client)
    resp = client.get("/login", follow_redirects=False)
    assert resp.status_code == 302
    assert resp.headers["location"] == "/admin"
