"""API 端点集成测试。"""

import hashlib
import json
import os
import tempfile
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from config import AppConfig
from main import app


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
    client.post("/login", json={
        "username": os.getenv("ADMIN_USERNAME", "admin"),
        "password": os.getenv("ADMIN_PASSWORD", "admin"),
        "totp_code": "",  # 无 TOTP_SECRET 时跳过
    })
    return client


def test_index_returns_html():
    """GET / 应返回 HTML 页面。"""
    client = TestClient(app)
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "JellySub-AI" in resp.text


def test_get_config():
    """GET /api/config 应返回当前配置。"""
    client = TestClient(app)
    _authenticated_client(client)
    resp = client.get("/api/config")
    assert resp.status_code == 200
    data = resp.json()
    assert data["jellyfin_url"] == "http://test:8096"
    assert data["target_language"] == "zh-CN"
    assert data["translate_model_local"] == "Qwen/Qwen3-0.6B"


def test_get_config_unauthenticated():
    """未认证时 GET /api/config 应返回 401。"""
    client = TestClient(app)
    resp = client.get("/api/config", follow_redirects=False)
    assert resp.status_code == 401


def test_put_config():
    """PUT /api/config 应保存新配置。"""
    client = TestClient(app)
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


def test_webhook_accepts_movie():
    """POST /webhook 接受 Movie 类型。"""
    client = TestClient(app)
    payload = {
        "Name": "Test Movie",
        "ItemId": "abc123",
        "Path": "/media/movies/test.mp4",
        "ItemType": "Movie",
    }
    resp = client.post("/webhook", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "accepted"
    assert data["item"] == "Test Movie"


def test_webhook_accepts_episode():
    """POST /webhook 接受 Episode 类型。"""
    client = TestClient(app)
    payload = {
        "Name": "Episode 1",
        "ItemId": "ep001",
        "Path": "/media/show/s01e01.mp4",
        "ItemType": "Episode",
    }
    resp = client.post("/webhook", json=payload)
    assert resp.status_code == 200
    assert resp.json()["status"] == "accepted"


def test_webhook_skips_non_video():
    """POST /webhook 跳过非视频类型。"""
    client = TestClient(app)
    for item_type in ("Music", "Book", "Photo", "Series"):
        payload = {"Name": "Test", "ItemId": "1", "Path": "/test", "ItemType": item_type}
        resp = client.post("/webhook", json=payload)
        assert resp.status_code == 200
        assert resp.json()["status"] == "skipped"


def test_webhook_missing_path():
    """POST /webhook 缺少 Path 应返回错误。"""
    client = TestClient(app)
    payload = {"Name": "Test", "ItemId": "1", "ItemType": "Movie"}
    resp = client.post("/webhook", json=payload)
    assert resp.status_code == 200
    assert resp.json()["status"] == "error"


def test_webhook_missing_item_id():
    """POST /webhook 缺少 ItemId 应返回错误。"""
    client = TestClient(app)
    payload = {"Name": "Test", "Path": "/test.mp4", "ItemType": "Movie"}
    resp = client.post("/webhook", json=payload)
    assert resp.status_code == 200
    assert resp.json()["status"] == "error"


def test_webhook_rejects_invalid_signature():
    """POST /webhook 带 WEBHOOK_SECRET 时应校验签名。"""
    with patch("main.WEBHOOK_SECRET", "test-secret"):
        client = TestClient(app)
        payload = {
            "Name": "Test Movie",
            "ItemId": "abc123",
            "Path": "/media/movies/test.mp4",
            "ItemType": "Movie",
        }
        # 无效签名
        resp = client.post("/webhook", json=payload, headers={"X-Jellyfin-Signature": "wrong"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "error"
        assert resp.json()["reason"] == "invalid signature"

        # 有效签名
        expected = hashlib.sha256("test-secret".encode()).hexdigest()
        resp = client.post("/webhook", json=payload, headers={"X-Jellyfin-Signature": expected})
        assert resp.status_code == 200
        assert resp.json()["status"] == "accepted"


def test_static_files_served():
    """静态文件应可通过 /static/ 访问。"""
    client = TestClient(app)
    resp = client.get("/static/style.css")
    assert resp.status_code == 200


def test_api_status():
    """GET /api/status 应返回脱敏状态信息。"""
    client = TestClient(app)
    resp = client.get("/api/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "asr_mode" in data
    assert "asr_model" in data
    assert "translate_mode" in data
    assert "target_language" in data


def test_login_redirect_when_authenticated():
    """已登录用户访问 /login 应重定向到 /admin。"""
    client = TestClient(app)
    _authenticated_client(client)
    resp = client.get("/login", follow_redirects=False)
    assert resp.status_code == 302
    assert resp.headers["location"] == "/admin"
