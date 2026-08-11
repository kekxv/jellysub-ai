"""ASR 时间戳诊断统计测试。

诊断用于排查"识别不全/时间轴错位"：ASR 引擎未返回时间戳时
（sensevoice.none / qwen3.none），字幕时间轴会退回估算。
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from config import AppConfig
from main import _credential_hash, app


@pytest.fixture(autouse=True)
def reset_config():
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
            path_mappings={},
            temp_dir=tmpdir,
        )
        with patch("main.get_config", return_value=cfg):
            with patch("config.get_config", return_value=cfg):
                yield cfg


@pytest.fixture(autouse=True)
def reset_stats():
    from core.asr import reset_asr_diagnostics
    reset_asr_diagnostics()
    yield


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, base_url="https://testserver")


def _authenticated_client(client: TestClient) -> TestClient:
    username = os.getenv("ADMIN_USERNAME", "admin")
    password = os.getenv("ADMIN_PASSWORD", "admin")
    client.post("/login", json={
        "username": username,
        "password": _credential_hash(username, password),
        "totp_code": "",
    })
    return client


# ------------------------------------------------------------------ #
#  引擎级统计
# ------------------------------------------------------------------ #

def test_sensevoice_stats_reset_returns_zeros():
    from core.asr.sensevoice import get_sensevoice_stats
    stats = get_sensevoice_stats()
    assert stats == {"calls": 0, "word": 0, "sentence": 0, "none": 0}


def test_qwen3_stats_reset_returns_zeros():
    from core.asr.qwen3 import get_qwen3_stats
    stats = get_qwen3_stats()
    assert stats == {"calls": 0, "with_ts": 0, "none": 0}


def test_sensevoice_no_timestamp_fallback_increments_none():
    """SenseVoice 未返回任何时间戳时：单段 start==end==0，none 计数 +1。"""
    from core.asr.sensevoice import SenseVoiceAsrEngine, get_sensevoice_stats

    engine = SenseVoiceAsrEngine()
    engine._pipeline = MagicMock()  # 跳过真实模型加载
    engine._pipeline.generate.return_value = [
        {"text": "<|zh|>你好，世界。", "language": "zh"}
    ]

    with patch("core.audio.get_audio_duration", return_value=5.0):
        segments, lang = engine.transcribe("/tmp/fake_chunk.wav")

    assert len(segments) == 1
    assert segments[0]["start"] == segments[0]["end"] == 0.0
    stats = get_sensevoice_stats()
    assert stats["calls"] == 1
    assert stats["none"] == 1
    assert stats["word"] == 0
    assert stats["sentence"] == 0


def test_sensevoice_word_timestamps_increment_word():
    """SenseVoice 返回词级时间戳时：word 计数 +1。"""
    from core.asr.sensevoice import SenseVoiceAsrEngine, get_sensevoice_stats

    engine = SenseVoiceAsrEngine()
    engine._pipeline = MagicMock()
    engine._pipeline.generate.return_value = [{
        "text": "<|en|>Hello world.",
        "language": "en",
        "timestamp": [[0, 300], [300, 700], [700, 1100]],
        "words": ["Hello", " world", "."],
    }]

    with patch("core.audio.get_audio_duration", return_value=5.0):
        segments, lang = engine.transcribe("/tmp/fake_chunk.wav")

    assert len(segments) >= 1
    stats = get_sensevoice_stats()
    assert stats["calls"] == 1
    assert stats["word"] == 1
    assert stats["none"] == 0


def test_qwen3_no_timestamp_fallback_increments_none():
    """Qwen3-ASR 未返回时间戳时：none 计数 +1。"""
    from core.asr.qwen3 import Qwen3AsrEngine, get_qwen3_stats

    fake_result = MagicMock()
    fake_result.language = "en"
    fake_result.text = "Hello world."
    fake_result.time_stamps = None

    engine = Qwen3AsrEngine()
    engine._model = MagicMock()
    engine._model.transcribe.return_value = [fake_result]

    with patch("core.audio.get_audio_duration", return_value=5.0):
        segments, lang = engine.transcribe("/tmp/fake_chunk.wav")

    assert len(segments) == 1
    assert segments[0]["start"] == segments[0]["end"] == 0.0
    stats = get_qwen3_stats()
    assert stats["calls"] == 1
    assert stats["none"] == 1
    assert stats["with_ts"] == 0


# ------------------------------------------------------------------ #
#  汇总入口与 API
# ------------------------------------------------------------------ #

def test_get_asr_diagnostics_aggregates_both_engines():
    from core.asr import get_asr_diagnostics
    diag = get_asr_diagnostics()
    assert set(diag.keys()) == {"sensevoice", "qwen3"}
    assert diag["sensevoice"]["calls"] == 0
    assert diag["qwen3"]["calls"] == 0


def test_diagnostics_endpoint_requires_auth(client):
    response = client.get("/api/asr/diagnostics")
    assert response.status_code == 401


def test_diagnostics_endpoint_returns_stats(client):
    authed = _authenticated_client(client)
    response = authed.get("/api/asr/diagnostics")
    assert response.status_code == 200
    data = response.json()
    assert set(data.keys()) == {"sensevoice", "qwen3"}
    assert data["sensevoice"]["calls"] == 0
    assert data["qwen3"]["calls"] == 0
