"""端到端字幕生成测试 — 验证完整字幕流程。"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from config import AppConfig
from core.subtitle_writer import generate_srt, generate_bilingual_srt

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
ASSETS_DIR = os.path.join(PROJECT_ROOT, "assets")
TEST_MP4 = os.path.join(ASSETS_DIR, "en.mp4")


@pytest.fixture
def mock_segments():
    """模拟 ASR 识别结果。"""
    return [
        {"start": 0.5, "end": 2.0, "text": "Hello world"},
        {"start": 2.5, "end": 4.0, "text": "This is a test"},
        {"start": 4.5, "end": 6.0, "text": "Goodbye"},
    ]


@pytest.fixture
def mock_translated():
    """模拟翻译结果。"""
    return [
        {"start": 0.5, "end": 2.0, "text": "你好世界"},
        {"start": 2.5, "end": 4.0, "text": "这是一个测试"},
        {"start": 4.5, "end": 6.0, "text": "再见"},
    ]


def test_generate_srt(mock_segments):
    """测试 SRT 文件生成。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        out = os.path.join(tmpdir, "test.srt")
        result = generate_srt(mock_segments, out)
        assert result is True
        assert os.path.exists(out)
        with open(out, "r", encoding="utf-8") as f:
            content = f.read()
        assert "00:00:" in content
        assert "Hello world" in content
        assert "This is a test" in content


def test_generate_bilingual_srt(mock_segments, mock_translated):
    """测试双语 SRT 文件生成。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        out = os.path.join(tmpdir, "bilingual.srt")
        result = generate_bilingual_srt(mock_segments, mock_translated, out)
        assert result is True
        assert os.path.exists(out)
        with open(out, "r", encoding="utf-8") as f:
            content = f.read()
        assert "你好世界" in content
        assert "Hello world" in content


def test_generate_srt_empty():
    """测试空片段生成 SRT。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        out = os.path.join(tmpdir, "empty.srt")
        result = generate_srt([], out)
        assert result is True
        assert os.path.exists(out)
        with open(out, "r", encoding="utf-8") as f:
            assert f.read().strip() == ""


# ---- Real pipeline test (run with: pytest -m integration) ----

@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_pipeline_from_mp4():
    """
    从 assets/en.mp4 真实生成字幕。
    仅当使用 -m integration 标记时才运行，避免拖慢常规测试。
    """
    if not os.path.exists(TEST_MP4):
        pytest.skip(f"测试文件不存在: {TEST_MP4}")

    from core.audio import extract_audio
    from core.asr import run_asr
    from core.translate import translate_segments

    cfg_path = os.path.join(PROJECT_ROOT, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = AppConfig(**json.load(f))
    else:
        cfg = AppConfig()

    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Extract audio
        audio_path = os.path.join(tmpdir, "audio.wav")
        ok = await extract_audio(TEST_MP4, audio_path)
        assert ok, "音频提取失败"

        # 2. ASR
        segments, detected_lang = run_asr(audio_path, mode=cfg.asr_mode, model_name=cfg.asr_model)
        assert len(segments) > 0, "ASR 未产生任何识别结果"

        # Generate source SRT
        source_srt = os.path.join(ASSETS_DIR, "1.source.srt")
        generate_srt(segments, source_srt)
        with open(source_srt, "r", encoding="utf-8") as f:
            content = f.read()
        assert "00:00:" in content

        # 3. Translate (only if online API configured)
        if cfg.translate_api_key and cfg.translate_api_url:
            translated = await translate_segments(
                segments, cfg.target_language,
                mode=cfg.translate_mode,
                api_url=cfg.translate_api_url,
                api_key=cfg.translate_api_key,
                model=cfg.translate_model,
                model_local=cfg.translate_model_local,
                thinking=cfg.translate_thinking,
            )
            if translated:
                target_srt = os.path.join(ASSETS_DIR, f"1.default.{cfg.target_language}.srt")
                generate_srt(translated, target_srt)
                assert os.path.exists(target_srt)

                bilingual = os.path.join(ASSETS_DIR, f"1.bilingual.{cfg.target_language}.srt")
                generate_bilingual_srt(segments, translated, bilingual)
                assert os.path.exists(bilingual)
