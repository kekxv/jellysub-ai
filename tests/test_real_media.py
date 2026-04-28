"""使用真实 MP4 文件测试音频提取和字幕流检测。"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

from config import AppConfig
from core.audio import extract_audio, has_internal_subtitle
from core.subtitle_checker import find_existing_subtitle
from core.subtitle_writer import generate_srt

# 项目根目录下的 assets/en.mp4
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
TEST_MP4 = os.path.join(PROJECT_ROOT, "assets", "en.mp4")


@pytest.fixture
def tmp_audio(tmp_path):
    """生成临时 WAV 输出路径。"""
    return str(tmp_path / "extracted.wav")


@pytest.fixture
def mock_config(tmp_path):
    """使用临时配置，避免被 config.json 影响。"""
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
        translate_model_local="Helsinki-NLP/opus-mt-en-zh",
        translate_thinking=False,
        path_mappings={},
        temp_dir=str(tmp_path),
    )
    with patch("main.get_config", return_value=cfg):
        with patch("config.get_config", return_value=cfg):
            yield cfg


@pytest.mark.asyncio
async def test_extract_audio_real_mp4(tmp_audio):
    """从真实 MP4 提取音频，应成功并生成 WAV 文件。"""
    assert os.path.exists(TEST_MP4)
    result = await extract_audio(TEST_MP4, tmp_audio)
    assert result is True
    assert os.path.exists(tmp_audio)
    assert os.path.getsize(tmp_audio) > 0


@pytest.mark.asyncio
async def test_has_internal_subtitle_real_mp4():
    """真实 MP4 不含内置字幕流，应返回 False。"""
    assert os.path.exists(TEST_MP4)
    result = await has_internal_subtitle(TEST_MP4)
    assert result is False


def test_subtitle_checker_finds_generated_subtitle():
    """生成的中文字幕应被检测到。"""
    media_dir = os.path.dirname(TEST_MP4)
    media_name = os.path.splitext(os.path.basename(TEST_MP4))[0]
    result = find_existing_subtitle(media_dir, media_name)
    # 如果之前生成了字幕，应找到它
    if os.path.exists(os.path.join(media_dir, f"{media_name}.default.zh-CN.srt")):
        assert result is not None, "Should find existing Chinese subtitle"


def test_task_pipeline_with_real_mp4(mock_config, tmp_path):
    """
    完整流程测试：通过 TaskManager 的 _execute_pipeline，Mock ASR 和翻译。
    字幕生成在 MP4 同级目录。
    """
    import shutil
    from core.task_manager import TaskManager

    media_dir = str(tmp_path)

    # 复制测试 MP4 到临时目录
    test_mp4 = str(tmp_path / "test_movie.mp4")
    shutil.copy(TEST_MP4, test_mp4)

    mock_segments = [
        {"start": 0.0, "end": 2.5, "text": "Hello world"},
        {"start": 3.0, "end": 5.5, "text": "This is a test"},
    ]

    mock_translated = [
        {"start": 0.0, "end": 2.5, "text": "你好世界"},
        {"start": 3.0, "end": 5.5, "text": "这是一个测试"},
    ]

    # Create task in DB
    tm = TaskManager(db_path=str(tmp_path / "tasks.db"))
    task_id = tm.create_task(
        video_path=test_mp4,
        item_id="test-item-123",
        item_type="Movie",
        item_name="test_movie",
        pipeline_type="webhook",
    )

    mock_translate = AsyncMock(return_value=mock_translated)
    mock_jellyfin = MagicMock()
    mock_jellyfin.refresh_item = AsyncMock(return_value=True)

    with patch("core.asr.run_asr", return_value=(mock_segments, "en")):
        with patch("core.translate.translate_segments", mock_translate):
            with patch("core.jellyfin_api.JellyfinClient", return_value=mock_jellyfin):
                task = {"id": task_id, "video_path": test_mp4,
                        "item_id": "test-item-123", "item_type": "Movie",
                        "item_name": "test_movie", "pipeline_type": "webhook"}
                tm._execute_pipeline(task)

    # Verify translated was called
    assert mock_translate.call_count == 1

    # Verify Jellyfin refresh was called
    mock_jellyfin.refresh_item.assert_called_once()

    # Verify subtitle files in MP4 sibling directory
    target_srt = os.path.join(media_dir, "test_movie.default.zh-CN.srt")
    assert os.path.exists(target_srt), f"Subtitle not found: {target_srt}"
    with open(target_srt, "r", encoding="utf-8") as f:
        content = f.read()
    assert "你好世界" in content
    assert "这是一个测试" in content

    bilingual_srt = os.path.join(media_dir, "test_movie.bilingual.zh-CN.srt")
    assert os.path.exists(bilingual_srt), f"Bilingual subtitle not found: {bilingual_srt}"
    with open(bilingual_srt, "r", encoding="utf-8") as f:
        bi_content = f.read()
    assert "你好世界" in bi_content
    assert "Hello world" in bi_content


def test_silence_gaps_between_segments():
    """相邻字幕段之间应有静音间隙，给观众阅读时间。"""
    from core.asr import _add_silence_gaps

    segments = [
        {"start": 0.0, "end": 3.0, "text": "Hello world"},
        {"start": 3.0, "end": 5.0, "text": "Good morning"},
        {"start": 5.5, "end": 7.0, "text": "Already has gap"},
    ]
    result = _add_silence_gaps(segments)

    # 第一段和第二段紧接，应缩短第一段 end 留出间隙
    assert result[0]["end"] < result[1]["start"], (
        f"No gap between segments: {result[0]['end']} >= {result[1]['start']}"
    )

    # 第二段和第三段已有间隙，应保持不变
    assert result[1]["end"] == 5.0
    assert result[2]["start"] == 5.5


def test_silence_gaps_no_change_needed():
    """已有足够间隙的片段不应被修改。"""
    from core.asr import _add_silence_gaps

    segments = [
        {"start": 0.0, "end": 2.0, "text": "Hello"},
        {"start": 3.0, "end": 5.0, "text": "World"},
    ]
    result = _add_silence_gaps(segments)
    # 已有 1s 间隙，不应被修改
    assert result[0]["end"] == 2.0
    assert result[1]["start"] == 3.0

