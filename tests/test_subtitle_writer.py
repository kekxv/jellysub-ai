"""SRT 字幕生成模块测试。"""

import os
import tempfile

from core.subtitle_writer import generate_srt, generate_bilingual_srt, _format_timestamp


def test_format_timestamp():
    """时间戳格式化应产生标准 SRT 格式。"""
    assert _format_timestamp(0.0) == "00:00:00,000"
    assert _format_timestamp(3.2) == "00:00:03,200"
    assert _format_timestamp(3661.456) == "01:01:01,456"


def test_generate_srt_basic():
    """生成基本 SRT 文件，内容正确。"""
    segments = [
        {"start": 0.0, "end": 3.0, "text": "Hello"},
        {"start": 3.5, "end": 6.0, "text": "World"},
    ]
    with tempfile.NamedTemporaryFile(suffix=".srt", mode="w", delete=False) as f:
        path = f.name

    try:
        assert generate_srt(segments, path) is True
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "00:00:00,000 --> 00:00:03,000" in content
        assert "Hello" in content
        assert "00:00:03,500 --> 00:00:06,000" in content
        assert "World" in content
    finally:
        os.unlink(path)


def test_generate_srt_chinese():
    """生成中文 SRT 文件，UTF-8 编码。"""
    segments = [
        {"start": 0.0, "end": 2.5, "text": "你好世界"},
    ]
    with tempfile.NamedTemporaryFile(suffix=".srt", mode="w", delete=False) as f:
        path = f.name

    try:
        generate_srt(segments, path)
        with open(path, "rb") as f:
            raw = f.read()
        raw.decode("utf-8")  # 不应抛出异常
        assert "你好世界" in raw.decode("utf-8")
    finally:
        os.unlink(path)


def test_generate_srt_empty():
    """空列表生成空文件。"""
    with tempfile.NamedTemporaryFile(suffix=".srt", mode="w", delete=False) as f:
        path = f.name

    try:
        assert generate_srt([], path) is True
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        assert content == ""
    finally:
        os.unlink(path)


def test_generate_srt_creates_parent_dir():
    """SRT 生成应自动创建父目录。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "subdir", "output.srt")
        segments = [{"start": 0.0, "end": 1.0, "text": "test"}]
        assert generate_srt(segments, path) is True
        assert os.path.exists(path)


def test_generate_bilingual_srt_basic():
    """生成双语 SRT，翻译在上、源语言在下。"""
    source = [
        {"start": 0.0, "end": 3.0, "text": "Hello world"},
        {"start": 3.5, "end": 6.0, "text": "Good morning"},
    ]
    translated = [
        {"start": 0.0, "end": 3.0, "text": "你好世界"},
        {"start": 3.5, "end": 6.0, "text": "早上好"},
    ]
    with tempfile.NamedTemporaryFile(suffix=".srt", mode="w", delete=False) as f:
        path = f.name

    try:
        assert generate_bilingual_srt(source, translated, path) is True
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        # 翻译在上，源语言在下
        assert "你好世界" in content
        assert "Hello world" in content
        assert "早上好" in content
        assert "Good morning" in content
        # 验证顺序：翻译行在源语言行之上
        lines = [l for l in content.split("\n") if l.strip()]
        zh_idx = lines.index("你好世界")
        en_idx = lines.index("Hello world")
        assert zh_idx < en_idx, "翻译文本应在源语言文本上方"
    finally:
        os.unlink(path)


def test_generate_bilingual_srt_chinese():
    """生成双语中文文件，UTF-8 编码。"""
    source = [{"start": 0.0, "end": 2.5, "text": "Hello world"}]
    translated = [{"start": 0.0, "end": 2.5, "text": "你好世界"}]
    with tempfile.NamedTemporaryFile(suffix=".srt", mode="w", delete=False) as f:
        path = f.name

    try:
        generate_bilingual_srt(source, translated, path)
        with open(path, "rb") as f:
            raw = f.read()
        raw.decode("utf-8")
        assert "你好世界" in raw.decode("utf-8")
        assert "Hello world" in raw.decode("utf-8")
    finally:
        os.unlink(path)


def test_generate_bilingual_srt_creates_parent_dir():
    """双语 SRT 生成应自动创建父目录。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "sub", "bilingual.srt")
        source = [{"start": 0.0, "end": 1.0, "text": "test"}]
        translated = [{"start": 0.0, "end": 1.0, "text": "测试"}]
        assert generate_bilingual_srt(source, translated, path) is True
        assert os.path.exists(path)
