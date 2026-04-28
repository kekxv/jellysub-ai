"""字幕检查模块测试。"""

import os
import tempfile
from pathlib import Path

from core.subtitle_checker import find_existing_subtitle, is_valid_subtitle


def test_is_valid_subtitle_utf8():
    """合法 UTF-8 字幕文件应通过校验。"""
    with tempfile.NamedTemporaryFile(suffix=".srt", mode="wb", delete=False) as f:
        f.write(b"1\n00:00:00,000 --> 00:00:03,000\n\xe4\xbd\xa0\xe5\xa5\xbd\xe4\xb8\x96\xe7\x95\x8c\n\n")
        f.flush()
        assert is_valid_subtitle(f.name) is True
    os.unlink(f.name)


def test_is_valid_subtitle_gbk():
    """GBK 编码字幕应被拒绝。"""
    with tempfile.NamedTemporaryFile(suffix=".srt", mode="wb", delete=False) as f:
        # GBK encoded "你好世界" = \xc4\xe3\xba\xc3\xca\xc0\xbd\xe7
        f.write(b"1\n00:00:00,000 --> 00:00:03,000\n\xc4\xe3\xba\xc3\xca\xc0\xbd\xe7\n\n")
        f.flush()
        assert is_valid_subtitle(f.name) is False
    os.unlink(f.name)


def test_is_valid_subtitle_empty():
    """空文件应被拒绝。"""
    with tempfile.NamedTemporaryFile(suffix=".srt", mode="wb", delete=False) as f:
        f.flush()
        assert is_valid_subtitle(f.name) is False
    os.unlink(f.name)


def test_is_valid_subtitle_binary():
    """二进制文件应被拒绝（可打印字符比例低）。"""
    with tempfile.NamedTemporaryFile(suffix=".srt", mode="wb", delete=False) as f:
        f.write(bytes(range(256)) * 10)
        f.flush()
        assert is_valid_subtitle(f.name) is False
    os.unlink(f.name)


def test_find_existing_subtitle_found():
    """目录中存在合法中文字幕应返回其路径。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        sub_path = os.path.join(tmpdir, "test_movie.zh-CN.srt")
        with open(sub_path, "w", encoding="utf-8") as f:
            f.write("1\n00:00:00,000 --> 00:00:03,000\n你好世界\n\n")
        result = find_existing_subtitle(tmpdir, "test_movie")
        assert result == sub_path


def test_find_existing_subtitle_not_found():
    """目录中无中文字幕应返回 None。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = find_existing_subtitle(tmpdir, "movie")
        assert result is None


def test_find_existing_subtitle_by_media_name():
    """通过媒体名前缀匹配找到字幕。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        sub_path = os.path.join(tmpdir, "my_movie.srt")
        with open(sub_path, "w", encoding="utf-8") as f:
            f.write("1\n00:00:00,000 --> 00:00:03,000\n你好世界\n\n")
        result = find_existing_subtitle(tmpdir, "my_movie")
        assert result == sub_path


def test_find_existing_subtitle_skips_non_utf8():
    """非 UTF-8 字幕应被跳过，返回 None。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        sub_path = os.path.join(tmpdir, "movie.zh.srt")
        with open(sub_path, "wb") as f:
            f.write(b"\xc4\xe3\xba\xc3")  # GBK
        result = find_existing_subtitle(tmpdir, "movie")
        assert result is None
