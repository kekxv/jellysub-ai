"""新增「从现有字幕翻译」相关模块的单元测试。"""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.subtitle_parse import (
    parse_srt,
    parse_vtt,
    parse_ass,
    parse_subtitle_file,
    SUBTITLE_EXTENSIONS,
)
from core.subtitle_source import (
    find_external_subtitles,
    list_internal_subtitle_streams,
    extract_internal_subtitle,
    resolve_subtitle_source,
)


# --------------------------------------------------------------------------- #
#  subtitle_parse
# --------------------------------------------------------------------------- #

def test_parse_srt():
    text = """1
00:00:01,000 --> 00:00:03,000
Hello world

2
00:00:05.000 --> 00:00:07,500
Line two
"""
    segs = parse_srt(text)
    assert segs == [
        {"start": 1.0, "end": 3.0, "text": "Hello world"},
        {"start": 5.0, "end": 7.5, "text": "Line two"},
    ]


def test_parse_vtt():
    text = """WEBVTT

00:00:01.000 --> 00:00:03.000 align:start
Hi there

01:00:00.500 --> 01:00:02.000
<v Bob>Hello</v>
"""
    segs = parse_vtt(text)
    assert segs == [
        {"start": 1.0, "end": 3.0, "text": "Hi there"},
        {"start": 3600.5, "end": 3602.0, "text": "Hello"},
    ]


def test_parse_ass():
    text = """[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,Hello,\\Nworld
Dialogue: 0,0:00:05.00,0:00:07.00,Default,,0,0,0,,{\\i1}Bye{\\i0}
"""
    segs = parse_ass(text)
    assert segs == [
        {"start": 1.0, "end": 3.0, "text": "Hello,\nworld"},
        {"start": 5.0, "end": 7.0, "text": "Bye"},
    ]


def test_parse_subtitle_file(tmp_path):
    srt = tmp_path / "a.srt"
    srt.write_text("1\n00:00:01,000 --> 00:00:02,000\nHi\n", encoding="utf-8")
    assert parse_subtitle_file(str(srt)) == [{"start": 1.0, "end": 2.0, "text": "Hi"}]

    # 非 UTF-8 文件应返回空，而不是抛异常
    bad = tmp_path / "b.srt"
    bad.write_bytes(b"\xff\xfe\x00\x01")
    assert parse_subtitle_file(str(bad)) == []


# --------------------------------------------------------------------------- #
#  subtitle_source — external files
# --------------------------------------------------------------------------- #

def test_find_external_subtitles_matches_name(tmp_path):
    video = tmp_path / "Movie (2024).mkv"
    video.touch()
    (tmp_path / "Movie (2024).en.srt").write_text("x", encoding="utf-8")
    (tmp_path / "Other.en.srt").write_text("x", encoding="utf-8")
    (tmp_path / "Movie (2024).default.zh-CN.srt").write_text("x", encoding="utf-8")

    found = find_external_subtitles(str(video))
    names = [f["name"] for f in found]
    assert "Movie (2024).en.srt" in names
    assert "Movie (2024).default.zh-CN.srt" in names
    assert "Other.en.srt" not in names


def test_find_external_subtitles_orders_generated_last(tmp_path):
    video = tmp_path / "S01E01.mkv"
    video.touch()
    (tmp_path / "S01E01.en.srt").write_text("x", encoding="utf-8")
    (tmp_path / "S01E01.default.zh-CN.srt").write_text("x", encoding="utf-8")

    found = find_external_subtitles(str(video))
    assert found[0]["name"] == "S01E01.en.srt"  # 非生成输出排在前面


def test_find_external_subtitles_no_match(tmp_path):
    video = tmp_path / "movie.mp4"
    video.touch()
    (tmp_path / "unrelated.srt").write_text("x", encoding="utf-8")
    assert find_external_subtitles(str(video)) == []


def test_find_external_subtitles_detects_lang(tmp_path):
    video = tmp_path / "movie.mkv"
    video.touch()
    (tmp_path / "movie.en.srt").write_text("x", encoding="utf-8")
    found = find_external_subtitles(str(video))
    assert found[0]["lang"] == "en"


# --------------------------------------------------------------------------- #
#  subtitle_source — internal streams (mocked ffprobe)
# --------------------------------------------------------------------------- #

def _fake_proc(returncode=0, stdout=b"", stderr=b""):
    proc = MagicMock()
    proc.communicate = AsyncMock(return_value=(stdout, stderr))
    proc.returncode = returncode
    return proc


@pytest.mark.asyncio
async def test_list_internal_subtitle_streams(monkeypatch):
    payload = {
        "streams": [
            {"index": 1, "codec_name": "subrip", "tags": {"language": "eng"}},
            {"index": 2, "codec_name": "ass", "tags": {"language": "chi", "title": "简体"}},
            {"index": 3, "codec_name": "hdmv_pgs_subtitle", "tags": {}},  # 图形字幕应被过滤
        ]
    }
    proc = _fake_proc(stdout=json.dumps(payload).encode())
    monkeypatch.setattr(
        "core.subtitle_source.asyncio.create_subprocess_exec",
        AsyncMock(return_value=proc),
    )

    streams = await list_internal_subtitle_streams("/media/movie.mkv")
    assert len(streams) == 2
    assert streams[0]["codec"] == "subrip"
    assert streams[1]["lang"] == "chi"
    assert streams[0]["kind"] == "internal"


@pytest.mark.asyncio
async def test_list_internal_subtitle_streams_ffprobe_failure(monkeypatch):
    proc = _fake_proc(returncode=1, stderr=b"error")
    monkeypatch.setattr(
        "core.subtitle_source.asyncio.create_subprocess_exec",
        AsyncMock(return_value=proc),
    )
    assert await list_internal_subtitle_streams("/media/movie.mkv") == []


@pytest.mark.asyncio
async def test_extract_internal_subtitle_success(monkeypatch):
    proc = _fake_proc(returncode=0)
    monkeypatch.setattr(
        "core.subtitle_source.asyncio.create_subprocess_exec",
        AsyncMock(return_value=proc),
    )
    assert await extract_internal_subtitle("/media/movie.mkv", 1, "/tmp/out.srt") is True


# --------------------------------------------------------------------------- #
#  subtitle_source — resolve
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_resolve_source_explicit_external_file(tmp_path):
    video = tmp_path / "movie.mkv"
    video.touch()
    srt = tmp_path / "movie.en.srt"
    srt.write_text("1\n00:00:01,000 --> 00:00:02,000\nHi\n", encoding="utf-8")

    resolved = await resolve_subtitle_source(str(video), source_path=str(srt))
    assert resolved["kind"] == "external"
    assert resolved["path"] == str(srt)
    assert resolved["lang"] == "en"


@pytest.mark.asyncio
async def test_resolve_source_auto_external(tmp_path):
    video = tmp_path / "movie.mkv"
    video.touch()
    (tmp_path / "movie.en.srt").write_text("x", encoding="utf-8")

    resolved = await resolve_subtitle_source(str(video), work_dir=str(tmp_path))
    assert resolved["kind"] == "external"
    assert resolved["path"].endswith("movie.en.srt")


@pytest.mark.asyncio
async def test_resolve_source_auto_internal(monkeypatch, tmp_path):
    video = tmp_path / "movie.mkv"
    video.touch()
    payload = {"streams": [{"index": 1, "codec_name": "subrip", "tags": {"language": "eng"}}]}
    probe = _fake_proc(stdout=json.dumps(payload).encode())
    monkeypatch.setattr(
        "core.subtitle_source.asyncio.create_subprocess_exec",
        AsyncMock(return_value=probe),
    )
    # 模拟 ffmpeg 抽取成功：给它后接一个会创建临时文件的副作用
    async def fake_extract(video_path, stream_index, out_path):
        Path(out_path).write_text("1\n00:00:01,000 --> 00:00:02,000\nHi\n", encoding="utf-8")
        return True

    monkeypatch.setattr("core.subtitle_source.extract_internal_subtitle", fake_extract)

    resolved = await resolve_subtitle_source(str(video), work_dir=str(tmp_path))
    assert resolved["kind"] == "internal"
    assert resolved["source_index"] == 1
    assert Path(resolved["path"]).exists()


@pytest.mark.asyncio
async def test_resolve_source_no_sources_raises(monkeypatch, tmp_path):
    video = tmp_path / "movie.mkv"
    video.touch()
    proc = _fake_proc(stdout=b'{"streams": []}')
    monkeypatch.setattr(
        "core.subtitle_source.asyncio.create_subprocess_exec",
        AsyncMock(return_value=proc),
    )
    with pytest.raises(ValueError):
        await resolve_subtitle_source(str(video), work_dir=str(tmp_path))
