"""字幕内容解析 — 把 .srt / .vtt / .ass 字幕文件解析为带时间戳的片段列表。

统一输出格式：
    [{"start": float, "end": float, "text": str}, ...]

供「从现有字幕翻译」使用：解析结果可直接喂给 core.translate.translate_segments()，
时间轴保持不变，仅翻译文本。

注意：该模块只负责「读+解析」，不做编码从其它格式转换；若遇到非 UTF-8 文件，
调用方应先用 ffmpeg 转码或在此处做编码探测（当前统一按 UTF-8 读取）。
"""

import logging
import re
from pathlib import Path

logger = logging.getLogger("uvicorn.error")

# 支持的扩展名
SUBTITLE_EXTENSIONS = {".srt", ".vtt", ".ass", ".ssa"}

# SRT / VTT 时间码：HH:MM:SS,mmm 或 HH:MM:SS.mmm 或 MM:SS.mmm
_TIME_RE = re.compile(
    r"(?:(?P<h>\d{1,2}):)?(?P<m>\d{1,2}):(?P<s>\d{1,2})[.,](?P<ms>\d{1,3})"
)

# VTT 时间码可能不带小时，也可能用句点分隔毫秒
_VTT_TIME_RE = re.compile(
    r"(?:(?P<h>\d{1,2}):)?(?P<m>\d{1,2}):(?P<s>\d{2})[.,](?P<ms>\d{1,3})"
)


def _to_seconds(h: str | None, m: str, s: str, ms: str) -> float:
    """把时间码各分量换算为秒。"""
    hours = int(h) if h else 0
    minutes = int(m)
    seconds = int(s)
    millis = int(ms.ljust(3, "0")[:3])
    return hours * 3600 + minutes * 60 + seconds + millis / 1000.0


def _clean_text(text: str) -> str:
    """清理字幕文本：去掉 HTML 标签、把 ASS 的 \\N/\\n 换成换行、\\h 换成空格。"""
    if not text:
        return ""
    # ASS/MicroDVD 换行符
    text = text.replace("\\N", "\n").replace("\\n", "\n").replace("\\h", " ")
    # 去掉形如 <i>, <b>, <font> 等 HTML 标签
    text = re.sub(r"<[^>]+>", "", text)
    # 去掉 ASS 覆盖标签 {\...}
    text = re.sub(r"\{[^}]*\}", "", text)
    # 压缩连续空行
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


# --------------------------------------------------------------------------- #
#  SRT
# --------------------------------------------------------------------------- #

def parse_srt(text: str) -> list[dict]:
    """解析 SRT 文本为片段列表。"""
    segments: list[dict] = []
    blocks = re.split(r"\n\s*\n", text.strip())
    for block in blocks:
        lines = [ln for ln in block.split("\n") if ln.strip()]
        if not lines:
            continue
        time_idx = -1
        for i, ln in enumerate(lines):
            if "-->" in ln:
                time_idx = i
                break
        if time_idx == -1:
            continue
        ts = _parse_time_line(lines[time_idx], _TIME_RE)
        if ts is None:
            continue
        start, end = ts
        body = "\n".join(lines[time_idx + 1:])
        body = _clean_text(body)
        if not body:
            continue
        segments.append({"start": start, "end": end, "text": body})
    return segments


# --------------------------------------------------------------------------- #
#  VTT
# --------------------------------------------------------------------------- #

def parse_vtt(text: str) -> list[dict]:
    """解析 WebVTT 文本为片段列表。"""
    segments: list[dict] = []
    # 去掉 WEBVTT 头及注释块
    lines = text.split("\n")
    body = "\n".join(lines)
    blocks = re.split(r"\n\s*\n", body.strip())
    for block in blocks:
        lines = [ln for ln in block.split("\n") if ln.strip()]
        if not lines:
            continue
        time_idx = -1
        for i, ln in enumerate(lines):
            if "-->" in ln:
                time_idx = i
                break
        if time_idx == -1:
            continue
        # VTT 行可能是 "00:00.000 --> 00:02.000 align:start position:0%"
        ts = _parse_time_line(lines[time_idx], _VTT_TIME_RE)
        if ts is None:
            continue
        start, end = ts
        body = "\n".join(lines[time_idx + 1:])
        body = _clean_text(body)
        if not body:
            continue
        segments.append({"start": start, "end": end, "text": body})
    return segments


def _parse_time_line(line: str, time_re: re.Pattern) -> tuple[float, float] | None:
    """从一行时间码文本中解析 (start, end) 秒。"""
    arrow = line.split("-->")
    if len(arrow) < 2:
        return None

    def _parse(part: str) -> float | None:
        m = time_re.search(part.strip())
        if not m:
            return None
        return _to_seconds(m.group("h"), m.group("m"), m.group("s"), m.group("ms"))

    start = _parse(arrow[0])
    end = _parse(arrow[1])
    if start is None or end is None:
        return None
    return start, end


# --------------------------------------------------------------------------- #
#  ASS / SSA
# --------------------------------------------------------------------------- #

def parse_ass(text: str, format_line: str | None = None) -> list[dict]:
    """解析 ASS/SSA 的 Dialogue 行为片段列表。

    format_line 若为 None 则尝试从 [Events] 段的 Format 行自动推断列顺序，
    否则按标准 ASS 默认列顺序（Layer, Start, End, Style, Name, MarginL, MarginR,
    MarginV, Effect, Text）。
    """
    segments: list[dict] = []

    # 找到 [Events] 段里的 Format 列
    cols = None
    if format_line is None:
        in_events = False
        for ln in text.split("\n"):
            if ln.strip().lower() == "[events]":
                in_events = True
                continue
            if in_events and ln.strip().lower().startswith("format:"):
                cols = [c.strip() for c in ln.split(":", 1)[1].split(",")]
                break
            if in_events and ln.strip() and not ln.strip().startswith((";", "Dialogue", "Comment")):
                break
    else:
        cols = [c.strip() for c in format_line.split(",")]

    if cols is None:
        cols = ["Layer", "Start", "End", "Style", "Name", "MarginL", "MarginR", "MarginV", "Effect", "Text"]

    # 标准文本列位置（按名称找；找不到则用默认位置，Text 通常在最后）
    start_idx = cols.index("Start") if "Start" in cols else 1
    end_idx = cols.index("End") if "End" in cols else 2
    text_idx = cols.index("Text") if "Text" in cols else len(cols) - 1

    for ln in text.split("\n"):
        stripped = ln.strip()
        if not stripped.lower().startswith("dialogue:"):
            continue
        # 去掉前缀
        payload = stripped.split(":", 1)[1]
        fields = payload.split(",")
        if len(fields) <= max(start_idx, end_idx, text_idx):
            continue
        try:
            start = _ass_time_to_seconds(fields[start_idx])
            end = _ass_time_to_seconds(fields[end_idx])
        except (ValueError, IndexError):
            continue
        # Text 列可能包含逗号，需把第 text_idx 列之后的字段重新拼接
        body = ",".join(fields[text_idx:]).strip()
        body = _clean_text(body)
        if not body:
            continue
        segments.append({"start": start, "end": end, "text": body})

    return segments


def _ass_time_to_seconds(value: str) -> float:
    """ASS 时间格式 H:MM:SS.cc（厘秒）转秒。"""
    value = value.strip()
    parts = value.split(":")
    if len(parts) != 3:
        raise ValueError(value)
    h, m, s = parts
    if "." in s:
        sec, frac = s.split(".")
        fraction = int(frac.ljust(2, "0")[:2]) / 100.0
    else:
        sec, fraction = s, 0.0
    return int(h) * 3600 + int(m) * 60 + int(sec) + fraction


# --------------------------------------------------------------------------- #
#  统一入口
# --------------------------------------------------------------------------- #

def parse_subtitle_text(text: str, ext: str) -> list[dict]:
    """按字幕格式解析文本，ext 传入带点的扩展名（如 '.srt'）。"""
    ext = ext.lower()
    if ext == ".srt":
        return parse_srt(text)
    if ext == ".vtt":
        return parse_vtt(text)
    if ext in (".ass", ".ssa"):
        return parse_ass(text)
    # 尝试自动探测
    if "webvtt" in text[:200].lower():
        return parse_vtt(text)
    if "dialogue:" in text.lower():
        return parse_ass(text)
    return parse_srt(text)


def parse_subtitle_file(path: str) -> list[dict]:
    """从字幕文件解析出片段列表（按 UTF-8 读取，非法则跳过）。"""
    try:
        raw = Path(path).read_bytes()
    except OSError as e:
        logger.warning("Cannot read subtitle file %s: %s", path, e)
        return []
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        logger.warning("Subtitle %s is not valid UTF-8", path)
        return []
    ext = Path(path).suffix.lower()
    return parse_subtitle_text(text, ext)
