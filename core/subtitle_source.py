"""字幕来源发现与解析 — 供「从已有字幕翻译」使用。

来源分两类：
1. 外部字幕文件：与视频同目录、且去掉语言后缀后与视频同名匹配的 .srt/.vtt/.ass 文件。
2. 内置字幕流：视频容器（如 MKV/MP4）里 ffprobe 探测到的字幕流，可经 ffmpeg 抽出。

对外提供：
- get_subtitle_sources(video_path)  -> 候选来源列表（供前端展示/选择）
- resolve_subtitle_source(...)      -> 把某个候选解析成可直接读取的字幕片段（内部流会先抽出为临时文件）
- read_subtitle_segments(...)       -> 读取并解析出 [{start,end,text}] 片段列表
"""

import asyncio
import json
import logging
import subprocess
from pathlib import Path

from core.subtitle_parse import SUBTITLE_EXTENSIONS, parse_subtitle_file
from core.subtitle_checker import _strip_lang_tags

logger = logging.getLogger("uvicorn.error")

# 常见语言标签 → 展示名
_LANG_NAMES = {
    "en": "English", "eng": "English", "zh": "中文", "chi": "中文", "chs": "简体中文",
    "cht": "繁體中文", "cn": "中文", "zho": "中文", "ja": "日本語", "jpn": "日本語",
    "ko": "한국어", "kor": "한국어", "zh-cn": "简体中文", "zh-tw": "繁體中文",
    "zh-hans": "简体中文", "zh-hant": "繁體中文", "yue": "粤语",
}


def _is_subtitle_ext(suffix: str) -> bool:
    return suffix.lower() in SUBTITLE_EXTENSIONS or suffix.lower() == ".ssa"


def _is_generated_output(stem: str) -> bool:
    """判断是否为项目生成的输出字幕（.default.<lang> / .bilingual.<lang>）。"""
    s = stem.lower()
    return ".default." in s or ".bilingual." in s


def _candidate_sort_key(path: Path) -> tuple:
    """排序：优先非生成输出、其次按文件名。生成产物排后面。"""
    return (_is_generated_output(path.stem), path.name)


def _detect_lang(stem: str) -> str:
    """从文件名（去扩展名）推断语言代码，如 'en'、'zh-cn'、'zh'。"""
    stem = stem.lower()
    # 直接去掉语言标签后缀前的部分；这里简单提取最后的标签段
    core = _strip_lang_tags(stem)
    remainder = stem[len(core):].lstrip(".")
    if not remainder:
        # 也许核心名本身含标签，尝试拆最后一段
        parts = stem.split(".")
        if len(parts) >= 2:
            remainder = parts[-1]
    # 统一规范化：'eng' -> 'en'，'chi'/'zho' -> 'zh'
    normalized = remainder
    if len(normalized) == 3:
        for code, aliases in _LANG_GROUP_ALIASES.items():
            if normalized in aliases:
                normalized = code
                break
    return normalized


_LANG_GROUP_ALIASES = {
    "en": {"en", "eng"},
    "zh": {"zh", "chi", "chs", "cht", "cn", "zho"},
    "ja": {"ja", "jpn"},
    "ko": {"ko", "kor"},
}


def _lang_display_name(lang: str) -> str:
    return _LANG_NAMES.get(lang.lower(), lang)


# --------------------------------------------------------------------------- #
#  外部字幕文件
# --------------------------------------------------------------------------- #

def find_external_subtitles(video_path: str) -> list[dict]:
    """返回与视频同名匹配的外部字幕文件候选列表。"""
    vpath = Path(video_path)
    media_dir = vpath.parent
    video_core = _strip_lang_tags(vpath.stem)
    if not media_dir.is_dir():
        return []

    results = []
    try:
        entries = sorted(media_dir.iterdir(), key=_candidate_sort_key)
    except OSError:
        return []
    for sub_path in entries:
        if not _is_subtitle_ext(sub_path.suffix):
            continue
        if _strip_lang_tags(sub_path.stem) != video_core:
            continue
        results.append({
            "kind": "external",
            "path": str(sub_path),
            "name": sub_path.name,
            "lang": _detect_lang(sub_path.stem),
            "display": sub_path.name,
        })
    return results


# --------------------------------------------------------------------------- #
#  内置字幕流
# --------------------------------------------------------------------------- #

async def list_internal_subtitle_streams(video_path: str) -> list[dict]:
    """使用 ffprobe 列出视频内的字幕流。"""
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-select_streams", "s",
        "-show_streams",
        video_path,
    ]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        if proc.returncode != 0:
            return []
        data = json.loads(stdout)
    except Exception:
        logger.exception("ffprobe internal subtitle check failed")
        return []

    streams = data.get("streams", [])
    results = []
    for idx, stream in enumerate(streams):
        codec = stream.get("codec_name") or stream.get("codec_type") or "?"
        tags = stream.get("tags", {}) or {}
        lang_tag = tags.get("language", "") or ""
        title = tags.get("title", "")
        # 只保留文本字幕；图形字幕（pgs/dvbsub）无法直接转文本
        if codec in ("hdmv_pgs_subtitle", "dvd_subtitle", "dvb_subtitle", "dvb_teletext"):
            continue
        label = f"内置字幕 #{idx + 1} ({codec}"
        if lang_tag:
            label += f", {_lang_display_name(lang_tag)}"
        if title:
            label += f", {title}"
        label += ")"
        results.append({
            "kind": "internal",
            "index": stream.get("index", idx),
            "codec": codec,
            "lang": lang_tag,
            "title": title,
            "display": label,
        })
    return results


async def get_subtitle_sources(video_path: str) -> list[dict]:
    """返回视频可用的全部字幕候选（外部文件优先，内置流在后）。"""
    sources = find_external_subtitles(video_path)
    internal = await list_internal_subtitle_streams(video_path)
    sources.extend(internal)
    return sources


# --------------------------------------------------------------------------- #
#  内置字幕流抽取
# --------------------------------------------------------------------------- #

async def extract_internal_subtitle(video_path: str, stream_index: int, out_path: str) -> bool:
    """把视频内第 stream_index 个流抽出来存为 out_path（带 .srt 后缀以触发文本转码）。"""
    cmd = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-map", f"0:{stream_index}",
        out_path,
    ]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            logger.error("ffmpeg extract subtitle failed: %s", stderr.decode(errors="replace"))
            return False
        return True
    except FileNotFoundError:
        logger.error("ffmpeg not found, cannot extract embedded subtitle")
        return False
    except Exception:
        logger.exception("Internal subtitle extraction failed")
        return False


# --------------------------------------------------------------------------- #
#  解析/解析到片段
# --------------------------------------------------------------------------- #

def read_subtitle_segments(source_path: str) -> list[dict]:
    """读取已落盘的字幕文件并解析为片段列表。"""
    return parse_subtitle_file(source_path)


async def resolve_subtitle_source(
    video_path: str,
    source_path: str | None = None,
    source_index: int | None = None,
    work_dir: str | None = None,
) -> dict:
    """把用户选择或自动推断的字幕来源解析为可读取的字幕文件。

    返回 dict：
        {"kind", "path", "lang", "source_path", "source_index", "display"}
    path 是实际可读的文件路径；对内部流会先抽取到 work_dir 下的临时文件。

    找不到任何可用来源时抛出 ValueError。
    """
    vpath = Path(video_path)
    vstem = vpath.stem

    # 1) 用户明确指定了外部文件
    if source_path:
        apath = Path(source_path)
        if apath.is_file():
            lang = _detect_lang(apath.stem)
            return {
                "kind": "external",
                "path": str(apath),
                "lang": lang,
                "source_path": str(apath),
                "source_index": None,
                "display": apath.name,
            }
        logger.warning("Specified source_path not found: %s", source_path)

    # 2) 用户指定了内部流索引
    if source_index is not None:
        ext_path = await _extract_stream_to_tmp(video_path, source_index, work_dir)
        if ext_path:
            lang = ""
            return {
                "kind": "internal",
                "path": ext_path,
                "lang": lang,
                "source_path": None,
                "source_index": source_index,
                "display": f"内置字幕流 #{source_index}",
            }

    # 3) 自动选择：优先外部字幕文件，其次内置流
    ext_sources = find_external_subtitles(video_path)
    if ext_sources:
        src = ext_sources[0]
        return {
            "kind": "external",
            "path": src["path"],
            "lang": src["lang"],
            "source_path": src["path"],
            "source_index": None,
            "display": src["display"],
        }

    internal = await list_internal_subtitle_streams(video_path)
    if internal:
        src = internal[0]
        ext_path = await _extract_stream_to_tmp(video_path, src["index"], work_dir)
        if ext_path:
            return {
                "kind": "internal",
                "path": ext_path,
                "lang": src["lang"],
                "source_path": None,
                "source_index": src["index"],
                "display": src["display"],
            }

    raise ValueError(f"No usable subtitle source found for {vstem}")


async def _extract_stream_to_tmp(video_path: str, stream_index: int, work_dir: str | None) -> str | None:
    """把内部字幕流抽到临时目录并返回路径；失败返回 None。"""
    import hashlib
    from pathlib import Path as _P

    work = _P(work_dir) if work_dir else _P(video_path).parent
    work.mkdir(parents=True, exist_ok=True)
    nhash = hashlib.sha256((f"{video_path}:{stream_index}").encode("utf-8")).hexdigest()[:10]
    out_path = work / f"subsrc_{nhash}.srt"
    ok = await extract_internal_subtitle(video_path, stream_index, str(out_path))
    if not ok or not out_path.exists() or out_path.stat().st_size == 0:
        logger.warning("Extracted internal subtitle stream %d collides or empty", stream_index)
        _safe_unlink(out_path)
        return None
    return str(out_path)


def _safe_unlink(p: Path):
    try:
        p.unlink(missing_ok=True)
    except Exception:
        pass
