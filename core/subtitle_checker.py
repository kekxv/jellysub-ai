"""检查已有字幕文件及其 UTF-8 编码合法性。"""

import logging
import re
from pathlib import Path

logger = logging.getLogger("uvicorn.error")

_SUBTITLE_EXTENSIONS = (".srt", ".vtt", ".ass")

# 字幕文件名中的语言/标签后缀，去掉后与视频名匹配
_LANG_TAGS_RE = re.compile(r'\.(?:zh|chi|chs|cht|cn|en|eng|ja|jpn|ko|kor|default|bilingual|foreign)(?:\.[^.]*)*$', re.IGNORECASE)


def _strip_lang_tags(stem: str) -> str:
    """去掉文件名末尾的语言/字幕标签后缀，返回核心名称。

    例: "Show.S01E01.default.zh-CN" -> "Show.S01E01"
    例: "Movie.2024.bilingual.zh-TW" -> "Movie.2024"
    """
    return _LANG_TAGS_RE.sub('', stem)


def _is_chinese_subtitle(path: Path) -> bool:
    """根据文件名判断是否为中文字幕。"""
    stem = path.stem.lower()
    last_tag = stem.rsplit(".", 1)[-1] if "." in stem else ""
    return last_tag in ("zh-cn", "zh-tw", "chi", "chs", "cht", "cn", "zho", "chinese")


def find_existing_subtitle(media_dir: str, media_name: str, language_hint: str = "zh") -> str | None:
    """扫描同级目录，返回与视频名匹配的中文字幕路径 (UTF-8 合法的)，未找到则返回 None。"""
    media_dir_path = Path(media_dir)
    if not media_dir_path.is_dir():
        return None

    # 视频的核心名称（去后缀）
    video_core = _strip_lang_tags(media_name)

    for sub_path in sorted(media_dir_path.iterdir()):
        if sub_path.suffix.lower() not in _SUBTITLE_EXTENSIONS:
            continue

        # 字幕去标签后的核心名必须与视频核心名完全一致
        sub_core = _strip_lang_tags(sub_path.stem)
        if sub_core != video_core:
            continue

        if is_valid_subtitle(str(sub_path)):
            logger.info("Found valid existing subtitle: %s", sub_path)
            return str(sub_path)

    return None


def is_valid_subtitle(filepath: str) -> bool:
    """检查字幕文件是否为合法 UTF-8 编码且包含有效文本内容。"""
    try:
        with open(filepath, "rb") as f:
            raw = f.read()
    except OSError as e:
        logger.warning("Cannot read subtitle file %s: %s", filepath, e)
        return False

    if not raw:
        return False

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        logger.warning("Subtitle %s is not valid UTF-8", filepath)
        return False

    printable_count = sum(1 for c in text if c.isprintable() or c in "\n\r\t")
    ratio = printable_count / max(len(text), 1)
    if ratio < 0.3:
        logger.warning("Subtitle %s has too few printable chars (%.1f%%)", filepath, ratio * 100)
        return False

    return True


async def has_any_subtitle(video_path: str, target_lang: str = "zh") -> tuple[bool, str]:
    """检查是否已有任何字幕（内置或外置）。

    Returns: (bool, reason)
    """
    vpath = Path(video_path)
    media_dir = str(vpath.parent)
    media_name = vpath.stem

    existing = find_existing_subtitle(media_dir, media_name, target_lang)
    if existing and is_valid_subtitle(existing):
        return True, f"external: {Path(existing).name}"

    from core.audio import has_internal_subtitle
    if await has_internal_subtitle(video_path):
        return True, "internal subtitle stream"

    return False, ""
