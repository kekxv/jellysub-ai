"""检查已有字幕文件及其 UTF-8 编码合法性。"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_SUBTITLE_EXTENSIONS = (".srt", ".vtt", ".ass")

_CHINESE_HINTS = ("zh", "chi", "chs", "cht", "cn", "chs", "zh-CN", "zh-TW", "Chinese")


def find_existing_subtitle(media_dir: str, media_name: str, language_hint: str = "zh") -> str | None:
    """扫描同级目录，返回已有的中文字幕路径 (UTF-8 合法的)，未找到则返回 None。"""
    media_dir_path = Path(media_dir)
    if not media_dir_path.is_dir():
        return None

    for sub_path in sorted(media_dir_path.iterdir()):
        if sub_path.suffix.lower() not in _SUBTITLE_EXTENSIONS:
            continue
        # 文件名需包含媒体名前缀或语言提示
        stem = sub_path.stem.lower()
        if media_name.lower() not in stem and not any(h in stem for h in _CHINESE_HINTS):
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

    # 尝试 UTF-8 解码
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        logger.warning("Subtitle %s is not valid UTF-8", filepath)
        return False

    # 检查可打印字符比例 (至少 30% 可打印)
    printable_count = sum(1 for c in text if c.isprintable() or c in "\n\r\t")
    ratio = printable_count / max(len(text), 1)
    if ratio < 0.3:
        logger.warning("Subtitle %s has too few printable chars (%.1f%%)", filepath, ratio * 100)
        return False

    return True
