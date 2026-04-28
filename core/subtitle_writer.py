"""SRT 字幕文件生成器 (UTF-8 编码)。"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _format_timestamp(seconds: float) -> str:
    """将秒数转为 SRT 时间格式 00:00:00,000。"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def generate_bilingual_srt(
    source_segments: list[dict],
    translated_segments: list[dict],
    output_path: str,
) -> bool:
    """
    从源语言字幕和翻译字幕生成双语 SRT 文件。
    每个字幕块包含两行：翻译文本在上，源文本在下。
    """
    try:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for i, (src, tgt) in enumerate(zip(source_segments, translated_segments), 1):
                start = _format_timestamp(src["start"])
                end = _format_timestamp(src["end"])
                target_text = tgt.get("text", "").strip()
                source_text = src.get("text", "").strip()
                f.write(f"{i}\n{start} --> {end}\n{target_text}\n{source_text}\n\n")
        logger.info("Bilingual SRT generated: %s (%d segments)", output_path, len(source_segments))
        return True
    except Exception:
        logger.exception("Failed to generate bilingual SRT: %s", output_path)
        return False


def generate_srt(segments: list[dict], output_path: str) -> bool:
    """
    从带时间戳的文本片段列表生成 SRT 文件。
    segments: [{"start": 0.0, "end": 3.2, "text": "你好"}, ...]
    """
    try:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for i, seg in enumerate(segments, 1):
                start = _format_timestamp(seg["start"])
                end = _format_timestamp(seg["end"])
                text = seg.get("text", "").strip()
                f.write(f"{i}\n{start} --> {end}\n{text}\n\n")
        logger.info("SRT generated: %s (%d segments)", output_path, len(segments))
        return True
    except Exception:
        logger.exception("Failed to generate SRT: %s", output_path)
        return False
