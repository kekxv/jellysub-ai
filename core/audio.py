"""FFmpeg 音频提取 + ffprobe 字幕流检查。"""

import asyncio
import json
import logging
import subprocess

logger = logging.getLogger(__name__)


async def extract_audio(media_path: str, output_path: str) -> bool:
    """从视频中提取音频，转为 16kHz 单声道 WAV。"""
    cmd = [
        "ffmpeg",
        "-i", media_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        "-y",
        output_path,
    ]
    logger.info("Extracting audio: %s -> %s", media_path, output_path)
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            logger.error("ffmpeg failed: %s", stderr.decode(errors="replace"))
            return False
        return True
    except FileNotFoundError:
        logger.error("ffmpeg not found, please install it")
        return False
    except Exception:
        logger.exception("Audio extraction failed")
        return False


async def has_internal_subtitle(media_path: str) -> bool:
    """使用 ffprobe 检查是否包含字幕流。"""
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-select_streams", "s",
        "-show_streams",
        media_path,
    ]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            return False
        data = json.loads(stdout)
        streams = data.get("streams", [])
        return len(streams) > 0
    except Exception:
        logger.exception("ffprobe subtitle check failed")
        return False
