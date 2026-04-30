"""FFmpeg 音频提取 + ffprobe 字幕流检查。"""

import asyncio
import json
import logging
import subprocess

logger = logging.getLogger("uvicorn.error")


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


def get_audio_duration(audio_path: str) -> float:
    """获取音频文件时长（秒），返回 0 表示无法获取。"""
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        audio_path,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=10)
        if proc.returncode != 0:
            return 0.0
        data = json.loads(proc.stdout)
        return float(data.get("format", {}).get("duration", 0))
    except Exception:
        return 0.0
