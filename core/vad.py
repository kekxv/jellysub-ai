"""Silero VAD — 语音活动检测，用于长音频分块。"""

import logging
import os
import subprocess
from dataclasses import dataclass

import torch

logger = logging.getLogger("uvicorn.error")

_silero_model = None
VAD_MODEL_DIR = "./model_cache/silero_vad"
VAD_MODEL_URL = "https://github.com/snakers4/silero-vad/raw/refs/heads/master/src/silero_vad/data/silero_vad.jit"


def load_vad_model():
    """懒加载 Silero VAD 模型，下载到 model_cache 目录。"""
    global _silero_model
    if _silero_model is None:
        os.makedirs(VAD_MODEL_DIR, exist_ok=True)
        model_path = os.path.join(VAD_MODEL_DIR, "silero_vad.jit")

        if not os.path.exists(model_path):
            logger.info("Downloading Silero VAD model to %s", model_path)
            import urllib.request
            urllib.request.urlretrieve(VAD_MODEL_URL, model_path)

        _silero_model = torch.jit.load(model_path, map_location="cpu")
        _silero_model.eval()
        logger.info("Silero VAD model loaded from %s", model_path)
    return _silero_model


def _read_wav_direct(audio_path: str) -> tuple[torch.Tensor, int] | None:
    """直接读取 WAV 文件（16kHz 单声道），不经过 ffmpeg。"""
    import scipy.io.wavfile as wavfile
    try:
        sr, data = wavfile.read(audio_path)
        wav = torch.tensor(data, dtype=torch.float32)
        if wav.dim() > 1:
            wav = wav.mean(dim=1)
        if data.dtype.kind in ('i', 'u'):
            wav = wav / float(2 ** (data.dtype.itemsize * 8 - 1))
        return wav, sr
    except Exception:
        return None


def _read_wav_via_ffmpeg(audio_path: str) -> tuple[torch.Tensor, int]:
    """用 ffmpeg 转换音频为 16kHz 单声道 WAV（处理非 WAV 格式）。"""
    import tempfile
    tmp_wav = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_wav = f.name
        cmd = [
            "ffmpeg", "-y", "-i", audio_path,
            "-f", "wav", "-acodec", "pcm_s16le",
            "-ar", "16000", "-ac", "1",
            "-loglevel", "error",
            tmp_wav,
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=600)
        if result.returncode != 0:
            logger.error("Failed to convert audio via ffmpeg: %s",
                         result.stderr.decode(errors="replace"))
            return torch.tensor([]), 0

        wav, sr = _read_wav_direct(tmp_wav)
        return (wav, sr) if wav is not None else (torch.tensor([]), 0)
    finally:
        if tmp_wav and os.path.exists(tmp_wav):
            os.unlink(tmp_wav)


def _read_audio(audio_path: str) -> tuple[torch.Tensor, int]:
    """读取音频：优先直接读 WAV，失败则走 ffmpeg 转换。"""
    if audio_path.lower().endswith(".wav"):
        result = _read_wav_direct(audio_path)
        if result is not None:
            return result
    return _read_wav_via_ffmpeg(audio_path)


@dataclass
class SpeechSegment:
    """语音片段。"""
    start: float  # 秒
    end: float    # 秒


def detect_speech_segments(audio_path: str, min_silence_ms: int = 500) -> list[SpeechSegment]:
    """
    使用 Silero VAD 检测音频中的语音片段。

    音频已是提取后的 WAV 文件（非原始视频），直接一次性读取即可。
    """
    from silero_vad import get_speech_timestamps

    model = load_vad_model()
    wav, sr = _read_audio(audio_path)
    if wav.numel() == 0:
        return []

    timestamps = get_speech_timestamps(
        wav,
        _silero_model,
        sampling_rate=sr,
        return_seconds=True,
        min_silence_duration_ms=min_silence_ms,
    )
    segments = [SpeechSegment(start=ts["start"], end=ts["end"]) for ts in timestamps]
    total = sum(s.end - s.start for s in segments)
    logger.info("VAD detected %d speech segments (%.1fs total) in %s",
                len(segments), total, os.path.basename(audio_path))
    return segments


def _get_audio_duration(audio_path: str) -> float:
    """获取音频时长（秒）。WAV 文件直接读头部，其他用 ffprobe。"""
    if audio_path.lower().endswith(".wav"):
        import scipy.io.wavfile as wavfile
        try:
            sr, data = wavfile.read(audio_path)
            return len(data) / sr
        except Exception:
            pass
    # 用 ffprobe
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
           "-of", "default=noprint_wrappers=1:nokey=1", audio_path]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    if result.returncode == 0:
        try:
            return float(result.stdout.strip())
        except ValueError:
            pass
    return 0.0


def split_audio_by_vad(
    audio_path: str,
    output_dir: str,
    min_silence_ms: int = 500,
    min_segment_sec: float = 0.5,
) -> list[tuple[str, float, float]]:
    """
    按 VAD 检测的静音边界拆分音频。

    Args:
        audio_path: 源音频文件路径
        output_dir: 存放分块文件的目录
        min_silence_ms: 最小静音持续时间（毫秒）
        min_segment_sec: 最小保留片段长度（秒）

    Returns:
        [(分块路径, 起始秒, 结束秒), ...]
    """
    import hashlib
    import time

    segments = detect_speech_segments(audio_path, min_silence_ms)
    chunks = []

    name_hash = hashlib.sha256(audio_path.encode("utf-8")).hexdigest()[:12]
    prefix = f"vad_{int(time.time())}_{name_hash}"

    for i, seg in enumerate(segments):
        if seg.end - seg.start < min_segment_sec:
            continue

        chunk_path = os.path.join(output_dir, f"{prefix}_chunk_{i:04d}.wav")

        cmd = [
            "ffmpeg", "-y", "-i", audio_path,
            "-ss", str(seg.start),
            "-to", str(seg.end),
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            chunk_path,
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=60)
        if result.returncode == 0:
            chunks.append((chunk_path, seg.start, seg.end))

    logger.info("VAD split %s into %d chunks", os.path.basename(audio_path), len(chunks))
    return chunks
