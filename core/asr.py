"""Qwen3-ASR 语音识别 — 支持本地模型和在线 API 两种模式。"""

import json
import logging
import re
import time

import torch

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

# 全局单例，避免重复加载模型
_model = None
_model_last_access: float = 0.0


def _touch_model():
    """记录模型最近访问时间。"""
    global _model_last_access
    _model_last_access = time.time()


def _load_model(model_name: str, device: str | None = None):
    """懒加载 ASR 模型。"""
    global _model, _model_last_access
    if _model is not None:
        _touch_model()
        return _model

    from env_config import MODEL_SOURCE
    forced_aligner_name = "Qwen/Qwen3-ForcedAligner-0.6B"
    if MODEL_SOURCE == "modelscope":
        try:
            from modelscope.hub.snapshot_download import snapshot_download
            local_dir = snapshot_download(model_name, cache_dir="./model_cache")
            model_name = local_dir
            forced_aligner_name = snapshot_download(
                "Qwen/Qwen3-ForcedAligner-0.6B", cache_dir="./model_cache"
            )
        except ModuleNotFoundError:
            logger.error("MODEL_SOURCE=modelscope 但 modelscope 未安装，请运行: pip install modelscope")
            raise

    from qwen_asr import Qwen3ASRModel

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float16
    device_map = device if device == "cpu" else {"": device}

    logger.info("Loading ASR model %s on %s (dtype=%s)", model_name, device, dtype)
    _model = Qwen3ASRModel.from_pretrained(
        model_name,
        dtype=dtype,
        device_map=device_map,
        max_inference_batch_size=32,
        max_new_tokens=4096,
        forced_aligner=forced_aligner_name,
        forced_aligner_kwargs=dict(dtype=dtype, device_map=device_map),
    )
    _touch_model()
    return _model


def _release_model():
    """释放 ASR 模型，释放显存/内存。"""
    global _model, _model_last_access
    if _model is not None:
        del _model
        _model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _model_last_access = 0.0
        logger.info("ASR model released")


def _check_model_idle(timeout: int):
    """检查模型是否空闲超时，如果是则释放。"""
    global _model_last_access
    if _model is not None and timeout > 0:
        elapsed = time.time() - _model_last_access
        if elapsed > timeout:
            _release_model()


# 句子级标点（句号/问号/叹号）
_SENTENCE_PUNCT = re.compile(r"[。！？.!?]+")

# 短语级标点（逗号/分号）
_PHRASE_PUNCT = re.compile(r"[;；，,、\n]+")


def _join_words(words: list) -> str:
    """拼接词级 token，英文加空格，中文直接拼接。"""
    text = "".join(w.text for w in words)
    latin_ratio = sum(1 for c in text if c.isascii() and c.isalpha()) / max(len(text), 1)
    if latin_ratio > 0.5:
        return " ".join(w.text for w in words).strip()
    return text.strip()


def _add_silence_gaps(segments: list[dict], min_gap: float = 0.2) -> list[dict]:
    """在相邻字幕段之间留出静音间隙，给观众阅读时间。"""
    if len(segments) < 2:
        return segments

    for i in range(len(segments) - 1):
        current_end = segments[i]["end"]
        next_start = segments[i + 1]["start"]
        if current_end >= next_start:
            new_end = next_start - min_gap
            if new_end > segments[i]["start"]:
                segments[i]["end"] = round(new_end, 3)

    return segments


def _group_into_segments(transcript: str, time_stamps: list) -> list[dict]:
    """将词级时间戳分组为字幕片段。"""
    if not time_stamps:
        return []

    segments = []
    current_words = []

    for i, ts in enumerate(time_stamps):
        current_words.append(ts)

        if _SENTENCE_PUNCT.search(ts.text):
            text = _join_words(current_words)
            if text:
                seg_start = round(current_words[0].start_time, 3)
                seg_end = round(ts.end_time, 3)
                if seg_end - seg_start >= 0.3:
                    segments.append({
                        "start": seg_start,
                        "end": seg_end,
                        "text": text,
                    })
            current_words = []
            continue

        if len(current_words) >= 2:
            gap = ts.start_time - current_words[-2].end_time
            if gap > 0.4:
                text = _join_words(current_words[:-1])
                if text:
                    seg_start = round(current_words[0].start_time, 3)
                    seg_end = round(current_words[-2].end_time, 3)
                    if seg_end - seg_start >= 0.3:
                        segments.append({
                            "start": seg_start,
                            "end": seg_end,
                            "text": text,
                        })
                current_words = [ts]
                continue

        if _PHRASE_PUNCT.search(ts.text) and len(current_words) >= 3:
            text_so_far = _join_words(current_words)
            latin_ratio = sum(1 for c in text_so_far if c.isascii() and c.isalpha()) / max(len(text_so_far), 1)
            max_len = 40 if latin_ratio > 0.5 else 50
            if len(text_so_far) >= max_len:
                text = text_so_far
                if text:
                    seg_start = round(current_words[0].start_time, 3)
                    seg_end = round(ts.end_time, 3)
                    if seg_end - seg_start >= 0.3:
                        segments.append({
                            "start": seg_start,
                            "end": seg_end,
                            "text": text,
                        })
                current_words = []
                continue

    if current_words:
        text = _join_words(current_words)
        if text:
            seg_start = round(current_words[0].start_time, 3)
            seg_end = round(current_words[-1].end_time, 3)
            if seg_end - seg_start >= 0.3:
                segments.append({
                    "start": seg_start,
                    "end": seg_end,
                    "text": text,
                })

    segments = _add_silence_gaps(segments)
    return segments


async def run_asr_online(
    audio_path: str,
    api_url: str,
    api_key: str,
    model: str,
) -> list[dict]:
    """使用在线 API 进行语音识别（OpenAI 兼容 Audio Transcription）。"""
    client = AsyncOpenAI(base_url=api_url, api_key=api_key)
    logger.info("Running online ASR on: %s (model=%s, url=%s)", audio_path, model, api_url)

    with open(audio_path, "rb") as f:
        resp = await client.audio.transcriptions.create(
            model=model,
            file=f,
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        )

    segments = []
    for seg in getattr(resp, "segments", []):
        segments.append({
            "start": round(seg["start"], 3),
            "end": round(seg["end"], 3),
            "text": seg["text"].strip(),
        })

    if not segments and getattr(resp, "text", ""):
        segments = [{"start": 0.0, "end": 0.0, "text": resp.text.strip()}]

    logger.info("Online ASR produced %d segments", len(segments))
    return segments


def run_asr(
    audio_path: str,
    mode: str = "local",
    model_name: str = "Qwen/Qwen3-ASR-0.6B",
    api_url: str = "",
    api_key: str = "",
    model_online: str = "",
) -> tuple[list[dict], str]:
    """
    对音频进行语音识别，返回 (片段列表, 检测到的语言代码)。
    mode="local": 使用本地 qwen-asr 模型
    mode="online": 使用在线 Audio Transcription API
    """
    if mode == "online":
        import asyncio
        segments = asyncio.new_event_loop().run_until_complete(
            run_asr_online(audio_path, api_url, api_key, model_online)
        )
        return segments, ""

    # 本地模式
    model = _load_model(model_name)
    logger.info("Running ASR on: %s", audio_path)

    results = model.transcribe(
        audio=audio_path,
        language=None,
        return_time_stamps=True,
    )

    if not results:
        logger.warning("ASR returned no results for %s", audio_path)
        return [], ""

    result = results[0]
    detected_lang = getattr(result, "language", "")
    logger.info("ASR detected language: %s", detected_lang)

    time_stamps = getattr(result, "time_stamps", None)
    if not time_stamps:
        logger.warning("No timestamps from ASR, using full text as single segment")
        return [{"start": 0.0, "end": 0.0, "text": result.text.strip()}], detected_lang

    segments = _group_into_segments(result.text, time_stamps)
    logger.info("ASR produced %d segments", len(segments))
    for seg in segments[:3]:
        logger.info("ASR [%ss-%ss] %s", seg["start"], seg["end"], seg["text"])
    if len(segments) > 3:
        logger.info("ASR ... and %d more segments", len(segments) - 3)
    return segments, detected_lang
