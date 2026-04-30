"""VAD 音频分块 ASR 封装器。

对长音频先用 Silero VAD 检测语音片段，逐段送入 ASR 引擎识别，
最后合并结果并还原时间戳。避免长音频导致 AI 死循环。
"""

import hashlib
import logging
import os
import tempfile
import time

from core.asr.base import AsrEngine, add_silence_gaps

logger = logging.getLogger("uvicorn.error")


def transcribe_with_vad(
    engine: AsrEngine,
    audio_path: str,
    min_silence_ms: int = 500,
    language: str = "auto",
) -> tuple[list[dict], str]:
    """
    使用 VAD 分块处理长音频并识别。
    """
    from core.vad import detect_speech_segments

    # 检测语音片段
    speech_segments = detect_speech_segments(audio_path, min_silence_ms)

    if not speech_segments:
        logger.info("VAD found no speech in %s, skipping ASR",
                     os.path.basename(audio_path))
        return [], ""

    # 计算总语音时长
    total_speech = sum(s.end - s.start for s in speech_segments)

    if len(speech_segments) == 1 or total_speech < 60:
        # 短音频或单一段，直接处理
        logger.info("VAD: total speech %.1fs < 60s, processing as single chunk (no split)", total_speech)
        segments, detected_lang = engine.transcribe(audio_path, language=language)
        # SenseVoice 不返回时间戳（start == end == 0.0），用 VAD 片段时间来分配
        return _fix_timestamps(segments, speech_segments, audio_path), detected_lang

    logger.info("VAD splitting audio into %d chunks (total speech: %.1f min)",
                len(speech_segments), total_speech / 60)

    all_segments: list[dict] = []
    detected_lang = ""
    current_language = language

    # 用时间戳+路径SHA256作为前缀，避免中文/特殊字符
    name_hash = hashlib.sha256(audio_path.encode("utf-8")).hexdigest()[:12]
    prefix = f"vad_{int(time.time())}_{name_hash}"

    with tempfile.TemporaryDirectory() as tmp_dir:
        for i, seg in enumerate(speech_segments):
            # 提取分块
            chunk_path = os.path.join(tmp_dir, f"{prefix}_chunk_{i:04d}.wav")
            import subprocess
            cmd = [
                "ffmpeg", "-y", "-i", audio_path,
                "-ss", str(seg.start),
                "-to", str(seg.end),
                "-acodec", "pcm_s16le",
                "-ar", "16000",
                "-ac", "1",
                "-loglevel", "error",
                chunk_path,
            ]
            result = subprocess.run(cmd, capture_output=True, timeout=60)
            if result.returncode != 0:
                logger.warning("Failed to extract chunk %d", i)
                continue

            # 识别该分块
            from core.utils import check_memory_limit
            check_memory_limit()
            chunk_segments, chunk_lang = engine.transcribe(chunk_path, language=current_language)

            # 关键 1：语言锁定。如果原始是 auto，识别出第一个有效语言后，后续分块锁定该语言。
            # 防止长音频中某一段英文被识别成日文或中文，保持整片一致性。
            if language == "auto" and chunk_lang and chunk_lang != "auto":
                if current_language == "auto":
                    current_language = chunk_lang
                    logger.info("VAD: Language locked to '%s' after chunk %d", current_language, i)
                if not detected_lang:
                    detected_lang = chunk_lang

            # 每个 chunk 只有一个片段时，直接用 VAD 边界
            if len(chunk_segments) == 1 and chunk_segments[0]["start"] == 0.0:
                chunk_segments[0]["start"] = round(seg.start, 3)
                chunk_segments[0]["end"] = round(seg.end, 3)

            # 还原时间戳到原始音频时间轴
            for s in chunk_segments:
                s["start"] = round(s["start"] + seg.start, 3)
                s["end"] = round(s["end"] + seg.start, 3)
                all_segments.append(s)

            logger.info("VAD chunk %d/%d: %.1f-%.1f min → %d segments (lang=%s)",
                        i + 1, len(speech_segments),
                        seg.start / 60, seg.end / 60,
                        len(chunk_segments), chunk_lang)

            # 关键 2：激进释放内存。
            # 除了 torch 的缓存，还显式触发 python 的 gc。
            import torch
            import gc
            del chunk_segments
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()
            gc.collect()

    # 按时间排序
    all_segments.sort(key=lambda x: x["start"])

    # 添加静音间隙
    all_segments = add_silence_gaps(all_segments)

    logger.info("VAD chunking produced %d total segments", len(all_segments))
    return all_segments, detected_lang


def _fix_timestamps(
    segments: list[dict],
    speech_segments: list,
    audio_path: str,
) -> list[dict]:
    """用 VAD 语音片段时间戳修复 SenseVoice 返回的无时间戳结果。

    SenseVoice 只返回完整文本，没有时间戳。将文本按标点拆分为句子，
    然后按顺序分配到 VAD 检测到的语音片段中。
    """
    if not segments or not speech_segments:
        return segments

    # 检查是否需要修复（所有 start == end）
    if any(s["start"] != s["end"] for s in segments):
        return segments  # 已有时间戳，不需要修复

    text = " ".join(s["text"] for s in segments).strip()
    if not text:
        return segments

    # 按标点符号拆分文本为句子
    import re
    sentence_end = re.compile(r"[。！？.!?]+")
    sentences = []
    current = ""
    for char in text:
        current += char
        if sentence_end.search(char) and current.strip():
            sentences.append(current.strip())
            current = ""
    if current.strip():
        sentences.append(current.strip())

    if not sentences:
        return segments

    # 将句子分配到 VAD 片段
    result = []
    vad_idx = 0
    sent_idx = 0

    while vad_idx < len(speech_segments) and sent_idx < len(sentences):
        vad_seg = speech_segments[vad_idx]
        if vad_seg.end - vad_seg.start < 0.3:
            vad_idx += 1
            continue

        # 分配 1-2 个句子到当前 VAD 片段
        group = []
        group.append(sentences[sent_idx])
        sent_idx += 1

        # 如果还有句子，尽量分配一个
        if sent_idx < len(sentences) and vad_idx + 1 < len(speech_segments):
            group.append(sentences[sent_idx])
            sent_idx += 1

        seg_text = " ".join(group)
        result.append({
            "start": round(vad_seg.start, 3),
            "end": round(vad_seg.end, 3),
            "text": seg_text,
        })
        vad_idx += 1

    # 剩余句子合并到最后一段
    if sent_idx < len(sentences) and result:
        result[-1]["text"] += " " + " ".join(sentences[sent_idx:])
    elif sent_idx < len(sentences):
        # 没有足够 VAD 片段，用最后一个片段延长
        if result:
            result[-1]["text"] += " " + " ".join(sentences[sent_idx:])

    return result if result else segments
