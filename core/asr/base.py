"""ASR 语音识别 — 基础类型 + 共享工具。"""

import logging
import re
import threading
import time

logger = logging.getLogger("uvicorn.error")


class AsrEngine:
    """ASR 引擎基类。"""

    def transcribe(self, audio_path: str, language: str = "auto") -> tuple[list[dict], str]:
        raise NotImplementedError

    def need_vad(self) -> bool:
        """是否需要外部 VAD 分块处理。返回 True 的引擎会在 run_asr 中被 vad_wrapper 包裹。
        SenseVoice 有内置 merge_vad，返回 False。
        """
        return True

    def log_progress(self, audio_path: str, done: threading.Event, start: float):
        """长音频后台进度日志，子类在 transcribe 中可选调用。"""
        from core.audio import get_audio_duration
        duration = get_audio_duration(audio_path)
        if duration <= 60:
            return

        def _logger():
            while not done.is_set():
                elapsed = time.time() - start
                logger.info("ASR running... %.1f min elapsed (audio: %.1f min)",
                            elapsed / 60, duration / 60)
                done.wait(60)

        t = threading.Thread(target=_logger, daemon=True)
        t.start()
        return t


# =========================================================================== #
#  共享工具函数（字幕片段分割）
# =========================================================================== #

_SENTENCE_PUNCT = re.compile(r"[。！？.!?]+")
_PHRASE_PUNCT = re.compile(r"[;；，,、\n]+")


def join_words(words: list) -> str:
    """拼接词级 token，英文加空格，中文直接拼接。"""
    text = "".join(w.text for w in words)
    latin_ratio = sum(1 for c in text if c.isascii() and c.isalpha()) / max(len(text), 1)
    if latin_ratio > 0.5:
        return " ".join(w.text for w in words).strip()
    return text.strip()


def add_silence_gaps(segments: list[dict], min_gap: float = 0.2) -> list[dict]:
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


def group_into_segments(transcript: str, time_stamps: list) -> list[dict]:
    """将词级时间戳分组为字幕片段（Qwen3-ASR 专用）。"""
    if not time_stamps:
        return []

    segments = []
    current_words = []

    for i, ts in enumerate(time_stamps):
        current_words.append(ts)

        if _SENTENCE_PUNCT.search(ts.text):
            text = join_words(current_words)
            if text:
                seg_start = round(current_words[0].start_time, 3)
                seg_end = round(ts.end_time, 3)
                if seg_end - seg_start >= 0.3:
                    segments.append({"start": seg_start, "end": seg_end, "text": text})
            current_words = []
            continue

        if len(current_words) >= 2:
            gap = ts.start_time - current_words[-2].end_time
            if gap > 0.4:
                text = join_words(current_words[:-1])
                if text:
                    seg_start = round(current_words[0].start_time, 3)
                    seg_end = round(current_words[-2].end_time, 3)
                    if seg_end - seg_start >= 0.3:
                        segments.append({"start": seg_start, "end": seg_end, "text": text})
                current_words = [ts]
                continue

        if _PHRASE_PUNCT.search(ts.text) and len(current_words) >= 3:
            text_so_far = join_words(current_words)
            latin_ratio = sum(1 for c in text_so_far if c.isascii() and c.isalpha()) / max(len(text_so_far), 1)
            max_len = 40 if latin_ratio > 0.5 else 50
            if len(text_so_far) >= max_len:
                text = text_so_far
                if text:
                    seg_start = round(current_words[0].start_time, 3)
                    seg_end = round(ts.end_time, 3)
                    if seg_end - seg_start >= 0.3:
                        segments.append({"start": seg_start, "end": seg_end, "text": text})
                current_words = []
                continue

    if current_words:
        text = join_words(current_words)
        if text:
            seg_start = round(current_words[0].start_time, 3)
            seg_end = round(current_words[-1].end_time, 3)
            if seg_end - seg_start >= 0.3:
                segments.append({"start": seg_start, "end": seg_end, "text": text})

    return add_silence_gaps(segments)
