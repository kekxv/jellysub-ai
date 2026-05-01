"""SenseVoice 引擎 — 基于 FunASR SenseVoiceSmall。"""

import logging
import re
import threading
import time

import torch

from core.asr.base import AsrEngine, _SENTENCE_PUNCT, add_silence_gaps, join_words

logger = logging.getLogger("uvicorn.error")

# SenseVoice 文本标签正则
_SENSEVOICE_TAGS = re.compile(r"<\|[^>]*\|>")

# 日语特有语尾助词，作为软断句边界
_JP_SENTENCE_ENDINGS = {"よ", "ね", "な", "さ", "か", "わ", "ぞ", "ぜ",
                         "でしょう", "でしょ", "ですよね", "よね", "かな",
                         "ます", "です", "した", "だった", "ている", "ってる"}

# 英文常见缩约词还原（SenseVoice 输出 `didn` `t` 时还原为 `didn't`）
_APOSTROPHE_RE = re.compile(
    r"\b(don|do|did|does|wo|ca|should|would|could|is|are|was|were|"
    r"has|have|had|I|he|she|it|they|you|we|that|who|there|what)"
    r"(nt|n|t|m|ve|ll|d|re|s)\b"
)

def _fix_apostrophes(text: str) -> str:
    """还原 SenseVoice 丢失的撇号：didnt → didn't, im → I'm。"""
    def _replace(m):
        base, suffix = m.group(1), m.group(2)
        # 精确映射
        if base.lower() in ("don", "do", "did", "does", "wo", "ca",
                            "should", "would", "could", "is", "are", "was", "were",
                            "has", "have", "had") and suffix == "nt":
            return base + "n't"
        if base == "I" and suffix == "m":
            return "I'm"
        if base == "I" and suffix == "ve":
            return "I've"
        if base == "I" and suffix == "ll":
            return "I'll"
        if base == "I" and suffix == "d":
            return "I'd"
        return m.group(0)  # 不匹配，保留原文
    return _APOSTROPHE_RE.sub(_replace, text)


class SenseVoiceAsrEngine(AsrEngine):
    """SenseVoiceSmall 引擎，基于 FunASR。"""

    def __init__(
        self,
        model_name: str = "iic/SenseVoiceSmall",
        device: str | None = None,
    ):
        self.model_name = model_name
        self._device = device
        self._pipeline = None

    def _load_model(self):
        if self._pipeline is not None:
            return self._pipeline

        from env_config import MODEL_SOURCE
        
        # 兼容性处理：HuggingFace 和 ModelScope 命名转换
        if MODEL_SOURCE == "modelscope":
            if "FunAudioLLM/SenseVoiceSmall" in self.model_name:
                self.model_name = "iic/SenseVoiceSmall"
        else:
            if "iic/SenseVoiceSmall" in self.model_name:
                self.model_name = "FunAudioLLM/SenseVoiceSmall"

        cache_dir = "./model_cache"
        from pathlib import Path

        def _cached_path(name: str) -> str | None:
            p = Path(cache_dir) / "models--" / name.replace("/", "--") / "snapshots"
            if p.is_dir():
                subs = [d for d in p.iterdir() if d.is_dir()]
                if subs:
                    return str(sorted(subs)[-1])
            p2 = Path(cache_dir) / "hub" / "models" / name.replace("/", "--")
            if p2.is_dir():
                return str(p2)
            return None

        if MODEL_SOURCE == "modelscope":
            try:
                from modelscope.hub.snapshot_download import snapshot_download
                cached = _cached_path(self.model_name)
                self.model_name = cached or snapshot_download(self.model_name, cache_dir=cache_dir)
            except ModuleNotFoundError:
                logger.error("MODEL_SOURCE=modelscope 但 modelscope 未安装")
                raise
        else:
            try:
                from huggingface_hub import snapshot_download
                cached = _cached_path(self.model_name)
                self.model_name = cached or snapshot_download(self.model_name, cache_dir=cache_dir)
            except ModuleNotFoundError:
                logger.warning("huggingface_hub 未安装，使用默认缓存路径")

        from funasr import AutoModel

        if torch.cuda.is_available():
            device = self._device or "cuda:0"
        elif torch.backends.mps.is_available():
            device = self._device or "mps"
        else:
            device = self._device or "cpu"
        logger.info("Loading SenseVoice model %s on %s", self.model_name, device)
        self._pipeline = AutoModel(
            model=self.model_name,
            trust_remote_code=True,
            device=device,
            disable_pbar=True,
            disable_update=True,
        )
        return self._pipeline

    def release(self):
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()
            logger.info("SenseVoice model released")

    def need_vad(self) -> bool:
        return True

    def transcribe(self, audio_path: str, language: str = "auto") -> tuple[list[dict], str]:
        pipeline = self._load_model()
        logger.info("Running SenseVoice ASR on: %s (lang=%s)", audio_path, language)

        done = threading.Event()
        start = time.time()
        self.log_progress(audio_path, done, start)

        # 构造 generate 参数
        gen_kwargs = {
            "input": audio_path,
            "batch_size_s": 60,
            "merge_vad": True,
            "merge_length_s": 15,
            "use_itn": True,
            "output_timestamp": True,
        }
        if language and language != "auto":
            gen_kwargs["language"] = language

        result = pipeline.generate(**gen_kwargs)

        done.set()
        elapsed = time.time() - start
        from core.audio import get_audio_duration
        duration = get_audio_duration(audio_path)
        if duration > 60:
            logger.info("ASR completed in %.1f min", elapsed / 60)

        if not result:
            logger.warning("SenseVoice returned no results for %s", audio_path)
            return [], ""

        # SenseVoice 返回 list[dict]，每项包含 "text", "timestamp" 等字段
        rec = result[0] if isinstance(result, list) else result
        raw_lang = rec.get("language", "")
        raw_text = rec.get("text", "")

        # 从文本中提取语言标签（如 <|ja|>, <|en|> 等）
        detected_lang = ""
        if isinstance(raw_lang, str) and raw_lang:
            detected_lang = raw_lang
        elif isinstance(raw_lang, (list, set)):
            for tag in raw_lang:
                if tag.startswith("<|") and tag.endswith("|>"):
                    code = tag[2:-2]
                    if code in ("zh", "en", "ja", "ko", "yue", "fr", "de", "es", "ru"):
                        detected_lang = code
                        break

        if not detected_lang:
            lang_tags = _SENSEVOICE_TAGS.findall(raw_text)
            for tag in lang_tags:
                code = tag[2:-2]  # 去掉 <| |>
                if code in ("zh", "en", "ja", "ko", "yue", "fr", "de", "es", "ru"):
                    detected_lang = code
                    break

        text = _SENSEVOICE_TAGS.sub("", raw_text).strip()

        # 尝试获取词级时间戳
        timestamps = rec.get("timestamp", [])
        words = rec.get("words", [])

        if timestamps:
            logger.info("SenseVoice raw timestamp sample: %s (audio duration: %.1fs)",
                        repr(timestamps[:3]), duration)
        if words:
            logger.info("SenseVoice raw words sample: %s", repr(words[:5]))

        if timestamps and words:
            segments = self._word_timestamps_to_segments(timestamps, words)
        elif timestamps:
            # 只有时间戳没有 words（句子级 [[start, end], ...]）
            segments = self._timestamps_to_segments(timestamps, text)
        else:
            logger.warning("No timestamps from SenseVoice, using full text as single segment")
            segments = [{"start": 0.0, "end": 0.0, "text": text}]

        logger.info("SenseVoice ASR produced %d segments", len(segments))
        for seg in segments[:3]:
            logger.info("ASR [%ss-%ss] %s", seg["start"], seg["end"], seg["text"])
        if len(segments) > 3:
            logger.info("ASR ... and %d more segments", len(segments) - 3)
        return segments, detected_lang

    @staticmethod
    def _frame_to_sec(frame: float) -> float:
        """将时间戳帧单位转为秒。SenseVoice 词级时间戳单位为毫秒。"""
        return round(frame / 1000, 3)

    @staticmethod
    def _join_words(word_list: list[str]) -> str:
        """拼接词级 token，英文加空格，中文直接拼接。
        同时合并被拆分的撇号词：didn ' t → didn't。
        """
        merged = SenseVoiceAsrEngine._merge_apostrophe_words(word_list)
        text = "".join(merged)
        latin_ratio = sum(1 for c in text if c.isascii() and c.isalpha()) / max(len(text), 1)
        if latin_ratio > 0.5:
            result = " ".join(merged).strip()
            result = _fix_apostrophes(result)
            return result
        return text.strip()

    @staticmethod
    def _merge_apostrophe_words(word_list: list[str]) -> list[str]:
        """合并被拆分的撇号词：didn ' t → didn't, can ' t → can't。"""
        if len(word_list) < 3:
            return word_list
        APOSTROPHES = {"'", "'", "'", "'", "'"}
        result = []
        i = 0
        while i < len(word_list):
            mid = word_list[i + 1].strip() if i + 1 < len(word_list) else ""
            nxt = word_list[i + 2].strip() if i + 2 < len(word_list) else ""
            if mid in APOSTROPHES and nxt:
                result.append(word_list[i].strip() + word_list[i + 1].strip() + nxt)
                i += 3
            else:
                result.append(word_list[i].strip())
                i += 1
        return result

    @staticmethod
    def _is_jp_ending(word: str) -> bool:
        """判断文本是否以日语语尾助词结尾（用于断句）。"""
        if not word:
            return False
        for ending in _JP_SENTENCE_ENDINGS:
            if word.endswith(ending):
                return True
        return False

    @staticmethod
    def _word_timestamps_to_segments(timestamps: list, words: list) -> list[dict]:
        """将词级时间戳分组为字幕片段。

        timestamps: [[start_frame, end_frame], ...] (60Hz 单位)
        words: 对应每个时间戳的单词/词
        """
        if len(timestamps) != len(words):
            logger.warning("Timestamp/word count mismatch: %d vs %d", len(timestamps), len(words))
            return []

        # 按标点分组
        segments = []
        group_starts = []
        group_ends = []
        group_words = []

        for i, (ts, word) in enumerate(zip(timestamps, words)):
            clean_word = _SENSEVOICE_TAGS.sub("", str(word)).strip()
            start_s = SenseVoiceAsrEngine._frame_to_sec(ts[0])
            end_s = SenseVoiceAsrEngine._frame_to_sec(ts[1])

            group_starts.append(start_s)
            group_ends.append(end_s)
            group_words.append(clean_word)

            # 判断是否到达句子结尾
            # 英文/中文/日语：以 . ! ? 。 ！ ？ 作为句子边界，逗号不断句
            # 日语：以语尾助词作为补充断句依据
            accumulated = SenseVoiceAsrEngine._join_words(group_words)
            is_jp = not any(c.isascii() and c.isalpha() for c in accumulated[:3])
            # 句子边界：. ! ? 。 ！ ？
            is_sentence_end = clean_word in (".", "!", "?", "。", "！", "？")
            if is_jp:
                is_sentence_end = is_sentence_end or SenseVoiceAsrEngine._is_jp_ending(accumulated)
            # 至少积累2个词才断句
            if is_sentence_end and len(group_words) < 2:
                is_sentence_end = False
            if is_sentence_end:
                seg_text = SenseVoiceAsrEngine._join_words(group_words)
                if seg_text and group_starts:
                    start_s = round(group_starts[0], 3)
                    end_s = round(group_ends[-1], 3)
                    if end_s - start_s >= 0.3:
                        # 纯标点片段合并到前一段
                        if segments and seg_text.strip() in (".", "!", "?", "，", "；", "。"):
                            segments[-1]["text"] += seg_text.strip()
                            segments[-1]["end"] = end_s
                        else:
                            segments.append({"start": start_s, "end": end_s, "text": seg_text})
                group_starts.clear()
                group_ends.clear()
                group_words.clear()
                continue

            # 无标点但超长：安全截断（尽量在逗号/连词后切分）
            text_so_far = SenseVoiceAsrEngine._join_words(group_words)
            latin_ratio = sum(1 for c in text_so_far if c.isascii() and c.isalpha()) / max(len(text_so_far), 1)
            if latin_ratio > 0.5:
                max_len = 120  # 英文（放宽，等待句号到来）
            else:
                max_len = 60  # 中日韩
            if len(text_so_far) >= max_len:
                seg_text = SenseVoiceAsrEngine._join_words(group_words)
                if seg_text and group_starts:
                    start_s = round(group_starts[0], 3)
                    end_s = round(group_ends[-1], 3)
                    if end_s - start_s >= 0.3:
                        segments.append({"start": start_s, "end": end_s, "text": seg_text})
                group_starts.clear()
                group_ends.clear()
                group_words.clear()

        if group_words:
            seg_text = SenseVoiceAsrEngine._join_words(group_words)
            if seg_text and group_starts:
                start_s = round(group_starts[0], 3)
                end_s = round(group_ends[-1], 3)
                if end_s - start_s >= 0.3:
                    # 纯标点合并到前一段
                    if segments and seg_text.strip() in (".", "!", "?", "，", "；", "。"):
                        segments[-1]["text"] += seg_text.strip()
                        segments[-1]["end"] = end_s
                    else:
                        segments.append({"start": start_s, "end": end_s, "text": seg_text})

        return add_silence_gaps(segments)

    @staticmethod
    def _timestamps_to_segments(timestamps: list, text: str = "") -> list[dict]:
        """将句子级时间戳转为字幕片段（备用）。

        SenseVoice 的 timestamp 格式为 [[start, end], ...] 句子级时间戳对（60Hz 单位）。
        """
        if not timestamps or not text:
            return []

        # 按标点符号拆分文本为句子
        sentences = _SENTENCE_PUNCT.split(text)
        sentences = [s.strip() for s in sentences if s.strip()]

        segments = []
        for i, sent in enumerate(sentences):
            if i >= len(timestamps):
                if segments:
                    segments[-1]["text"] += " " + sent
                continue

            ts = timestamps[i]
            if len(ts) < 2:
                continue

            start_s = SenseVoiceAsrEngine._frame_to_sec(ts[0])
            end_s = SenseVoiceAsrEngine._frame_to_sec(ts[1])

            if end_s - start_s < 0.3:
                continue

            # 纯标点句子合并到前一段
            if sent.strip() in (".", "!", "?", "，", "；", "。"):
                if segments:
                    segments[-1]["text"] += sent.strip()
                    segments[-1]["end"] = end_s
                continue

            segments.append({"start": start_s, "end": end_s, "text": sent})

        return add_silence_gaps(segments)
