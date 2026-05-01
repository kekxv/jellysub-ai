"""Qwen3-ASR 引擎 — 支持 0.6B 和 1.7B 模型。"""

import logging
import threading
import time

import torch

from core.asr.base import AsrEngine, group_into_segments

logger = logging.getLogger("uvicorn.error")


class Qwen3AsrEngine(AsrEngine):
    """Qwen3-ASR 引擎，支持 0.6B 和 1.7B 模型。"""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-ASR-0.6B",
        device: str | None = None,
    ):
        self.model_name = model_name
        self._device = device
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return self._model

        from env_config import MODEL_SOURCE
        from pathlib import Path

        cache_dir = "./model_cache"
        forced_aligner_name = "Qwen/Qwen3-ForcedAligner-0.6B"

        def _cached_path(name: str) -> str | None:
            """检查 model_cache 下是否已有缓存，有则返回本地路径，跳过远程检查。"""
            # HF 缓存结构: model_cache/models--{org}--{model}/snapshots/{commit}/
            p = Path(cache_dir) / "models--" / name.replace("/", "--") / "snapshots"
            if p.is_dir():
                subs = [d for d in p.iterdir() if d.is_dir()]
                if subs:
                    return str(sorted(subs)[-1])
            # ModelScope 缓存结构: model_cache/hub/models/{org}--{model}/
            p2 = Path(cache_dir) / "hub" / "models" / name.replace("/", "--")
            if p2.is_dir():
                return str(p2)
            return None

        if MODEL_SOURCE == "modelscope":
            try:
                from modelscope.hub.snapshot_download import snapshot_download
                cached = _cached_path(self.model_name)
                local_dir = cached or snapshot_download(self.model_name, cache_dir=cache_dir)
                self.model_name = local_dir
                cached_fa = _cached_path(forced_aligner_name)
                forced_aligner_name = cached_fa or snapshot_download(
                    forced_aligner_name, cache_dir=cache_dir
                )
            except ModuleNotFoundError:
                logger.error("MODEL_SOURCE=modelscope 但 modelscope 未安装")
                raise
        else:
            try:
                from huggingface_hub import snapshot_download
                cached = _cached_path(self.model_name)
                if cached:
                    local_dir = cached
                else:
                    local_dir = snapshot_download(self.model_name, cache_dir=cache_dir)
                self.model_name = local_dir
                cached_fa = _cached_path(forced_aligner_name)
                if cached_fa:
                    forced_aligner_name = cached_fa
                else:
                    forced_aligner_name = snapshot_download(
                        forced_aligner_name, cache_dir=cache_dir
                    )
            except ModuleNotFoundError:
                logger.warning("huggingface_hub 未安装，使用默认缓存路径")

        from qwen_asr import Qwen3ASRModel

        if torch.cuda.is_available():
            device = self._device or "cuda:0"
            dtype = torch.bfloat16
        elif torch.backends.mps.is_available():
            device = self._device or "mps"
            dtype = torch.bfloat16
        else:
            device = self._device or "cpu"
            dtype = torch.float16

        device_map = device if device == "cpu" else {"": device}

        logger.info("Loading Qwen3-ASR model %s on %s (dtype=%s)", self.model_name, device, dtype)
        self._model = Qwen3ASRModel.from_pretrained(
            self.model_name,
            dtype=dtype,
            device_map=device_map,
            max_inference_batch_size=32,
            max_new_tokens=4096,
            forced_aligner=forced_aligner_name,
            forced_aligner_kwargs=dict(dtype=dtype, device_map=device_map),
        )
        return self._model

    def release(self):
        if self._model is not None:
            del self._model
            self._model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()
            logger.info("Qwen3-ASR model released")

    def transcribe(self, audio_path: str, language: str = "auto") -> tuple[list[dict], str]:
        model = self._load_model()
        logger.info("Running Qwen3-ASR on: %s (lang=%s)", audio_path, language)

        done = threading.Event()
        start = time.time()
        self.log_progress(audio_path, done, start)

        # 短代码 → Qwen3-ASR 全名映射
        _LANG_MAP = {
            "zh": "Chinese", "en": "English", "yue": "Cantonese",
            "ja": "Japanese", "ko": "Korean", "ar": "Arabic",
            "de": "German", "fr": "French", "es": "Spanish",
            "pt": "Portuguese", "id": "Indonesian", "it": "Italian",
            "ru": "Russian", "th": "Thai", "vi": "Vietnamese",
            "tr": "Turkish", "hi": "Hindi", "ms": "Malay",
        }
        qwen_lang = _LANG_MAP.get(language, language) if language != "auto" else None

        results = model.transcribe(
            audio=audio_path,
            language=qwen_lang,
            return_time_stamps=True,
        )

        done.set()
        elapsed = time.time() - start
        from core.audio import get_audio_duration
        duration = get_audio_duration(audio_path)
        if duration > 60:
            logger.info("ASR completed in %.1f min", elapsed / 60)

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

        segments = group_into_segments(result.text, time_stamps)
        logger.info("ASR produced %d segments", len(segments))
        for seg in segments[:3]:
            logger.info("ASR [%ss-%ss] %s", seg["start"], seg["end"], seg["text"])
        if len(segments) > 3:
            logger.info("ASR ... and %d more segments", len(segments) - 3)
        return segments, detected_lang
