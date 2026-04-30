"""OpenAI 兼容 Audio Transcription API 引擎。"""

import asyncio
import logging

from openai import AsyncOpenAI

from core.asr.base import AsrEngine

logger = logging.getLogger("uvicorn.error")


class OpenaiAsrEngine(AsrEngine):
    """OpenAI 兼容 Audio Transcription API 引擎。"""

    def __init__(self, api_url: str, api_key: str, model: str):
        self.api_url = api_url
        self.api_key = api_key
        self.model = model

    def transcribe(self, audio_path: str, language: str = "auto") -> tuple[list[dict], str]:
        logger.info("Running online ASR on: %s (model=%s, url=%s, lang=%s)", 
                    audio_path, self.model, self.api_url, language)

        async def _do():
            client = AsyncOpenAI(base_url=self.api_url, api_key=self.api_key)
            with open(audio_path, "rb") as f:
                # 构造 transcription 参数
                kwargs = {
                    "model": self.model,
                    "file": f,
                    "response_format": "verbose_json",
                    "timestamp_granularities": ["segment"],
                }
                if language and language != "auto":
                    kwargs["language"] = language

                resp = await client.audio.transcriptions.create(**kwargs)
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
            return segments, ""

        return asyncio.new_event_loop().run_until_complete(_do())
