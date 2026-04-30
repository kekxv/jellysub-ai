"""在线翻译引擎 — OpenAI 兼容 Chat Completions API。"""

import json
import logging

from openai import OpenAI

from core.translate.base import TranslateEngine, _INSTRUCTION, parse_json_output

logger = logging.getLogger("uvicorn.error")


class OnlineTranslateEngine(TranslateEngine):
    """在线 API 翻译引擎。"""

    def __init__(self, api_url: str, api_key: str, model: str):
        self.api_url = api_url
        self.api_key = api_key
        self.model = model

    def translate_batch(self, texts: list[str], target_lang: str,
                        prompt_format: str = "json", thinking: bool = False,
                        context: str = "", source_lang: str = "") -> list[str] | None:
        return _translate_batch_online(texts, target_lang, self.api_url, self.api_key, self.model, thinking, context, source_lang)


def _translate_batch_online(
    texts: list[str],
    target_lang: str,
    api_url: str,
    api_key: str,
    model: str,
    thinking: bool,
    context: str = "",
    source_lang: str = "",
) -> list[str] | None:
    """在线模式翻译一批文本。"""
    source_label = f" (source language: {source_lang.upper()})" if source_lang else ""
    prompt = (
        f"{_INSTRUCTION} {target_lang}{source_label}\n\n"
        f"{json.dumps(texts, ensure_ascii=False)}\n\n"
        f"Output only the translated JSON array."
    )

    client = OpenAI(base_url=api_url, api_key=api_key, timeout=60.0)

    try:
        kwargs: dict = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 1024,
        }
        if not thinking:
            kwargs["extra_body"] = {"thinking": False}

        resp = client.chat.completions.create(**kwargs)
        content = resp.choices[0].message.content.strip()

        return parse_json_output(content, len(texts))

    except json.JSONDecodeError:
        logger.error("Translation response not valid JSON: %s", content[:200])
        return None
    except Exception:
        logger.exception("Translation batch failed")
        return None
