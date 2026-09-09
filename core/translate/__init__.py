"""翻译引擎 — 统一入口。"""

import logging
import re
from collections.abc import Callable

from core.translate.base import (
    TranslateEngine,
    _LANG_GROUP,
    parse_json_output,
    parse_numbered_output,
)
from core.translate.local import (
    LocalTranslateEngine,
    load_local_model,
    release_model,
    check_model_idle,
    set_translate_busy,
)
from core.translate.openai_api import (
    OnlineTranslateEngine,
    _translate_batch_online,
)

logger = logging.getLogger("uvicorn.error")

__all__ = [
    "TranslateEngine",
    "LocalTranslateEngine",
    "OnlineTranslateEngine",
    "load_local_model",
    "release_model",
    "check_model_idle",
    "set_translate_busy",
    "parse_json_output",
    "parse_numbered_output",
    "translate_segments",
]

# =========================================================================== #
#  工厂函数
# =========================================================================== #


def get_translate_engine(
    mode: str = "local",
    model_name: str = "Qwen/Qwen3-0.6B",
    api_url: str = "",
    api_key: str = "",
    api_model: str = "",
    device: str | None = None,
) -> TranslateEngine:
    """创建翻译引擎实例。

    mode: "local" | "online"
    """
    if mode == "local":
        return LocalTranslateEngine(model_name=model_name, device=device)
    if mode == "online":
        return OnlineTranslateEngine(api_url=api_url, api_key=api_key, model=api_model)
    raise ValueError(f"Unknown translate mode: {mode}")


# =========================================================================== #
#  向后兼容入口
# =========================================================================== #

_MAX_BATCH_ITEMS = 5
_MAX_BATCH_CHARS = 300  # 单批总字符数上限，防止模型输出截断


async def translate_segments(
    segments: list[dict],
    target_lang: str,
    mode: str = "local",
    api_url: str = "",
    api_key: str = "",
    model: str = "",
    model_local: str = "",
    thinking: bool = False,
    prompt_format: str = "json",
    source_lang: str = "",
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[dict]:
    """
    翻译字幕片段。
    mode="online": 使用 OpenAI 兼容 API
    mode="local": 使用本地 transformers 模型
    source_lang: ASR 检测到的源语言代码
    """
    if not segments:
        return []

    # 纯标点片段不需要翻译
    _ONLY_PUNCT = re.compile(r'^[\s。，、；：！？.!?,:;"\'\'\"（）()\[\]{}]*$')

    all_texts = [seg["text"] for seg in segments]
    translated_texts: list[str | None] = [None] * len(all_texts)
    if progress_callback:
        progress_callback(0, len(all_texts))
    # 纯标点直接跳过，用原文填充
    for i, t in enumerate(all_texts):
        if _ONLY_PUNCT.match(t.strip()):
            translated_texts[i] = t.strip()
    failed_indices = [i for i, t in enumerate(translated_texts) if t is None]

    engine = get_translate_engine(
        mode=mode,
        model_name=model_local,
        api_url=api_url,
        api_key=api_key,
        api_model=model,
    )

    # 引擎自动选择最合适的输出格式
    engine_format = engine.preferred_format()
    logger.info("Translation engine prefers format: %s", engine_format)

    # 构建上下文：为每个文本索引预计算前后文
    def _build_context(indices: list[int], context_span: int = 3) -> str:
        """为给定索引列表构建上下文（前后各取 context_span 个句子）。"""
        if not indices:
            return ""
        min_idx = max(0, min(indices) - context_span)
        max_idx = min(len(all_texts), max(indices) + context_span)
        # 排除当前批次的索引，只取上下文的文本
        index_set = set(indices)
        context_parts = []
        for i in range(min_idx, max_idx):
            if i not in index_set and all_texts[i].strip():
                context_parts.append(all_texts[i])
        return " ".join(context_parts)

    source_str = str(source_lang) if isinstance(source_lang, (set, list)) else (source_lang or "")
    target_group = _LANG_GROUP.get(target_lang, {target_lang[:2]})
    source_base = _LANG_GROUP.get(source_str, {source_str[:2]} if source_str else set())
    source_matches_target = bool(source_base & target_group)

    def is_valid_translation(index: int, text: str) -> bool:
        original = all_texts[index].strip()
        translated = text.strip()
        return (
            bool(translated)
            and not _ONLY_PUNCT.match(translated)
            and (original != translated or source_matches_target)
        )

    # Process each batch independently.  Validate it before moving on so a
    # malformed response only retries the affected entries, never the task.
    pending_batches = []
    current = []
    current_chars = 0
    for idx in failed_indices:
        text = all_texts[idx]
        if not current or (len(current) < _MAX_BATCH_ITEMS and current_chars + len(text) <= _MAX_BATCH_CHARS):
            current.append(idx)
            current_chars += len(text)
        else:
            pending_batches.append(current)
            current = [idx]
            current_chars = len(text)
    if current:
        pending_batches.append(current)

    def translate_batch_or_split(indices: list[int]) -> tuple[list[int], list[int]]:
        """Return items needing one retry and items that exhausted a split attempt."""
        texts = [all_texts[index] for index in indices]
        batch_translated = engine.translate_batch(
            texts, target_lang, engine_format, thinking,
            context=_build_context(indices), source_lang=source_lang,
        )
        if batch_translated and len(batch_translated) == len(texts):
            retry_indices = []
            for index, translated in zip(indices, batch_translated):
                if is_valid_translation(index, translated):
                    translated_texts[index] = translated.strip()
                else:
                    retry_indices.append(index)
            return retry_indices, []

        if len(indices) == 1:
            logger.warning("Translation item %d failed; using source text", indices[0])
            return [], indices

        logger.warning("Translation batch failed, splitting it: indices=%s", indices)
        midpoint = len(indices) // 2
        left_retry, left_fallback = translate_batch_or_split(indices[:midpoint])
        right_retry, right_fallback = translate_batch_or_split(indices[midpoint:])
        return left_retry + right_retry, left_fallback + right_fallback

    for indices in pending_batches:
        retry_indices, fallback_indices = translate_batch_or_split(indices)

        for index in retry_indices:
            translated = engine.translate_batch(
                [all_texts[index]], target_lang, engine_format, thinking,
                context=_build_context([index]), source_lang=source_lang,
            )
            if translated and len(translated) == 1 and is_valid_translation(index, translated[0]):
                translated_texts[index] = translated[0].strip()
            else:
                translated_texts[index] = all_texts[index].strip()
                logger.warning("Translation item %d remained invalid; using source text", index)

        for index in fallback_indices:
            translated_texts[index] = all_texts[index].strip()

        if progress_callback:
            progress_callback(sum(text is not None for text in translated_texts), len(all_texts))

    result = [
        {"start": seg["start"], "end": seg["end"], "text": translated_text}
        for seg, translated_text in zip(segments, translated_texts)
    ]

    return result
