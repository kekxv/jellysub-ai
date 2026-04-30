"""翻译引擎 — 统一入口。"""

import logging
import re

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

    # 最多重试 3 次翻译失败的批次
    for attempt in range(3):
        if not failed_indices:
            break

        # 如果是重试，减小 Batch Size 以提高成功率
        retry_batch_size = _MAX_BATCH_ITEMS if attempt == 0 else max(1, _MAX_BATCH_ITEMS - attempt * 2)

        # 将失败索引按批次切分，只翻译需要的批次
        retry_batches = []
        current = []
        current_chars = 0
        for idx in failed_indices:
            text = all_texts[idx]
            if not current or (len(current) < retry_batch_size and current_chars + len(text) <= _MAX_BATCH_CHARS):
                current.append((idx, text))
                current_chars += len(text)
            else:
                if current:
                    retry_batches.append(current)
                current = [(idx, text)]
                current_chars = len(text)
        if current:
            retry_batches.append(current)

        failed_indices = []
        for batch_items in retry_batches:
            indices = [item[0] for item in batch_items]
            texts = [item[1] for item in batch_items]

            batch_translated = engine.translate_batch(texts, target_lang, engine_format, thinking,
                                                       context=_build_context(indices), source_lang=source_lang)
            if batch_translated and len(batch_translated) == len(texts):
                for idx, translated in zip(indices, batch_translated):
                    translated_texts[idx] = translated.strip()
            else:
                logger.warning(
                    "Translation batch failed (attempt %d), indices=%s",
                    attempt + 1, indices,
                )
                failed_indices.extend(indices)

    # 用原文填充仍失败的索引
    for i in failed_indices:
        if translated_texts[i] is None:
            translated_texts[i] = all_texts[i]

    result = []
    for seg, translated_text in zip(segments, translated_texts):
        result.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": translated_text,
        })

    # --- 翻译质量检查 ---
    retry_indices = []
    source_str = str(source_lang) if isinstance(source_lang, (set, list)) else (source_lang or "")
    target_group = _LANG_GROUP.get(target_lang, {target_lang[:2]})
    source_base = _LANG_GROUP.get(source_str, {source_str[:2]} if source_str else set())
    source_matches_target = bool(source_base & target_group)

    for i, (seg, translated) in enumerate(zip(segments, translated_texts)):
        orig = seg["text"].strip()
        trans = translated.strip()

        # 翻译结果只有标点（说明模型没翻译出内容），需要重试
        if _ONLY_PUNCT.match(trans):
            retry_indices.append(i)
            continue

        # 翻译结果与原文相同，且源语言与目标语言不匹配 → 需要重试
        if orig == trans and not source_matches_target:
            retry_indices.append(i)

    if retry_indices:
        logger.info(
            "Found %d untranslated/poor items, forcing retry (source_lang=%s, target_lang=%s)",
            len(retry_indices), source_str, target_lang,
        )
        retry_texts = [segments[i]["text"] for i in retry_indices]
        retry_translated = engine.translate_batch(retry_texts, target_lang, engine_format, thinking,
                                                   context=_build_context(retry_indices), source_lang=source_lang)
        if retry_translated and len(retry_translated) == len(retry_texts):
            for idx, new_text in zip(retry_indices, retry_translated):
                new_stripped = new_text.strip()
                # 重试后仍然只有标点或和原文一样，放弃重试
                orig = segments[idx]["text"].strip()
                if not _ONLY_PUNCT.match(new_stripped) and new_stripped != orig:
                    result[idx]["text"] = new_stripped
                    logger.info("Retry fixed index %d: %s -> %s",
                                idx, orig[:40], new_stripped[:40])
                else:
                    logger.info("Retry index %d still untranslated, keeping original", idx)
        else:
            logger.warning("Retry for untranslated also failed, keeping originals")

    return result
