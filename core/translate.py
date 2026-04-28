"""翻译字幕 — 支持在线 OpenAI API 和本地 transformers 模型两种模式。"""

import json
import logging
import re
import time

import torch
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

# ASR 识别语言 → 目标语言代码的映射，用于判断是否需要重试翻译
_LANG_GROUP = {
    "zh-CN": {"zh"},
    "zh-TW": {"zh"},
    "zh": {"zh"},
    "en": {"en"},
    "ja": {"ja"},
    "ko": {"ko"},
    "fr": {"fr"},
    "de": {"de"},
    "es": {"es"},
    "ru": {"ru"},
}

# 分批发送的批次大小（小模型 JSON 输出能力有限，默认调小）
_BATCH_SIZE = 5

_INSTRUCTION = """You are a professional subtitle translator. Translate the given text from English to Chinese.

Rules:
1. Return ONLY a JSON array of translated strings, same length as input.
2. No explanations, no markdown, no extra text.
3. Keep translations concise and natural for subtitles.

Input JSON array:"""

_JSON_ARRAY_PATTERN = re.compile(r"\[[\s\S]*\]")

# 本地翻译模型单例
_local_model = None
_model_last_access: float = 0.0


def _touch_model():
    """记录模型最近访问时间。"""
    global _model_last_access
    _model_last_access = time.time()


def _load_local_model(model_name: str, device: str | None = None):
    """懒加载本地翻译模型。"""
    global _local_model, _model_last_access
    if _local_model is not None:
        _touch_model()
        return _local_model

    from env_config import MODEL_SOURCE
    if MODEL_SOURCE == "modelscope":
        try:
            from modelscope.hub.snapshot_download import snapshot_download
            local_dir = snapshot_download(model_name, cache_dir="./model_cache")
            model_name = local_dir
        except ModuleNotFoundError:
            logger.error("MODEL_SOURCE=modelscope 但 modelscope 未安装，请运行: pip install modelscope")
            raise

    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    logger.info("Loading translation model %s on %s", model_name, device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dtype = torch.bfloat16 if device == "cuda" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
    ).to(device)
    _local_model = {
        "tokenizer": tokenizer,
        "model": model,
        "device": device,
    }
    _touch_model()
    return _local_model


def _release_model():
    """释放翻译模型，释放显存/内存。"""
    global _local_model, _model_last_access
    if _local_model is not None:
        del _local_model
        _local_model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _model_last_access = 0.0
        logger.info("Translation model released")


def _check_model_idle(timeout: int):
    """检查模型是否空闲超时，如果是则释放。"""
    global _model_last_access
    if _local_model is not None and timeout > 0:
        elapsed = time.time() - _model_last_access
        if elapsed > timeout:
            _release_model()


def _local_generate(translator, texts: list[str], target_lang: str, prompt_format: str = "json", thinking: bool = False) -> list[str] | None:
    """使用本地模型生成翻译，返回翻译后的字符串列表。"""
    try:
        tokenizer = translator["tokenizer"]
        model = translator["model"]
        device = translator["device"]

        if prompt_format == "numbered":
            lines = "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))
            prompt = f"Translate the following English sentences to {target_lang}:\n\n{lines}"
        else:
            prompt = (
                f"{_INSTRUCTION} {target_lang}\n\n"
                f"{json.dumps(texts, ensure_ascii=False)}\n\n"
                f"Output only the translated JSON array."
            )

        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
        model_inputs = tokenizer([text], return_tensors="pt").to(device)

        max_tokens = 1024 if not thinking else 2048
        with torch.no_grad():
            generated_ids = model.generate(**model_inputs, max_new_tokens=max_tokens)

        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

        # 解析 thinking 分隔符 (151668 = </think>)
        try:
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip()

        try:
            if prompt_format == "numbered":
                return _parse_numbered_output(content, len(texts))
            return _parse_json_output(content, len(texts))
        except Exception:
            logger.warning("Model output unparsable: %r", content[:500])
            raise
    except Exception:
        logger.exception("Local translation failed")
    return None


def _parse_json_output(content: str, expected_count: int) -> list[str] | None:
    """解析 JSON 数组格式输出。"""
    if "```" in content:
        parts = content.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("["):
                content = part
                break

    # 修复 [{ "a", "b" }] 形式的错误输出：去掉外层 {}
    if content.lstrip().startswith("[{"):
        # 可能是 [{ "t1", "t2" }] 而非 [{"key": "val"}]
        # 提取引号之间的字符串
        inner = re.findall(r'"([^"]*)"', content)
        if inner and len(inner) >= expected_count:
            return inner[:max(expected_count, len(inner))]

    match = _JSON_ARRAY_PATTERN.search(content)
    if match:
        content = match.group(0)

    translated = json.loads(content)
    if not isinstance(translated, list):
        return None

    return [str(t) for t in translated]


def _parse_numbered_output(content: str, expected_count: int) -> list[str] | None:
    """解析编号格式输出: "1. 你好世界\n2. 这是测试"。"""
    results = []
    for line in content.split("\n"):
        line = line.strip()
        if not line:
            continue
        cleaned = re.sub(r"^\d+[.、\s]*\s*", "", line)
        if cleaned:
            results.append(cleaned)
    if len(results) == expected_count:
        return results
    return results if results else None


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
    prompt_format="json" | "numbered"（本地模式有效）
    source_lang: ASR 检测到的源语言代码（如 "en"），用于判断是否需要重试
    """
    if not segments:
        return []

    all_texts = [seg["text"] for seg in segments]
    translated_texts = []

    for i in range(0, len(all_texts), _BATCH_SIZE):
        batch = all_texts[i : i + _BATCH_SIZE]
        if mode == "local":
            batch_translated = await _translate_batch_local(
                batch, target_lang, model_local, prompt_format, thinking
            )
        else:
            batch_translated = await _translate_batch_online(
                batch, target_lang, api_url, api_key, model, thinking
            )
        if batch_translated is None or len(batch_translated) != len(batch):
            logger.warning(
                "Translation batch %d failed or length mismatch, using originals", i
            )
            logger.info("Translate batch %d (src): %s", i, json.dumps(batch, ensure_ascii=False))
            translated_texts.extend(batch)
        else:
            logger.info(
                "Translate batch %d: %s -> %s",
                i,
                json.dumps(batch, ensure_ascii=False),
                json.dumps(batch_translated, ensure_ascii=False),
            )
            translated_texts.extend(batch_translated)

    result = []
    for seg, translated_text in zip(segments, translated_texts):
        result.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": translated_text.strip(),
        })

    # 翻译结果与原文相同，判断是否需要重试
    retry_indices = []
    target_group = _LANG_GROUP.get(target_lang, {target_lang[:2]})
    source_base = _LANG_GROUP.get(source_lang, {source_lang[:2] if source_lang else set()})
    source_matches_target = bool(source_base & target_group)

    for i, (seg, translated) in enumerate(zip(segments, translated_texts)):
        if seg["text"].strip() == translated.strip() and not source_matches_target:
            retry_indices.append(i)

    if retry_indices:
        logger.info(
            "Found %d untranslated items (source_lang=%s, target_lang=%s), retrying",
            len(retry_indices), source_lang, target_lang,
        )
        retry_texts = [segments[i]["text"] for i in retry_indices]
        if mode == "local":
            retry_translated = await _translate_batch_local(
                retry_texts, target_lang, model_local, prompt_format, thinking
            )
        else:
            retry_translated = await _translate_batch_online(
                retry_texts, target_lang, api_url, api_key, model, thinking
            )
        if retry_translated:
            for idx, new_text in zip(retry_indices, retry_translated):
                result[idx]["text"] = new_text.strip()

    return result


async def _translate_batch_online(
    texts: list[str],
    target_lang: str,
    api_url: str,
    api_key: str,
    model: str,
    thinking: bool,
) -> list[str] | None:
    """在线模式翻译一批文本。"""
    prompt = (
        f"{_INSTRUCTION} {target_lang}\n\n"
        f"{json.dumps(texts, ensure_ascii=False)}\n\n"
        f"Output only the translated JSON array."
    )

    client = AsyncOpenAI(base_url=api_url, api_key=api_key)

    try:
        kwargs: dict = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
        }
        if not thinking:
            kwargs["extra_body"] = {"thinking": False}

        resp = await client.chat.completions.create(**kwargs)
        content = resp.choices[0].message.content.strip()

        if "```" in content:
            parts = content.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("["):
                    content = part
                    break

        match = _JSON_ARRAY_PATTERN.search(content)
        if match:
            content = match.group(0)

        translated = json.loads(content)
        if not isinstance(translated, list):
            logger.error("Translation response is not a JSON array: %s", content[:200])
            return None

        return [str(t) for t in translated]

    except json.JSONDecodeError:
        logger.error("Translation response not valid JSON: %s", content[:200])
        return None
    except Exception:
        logger.exception("Translation batch failed")
        return None


async def _translate_batch_local(
    texts: list[str],
    target_lang: str,
    model_name: str,
    prompt_format: str = "json",
    thinking: bool = False,
) -> list[str] | None:
    """本地模式翻译一批文本。"""
    try:
        translator = _load_local_model(model_name)
        return _local_generate(translator, texts, target_lang, prompt_format, thinking)
    except Exception:
        logger.exception("Local translation batch failed")
        return None
