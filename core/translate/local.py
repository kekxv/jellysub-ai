"""本地翻译引擎 — 基于 transformers 模型。"""

import json
import logging
import time

import torch

from core.translate.base import TranslateEngine, parse_json_output, parse_numbered_output

logger = logging.getLogger("uvicorn.error")

_local_model = None
_model_last_access: float = 0.0
_model_busy: bool = False


def set_translate_busy(busy: bool):
    global _model_busy
    _model_busy = busy
    if busy:
        _touch_model()


def _touch_model():
    global _model_last_access
    _model_last_access = time.time()


def load_local_model(model_name: str, device: str | None = None):
    """懒加载本地翻译模型。"""
    global _local_model, _model_last_access
    if _local_model is not None:
        _touch_model()
        return _local_model

    from env_config import MODEL_SOURCE
    from pathlib import Path

    cache_dir = "./model_cache"

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
            cached = _cached_path(model_name)
            local_dir = cached or snapshot_download(model_name, cache_dir=cache_dir)
            model_name = local_dir
        except ModuleNotFoundError:
            logger.error("MODEL_SOURCE=modelscope 但 modelscope 未安装")
            raise
    else:
        try:
            from huggingface_hub import snapshot_download
            cached = _cached_path(model_name)
            local_dir = cached or snapshot_download(model_name, cache_dir=cache_dir)
            model_name = local_dir
        except ModuleNotFoundError:
            logger.warning("huggingface_hub 未安装，使用默认缓存路径")

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


def release_model():
    """释放翻译模型。"""
    global _local_model, _model_last_access
    if _local_model is not None:
        del _local_model
        _local_model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
        _model_last_access = 0.0
        logger.info("Translation model released")


def check_model_idle(timeout: int):
    """检查模型空闲超时并释放。"""
    global _model_last_access, _model_busy
    if _local_model is not None and timeout > 0:
        if _model_busy:
            _touch_model()  # 正在使用中，自动续期
            return
        elapsed = time.time() - _model_last_access
        if elapsed > timeout:
            release_model()


class LocalTranslateEngine(TranslateEngine):
    """本地 transformers 翻译引擎。"""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        device: str | None = None,
    ):
        self.model_name = model_name
        self._device = device

    def preferred_format(self) -> str:
        return "numbered"

    def translate_batch(self, texts: list[str], target_lang: str,
                        prompt_format: str = "json", thinking: bool = False,
                        context: str = "", source_lang: str = "") -> list[str] | None:
        translator = load_local_model(self.model_name, self._device)
        return _local_generate(translator, texts, target_lang, prompt_format, thinking, context, source_lang)


def _local_generate(
    translator: dict,
    texts: list[str],
    target_lang: str,
    prompt_format: str = "json",
    thinking: bool = False,
    context: str = "",
    source_lang: str = "",
) -> list[str] | None:
    """使用本地模型生成翻译。"""
    from core.translate.base import _INSTRUCTION
    try:
        tokenizer = translator["tokenizer"]
        model = translator["model"]
        device = translator["device"]

        if prompt_format == "numbered":
            lines = "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))
            context_line = ""
            if context:
                context_line = (
                    f"Context (for reference only, DO NOT translate):\n{context}\n\n"
                )
            # 明确指明源语言和目标语言，使用更强烈的指令
            src = source_lang.upper() if source_lang else "the source"
            tgt = target_lang
            prompt = (
                f"Translate these {src} sentences to {tgt}. "
                f"Each line must be translated to {tgt}. "
                f"Do NOT keep the original {src} text.\n"
                f"Output ONLY the {tgt} translations, one per line, preserving the numbering.\n\n"
                f"{context_line}{lines}\n\n"
                f"Translate to {tgt}:"
            )
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

        # Qwen 的 thinking EOS token
        try:
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip()
        logger.info("Local model output (format=%s): %r", prompt_format, content[:500])

        try:
            if prompt_format == "numbered":
                return parse_numbered_output(content, len(texts))
            return parse_json_output(content, len(texts))
        except Exception:
            logger.warning("Model output unparsable: %r", content[:500])
            raise
    except Exception:
        logger.exception("Local translation failed")
    return None
