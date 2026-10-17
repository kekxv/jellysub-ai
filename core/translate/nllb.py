"""Offline Meta NLLB-200 translation with fastText language detection."""

from __future__ import annotations

import logging
import os
import re
from urllib.request import urlretrieve
from pathlib import Path

import torch

from core.translate.base import TranslateEngine

logger = logging.getLogger("uvicorn.error")

NLLB_MODEL_DEFAULT = "JustFrederik/nllb-200-distilled-600M-ct2-int8"
NLLB_MODEL_REVISION = "302d78f00e6fdb50a1064059df7c392b735e9d05"
FASTTEXT_MODEL_DEFAULT = "model_cache/lid.176.ftz"
FASTTEXT_MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"

# NLLB uses three-letter ISO-639-3 language identifiers plus script names.
_ISO_TO_NLLB = {
    "en": "eng_Latn", "eng": "eng_Latn", "zh": "zho_Hans", "zho": "zho_Hans",
    "zh-cn": "zho_Hans", "zh-tw": "zho_Hant", "ja": "jpn_Jpan", "jpn": "jpn_Jpan",
    "ko": "kor_Hang", "kor": "kor_Hang", "fr": "fra_Latn", "fra": "fra_Latn",
    "de": "deu_Latn", "deu": "deu_Latn", "es": "spa_Latn", "spa": "spa_Latn",
    "it": "ita_Latn", "ita": "ita_Latn", "pt": "por_Latn", "por": "por_Latn",
    "ru": "rus_Cyrl", "rus": "rus_Cyrl", "ar": "arb_Arab", "ara": "arb_Arab",
    "nl": "nld_Latn", "nld": "nld_Latn", "pl": "pol_Latn", "pol": "pol_Latn",
    "tr": "tur_Latn", "tur": "tur_Latn", "uk": "ukr_Cyrl", "ukr": "ukr_Cyrl",
    "vi": "vie_Latn", "vie": "vie_Latn", "th": "tha_Thai", "tha": "tha_Thai",
    "hi": "hin_Deva", "hin": "hin_Deva", "he": "heb_Hebr", "heb": "heb_Hebr",
}


def normalize_fasttext_label(label: str) -> str:
    """Convert fastText's ``__label__foo`` output to a bare language code."""
    return re.sub(r"^__label__", "", str(label)).strip().lower()


def fasttext_label_to_nllb(label: str) -> str:
    """Map a fastText/ISO language label to an NLLB language token."""
    normalized = normalize_fasttext_label(label)
    mapped = _ISO_TO_NLLB.get(normalized)
    if mapped:
        return mapped
    # Accept an NLLB token supplied by callers while normalizing fastText's
    # lower-case label output (for example ``eng_latn``).
    if "_" in normalized:
        language, script = normalized.split("_", 1)
        return f"{language}_{script.title()}"
    return normalized


_fasttext_model = None
_nllb_model = None


def _offline_models() -> bool:
    return os.environ.get("OFFLINE_MODELS", "false").lower() in {"1", "true", "yes", "on"}


def _fasttext_path(model_path: str) -> Path:
    path = Path(model_path)
    # Keep the default path aligned with MODEL_CACHE when operators override it.
    if model_path == FASTTEXT_MODEL_DEFAULT and os.environ.get("MODEL_CACHE"):
        path = Path(os.environ["MODEL_CACHE"]) / "lid.176.ftz"
    return path


def load_fasttext_model(model_path: str = FASTTEXT_MODEL_DEFAULT):
    """Load the local fastText lid.176 model lazily."""
    global _fasttext_model
    if _fasttext_model is None:
        path = _fasttext_path(model_path)
        if not path.exists():
            if _offline_models():
                raise FileNotFoundError(
                    f"fastText model not found: {path}; OFFLINE_MODELS is enabled. "
                    "Download lid.176.ftz into the model cache before starting."
                )
            path.parent.mkdir(parents=True, exist_ok=True)
            url = os.environ.get("FASTTEXT_MODEL_URL", FASTTEXT_MODEL_URL)
            try:
                logger.info("Downloading fastText language model to %s", path)
                urlretrieve(url, str(path))
            except Exception as exc:
                raise RuntimeError(
                    f"Unable to download fastText model to {path}. "
                    "Set OFFLINE_MODELS=true and provide lid.176.ftz, or check network access."
                ) from exc
        import fasttext
        _fasttext_model = fasttext.load_model(str(path))
    return _fasttext_model


def detect_fasttext_language(text: str, model=None) -> str:
    """Detect an ISO language code from subtitle text using fastText."""
    model = model or load_fasttext_model()
    # The batch API avoids fastText's NumPy-2-incompatible copy=False path.
    labels, _ = model.predict([text.replace("\n", " ")], k=1)
    return normalize_fasttext_label(labels[0][0]) if labels and labels[0] else ""


def load_nllb_model(model_name: str = NLLB_MODEL_DEFAULT, device: str | None = None):
    """Download the prequantized CT2 snapshot, then load with INT8 compute."""
    global _nllb_model
    # Existing configurations used the original Meta model ID.
    if model_name in {"facebook/nllb-200-distilled-600M", "meta/nllb-200-distilled-600M"}:
        model_name = NLLB_MODEL_DEFAULT
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if _nllb_model is not None and _nllb_model["key"] == (model_name, device):
        return _nllb_model
    import ctranslate2
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    cache_dir = os.environ.get("MODEL_CACHE", "./model_cache")
    source = os.environ.get("NLLB_MODEL_SOURCE", "huggingface").lower()
    if source not in {"huggingface", "modelscope", "offline"}:
        raise ValueError("NLLB_MODEL_SOURCE must be huggingface, modelscope, or offline")
    local_only = _offline_models() or source == "offline"
    try:
        if Path(model_name).is_dir():
            resolved_name = model_name
        elif source == "modelscope":
            # Use an explicitly configured CT2 repository on ModelScope. Never
            # download the original FP32 checkpoint or convert it implicitly.
            if local_only:
                raise ValueError("Use a local CT2 directory with ModelScope in offline mode")
            from modelscope import snapshot_download as ms_download
            resolved_name = ms_download(model_id=model_name, cache_dir=cache_dir)
        else:
            resolved_name = snapshot_download(
                repo_id=model_name,
                revision=NLLB_MODEL_REVISION if model_name == NLLB_MODEL_DEFAULT else None,
                cache_dir=cache_dir, local_files_only=local_only,
                allow_patterns=["model.bin", "config.json", "shared_vocabulary.*",
                                "source_vocabulary.*", "target_vocabulary.*",
                                "tokenizer*", "special_tokens_map.json", "sentencepiece.bpe.model"],
            )
        tokenizer = AutoTokenizer.from_pretrained(
            resolved_name, local_files_only=True, trust_remote_code=False,
        )
        model = ctranslate2.Translator(resolved_name, device=device, compute_type="int8")
    except Exception as exc:
        location = "offline cache" if local_only else "model source"
        raise RuntimeError(
            f"Unable to load NLLB INT8 from {location}: {model_name}. "
            "Provide a complete CTranslate2 INT8 model and tokenizer, or check download access."
        ) from exc
    _nllb_model = {"tokenizer": tokenizer, "model": model, "device": device,
                   "key": (model_name, device)}
    return _nllb_model


def _nllb_generate(translator: dict, texts: list[str], source_lang: str, target_lang: str) -> list[str]:
    tokenizer = translator["tokenizer"]
    tokenizer.src_lang = source_lang
    # Explicitly reject unknown language tokens instead of silently using UNK.
    for lang in (source_lang, target_lang):
        language_ids = getattr(tokenizer, "lang_code_to_id", None)
        language_tokens = getattr(tokenizer, "added_tokens_encoder", None)
        known_language_tokens = language_ids if isinstance(language_ids, dict) else language_tokens
        if isinstance(known_language_tokens, dict) and lang not in known_language_tokens:
            raise ValueError(f"Unsupported NLLB language: {lang}")
        if tokenizer.convert_tokens_to_ids(lang) == tokenizer.unk_token_id:
            raise ValueError(f"Unsupported NLLB language: {lang}")
    encoded = tokenizer(texts, truncation=True, max_length=512)["input_ids"]
    tokens = [tokenizer.convert_ids_to_tokens(ids) for ids in encoded]
    results = translator["model"].translate_batch(
        tokens, target_prefix=[[target_lang] for _ in texts],
        beam_size=2, max_decoding_length=512,
    )
    translations = []
    for result in results:
        output = result.hypotheses[0]
        if output and output[0] == target_lang:
            output = output[1:]
        translations.append(tokenizer.decode(
            tokenizer.convert_tokens_to_ids(output), skip_special_tokens=True,
        ).strip())
    return translations


class NLLBTranslateEngine(TranslateEngine):
    """NLLB-200 Seq2Seq engine with fastText source-language detection."""

    def __init__(self, model_name: str = NLLB_MODEL_DEFAULT, device: str | None = None):
        self.model_name = model_name
        self._device = device

    def preferred_format(self) -> str:
        return "numbered"

    def translate_batch(self, texts, target_lang, prompt_format="numbered", thinking=False,
                        context="", source_lang=""):
        translator = load_nllb_model(self.model_name, self._device)
        target_tag = fasttext_label_to_nllb(target_lang)
        if isinstance(source_lang, (set, list, tuple)):
            source_lang = next(iter(source_lang), "")

        # Detect each subtitle independently. A mixed-language batch must not
        # fail as a whole when one line has an unknown or unsupported language.
        groups: dict[str, list[tuple[int, str]]] = {}
        results: list[str | None] = [None] * len(texts)
        for index, text in enumerate(texts):
            try:
                detected = source_lang or detect_fasttext_language(text)
                source_tag = fasttext_label_to_nllb(detected)
                if not source_tag or not target_tag:
                    raise ValueError(f"source={detected!r}, target={target_lang!r}")
                groups.setdefault(source_tag, []).append((index, text))
            except (RuntimeError, ValueError, TypeError) as exc:
                logger.warning("Skipping subtitle %d: unsupported NLLB language (%s)", index, exc)

        for source_tag, items in groups.items():
            try:
                translated = _nllb_generate(
                    translator, [text for _, text in items], source_tag, target_tag,
                )
            except (RuntimeError, ValueError, TypeError) as exc:
                logger.warning("Skipping %d subtitles for unsupported NLLB language %s: %s",
                               len(items), source_tag, exc)
                continue
            for (index, _), value in zip(items, translated):
                results[index] = value
        return results
