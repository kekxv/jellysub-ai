# Offline NLLB-200 + fastText Translation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a selectable offline translation engine using Meta NLLB-200 and fastText language identification.

**Architecture:** Keep the existing `TranslateEngine` interface and add an NLLB Seq2Seq implementation beside the current causal-language-model implementation. Load the NLLB tokenizer/model and fastText language model lazily from the shared model cache, map detected ISO languages to NLLB language tags, and expose the engine through config and the admin selector.

**Tech Stack:** Python, PyTorch, transformers, fasttext-wheel, FastAPI/Pydantic, pytest.

**Spec:** User request: “帮我增加一个离线翻译的支持：Meta NLLB-200 + fastText”.

## Global Constraints

- Existing online and Qwen local translation behavior remains compatible.
- Tests must not download or load production models.
- Model loading remains lazy and uses `model_cache`.

### Task 1: NLLB engine contract and language detection
**Files:** Create `core/translate/nllb.py`; Modify `core/translate/__init__.py`, `pyproject.toml`; Test `tests/test_translate_nllb.py`.

- [ ] Add failing tests for ISO-to-NLLB mapping, fastText label normalization, NLLB batch generation with injected tokenizer/model, and factory selection.
- [ ] Implement lazy NLLB model loading, fastText detection, language mapping, and `NLLBTranslateEngine`.
- [ ] Run focused tests.

### Task 2: Configuration and admin selection
**Files:** Modify `config.py`, `static/admin.html`; Test `tests/test_config.py`, `tests/test_admin_html.py`.

- [ ] Add NLLB and fastText model defaults/config fields while preserving old payloads.
- [ ] Add a NLLB-200 + fastText selector and ensure it serializes as local NLLB mode.
- [ ] Run focused tests.

### Task 3: Dependency and regression verification
**Files:** Modify `README.md` if needed; Test full suite.

- [ ] Document model download/cache requirements and offline operation.
- [ ] Run the full non-integration pytest suite and inspect the diff.
