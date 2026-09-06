# Online Translation Test and Failure Handling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let an administrator test the configured online translation service from Settings and prevent subtitle files from being written when translation fails or returns untranslated source text.

**Architecture:** Extend the authenticated translation test API to accept the current form's online endpoint settings without persisting them and to report a failed test as an HTTP error. Make `translate_segments` strict: after its existing retries, it returns `None` rather than substituting the source text for failed translations; the task pipeline already converts that into a failed task before subtitle writers run. Add the test action and result panel beside the online translation fields.

**Tech Stack:** FastAPI/Pydantic, OpenAI-compatible Python client, vanilla browser JavaScript, pytest/TestClient, Node VM HTML tests.

**Spec:** User request (2026-09-06): add a Settings test button for configured online endpoints and show test results; do not write source-language subtitles when translation fails, and report the error.

## Global Constraints

- The endpoint remains authenticated and never returns an API key.
- The online test uses unsaved values currently present in the Settings form.
- No target or bilingual SRT is written after a translation failure.
- Translation output must contain one non-empty translated string for every non-punctuation input item.

---

### Task 1: Strict translation success contract

**Files:**
- Modify: `core/translate/__init__.py:translate_segments`
- Test: `tests/test_translate.py`

**Interfaces:**
- Consumes: `TranslateEngine.translate_batch(texts, target_lang, ...) -> list[str] | None`
- Produces: `translate_segments(...) -> list[dict] | None`, where `None` means no safe translated subtitle output exists.

- [ ] **Step 1: Write the failing test**

```python
@pytest.mark.asyncio
async def test_translate_segments_returns_none_when_a_batch_never_translates(monkeypatch):
    monkeypatch.setattr("core.translate.get_translate_engine", lambda **_: FailingEngine())
    result = await translate_segments([
        {"start": 0.0, "end": 1.0, "text": "Hello"},
    ], "zh-CN", mode="online", source_lang="en")
    assert result is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_translate.py::test_translate_segments_returns_none_when_a_batch_never_translates -q`

Expected: FAIL because the implementation substitutes `"Hello"` and returns a segment list.

- [ ] **Step 3: Write minimal implementation**

```python
if failed_indices:
    logger.error("Translation failed for %d subtitle segments", len(failed_indices))
    return None
```

Place it after all translation and quality retries, before constructing the result; retain source text only for punctuation-only segments.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_translate.py::test_translate_segments_returns_none_when_a_batch_never_translates -q`

Expected: PASS.

### Task 2: Test API exposes online endpoint outcome

**Files:**
- Modify: `main.py:TestTranslationRequest` and `/api/test/translate`
- Test: `tests/test_api.py`

**Interfaces:**
- Consumes: JSON `{api_url, api_key, model, target_language?}` from an authenticated Settings form.
- Produces: success JSON `{results: [{original, translated}]}` or HTTP 502 `{detail: "Translation test failed"}`.

- [ ] **Step 1: Write the failing test**

```python
def test_online_translation_test_uses_request_configuration(client, monkeypatch):
    translate = AsyncMock(return_value=[{"start": 0, "end": 2, "text": "你好"}])
    monkeypatch.setattr("core.translate.translate_segments", translate)
    _authenticated_client(client)
    response = client.post("/api/test/translate", json={
        "api_url": "https://example.test/v1", "api_key": "test-key",
        "model": "test-model", "texts": ["Hello"],
    })
    assert response.status_code == 200
    assert translate.await_args.kwargs["api_url"] == "https://example.test/v1"

def test_online_translation_test_returns_error_for_failed_translation(client, monkeypatch):
    monkeypatch.setattr("core.translate.translate_segments", AsyncMock(return_value=None))
    _authenticated_client(client)
    response = client.post("/api/test/translate", json={"api_url": "https://example.test/v1"})
    assert response.status_code == 502
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_api.py -k 'online_translation_test' -q`

Expected: FAIL because the request model ignores the online settings and the API iterates `None`.

- [ ] **Step 3: Write minimal implementation**

```python
if not translated:
    raise HTTPException(status_code=502, detail="Translation test failed")
```

Add optional request fields and resolve each field from the request first, then configuration defaults; invoke translation in `online` mode whenever a request endpoint is supplied.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_api.py -k 'online_translation_test' -q`

Expected: PASS.

### Task 3: Settings test control and result feedback

**Files:**
- Modify: `static/admin.html:translate-online-fields` and Settings JavaScript
- Test: `tests/test_admin_html.py`

**Interfaces:**
- Consumes: values from `translate_api_url`, `translate_api_key`, `translate_model`, and `target_language` fields.
- Produces: `POST /api/test/translate` and renders either original/translated pairs or the API error in `translate-test-result`.

- [ ] **Step 1: Write the failing test**

```python
def test_admin_online_translation_settings_include_a_test_control_and_result_panel():
    html = _admin_html()
    assert 'id="translate-test-btn"' in html
    assert 'id="translate-test-result"' in html
    assert "async function testOnlineTranslation()" in html
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_admin_html.py::test_admin_online_translation_settings_include_a_test_control_and_result_panel -q`

Expected: FAIL because the online Settings fields have no test action or result container.

- [ ] **Step 3: Write minimal implementation**

```javascript
const response = await fetch('/api/test/translate', {
  method: 'POST', headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ api_url, api_key, model, target_language })
});
```

Disable the button while testing, show a concise failure message for non-2xx/network errors, and render escaped original/translated text pairs for successful results.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_admin_html.py::test_admin_online_translation_settings_include_a_test_control_and_result_panel -q`

Expected: PASS.

### Task 4: Pipeline regression and full verification

**Files:**
- Modify: `tests/test_task_manager.py`

**Interfaces:**
- Consumes: `translate_segments(...) -> None` during a queued task.
- Produces: failed task with no `translated_segments`; subtitle writers are not called.

- [ ] **Step 1: Write the failing test**

```python
def test_pipeline_does_not_write_subtitles_after_translation_failure(tmp_path, monkeypatch):
    monkeypatch.setattr("core.translate.translate_segments", async_returning_none)
    write_target = Mock()
    monkeypatch.setattr("core.subtitle_writer.generate_srt", write_target)
    manager._execute_pipeline(task)
    assert manager.get_task(task_id)["status"] == "pending"
    write_target.assert_not_called()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_task_manager.py::test_pipeline_does_not_write_subtitles_after_translation_failure -q`

Expected: FAIL before Task 1 because source-language fallback reaches the writers.

- [ ] **Step 3: Confirm minimal implementation path**

No additional production code is needed: the existing `if not translated: raise RuntimeError("Translation failed")` gate precedes both writer calls.

- [ ] **Step 4: Run focused and complete verification**

Run: `uv run pytest tests/test_translate.py tests/test_api.py tests/test_admin_html.py tests/test_task_manager.py -q && uv run pytest -q`

Expected: all selected and non-integration tests PASS.
