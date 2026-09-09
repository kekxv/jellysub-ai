# ASR and Translation Item Progress Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Display exact completed/total units while a task recognizes VAD speech chunks and translates subtitle segments.

**Architecture:** Persist four integer counters on `tasks`: ASR completed/total and translation completed/total. VAD reports a callback after every audio chunk, and `translate_segments` reports cumulative successfully translated subtitle items after each serial batch. The task manager writes these callbacks into the task row and the existing task APIs expose them automatically; task cards render the matching counter for their active stage.

**Tech Stack:** SQLite, Python callbacks, FastAPI task API, vanilla JavaScript, pytest.

**Spec:** User request (2026-09-09): show recognition and translation progress such as `识别 12/100` and `翻译 34/120`.

## Global Constraints

- Online API requests remain serial; progress callbacks must not introduce parallel work.
- Failed translation still writes no subtitle files and resets progress on its next task attempt.
- Existing databases migrate without data loss; counters default to zero.

---

### Task 1: Persist task counters

**Files:**
- Modify: `core/task_manager.py`
- Test: `tests/test_task_manager.py`

**Interfaces:**
- Produces task API fields `asr_completed`, `asr_total`, `translate_completed`, `translate_total` as integers.

- [ ] **Step 1: Write the failing test**

```python
def test_task_progress_counters_are_saved(tmp_path):
    manager = TaskManager(str(tmp_path / "tasks.db"))
    task_id = manager.create_task("/media/movie.mkv")
    manager._update_task(task_id, asr_completed=12, asr_total=100,
                         translate_completed=34, translate_total=120)
    task = manager.get_task(task_id)
    assert (task["asr_completed"], task["asr_total"]) == (12, 100)
    assert (task["translate_completed"], task["translate_total"]) == (34, 120)
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `uv run pytest tests/test_task_manager.py::test_task_progress_counters_are_saved -q`

Expected: FAIL because task columns do not exist.

- [ ] **Step 3: Implement the schema and migration**

Add four `INTEGER DEFAULT 0` fields to fresh schemas and `columns_to_add`.

- [ ] **Step 4: Run the test and verify it passes**

Run: `uv run pytest tests/test_task_manager.py::test_task_progress_counters_are_saved -q`

Expected: PASS.

### Task 2: Report VAD and translation progress serially

**Files:**
- Modify: `core/asr/__init__.py`, `core/asr/vad_wrapper.py`, `core/translate/__init__.py`, `core/task_manager.py`
- Test: `tests/test_translate.py`, `tests/test_task_manager.py`

**Interfaces:**
- Consumes optional `progress_callback(completed: int, total: int)` in `run_asr`, `transcribe_with_vad`, and `translate_segments`.
- Produces persisted task counters during active ASR/translation stages.

- [ ] **Step 1: Write failing tests**

```python
@pytest.mark.asyncio
async def test_translate_segments_reports_completed_items_after_each_serial_batch(monkeypatch):
    updates = []
    monkeypatch.setattr("core.translate.get_translate_engine", lambda **_: SuccessfulEngine())
    await translate_segments(three_segments, "zh-CN", progress_callback=lambda done, total: updates.append((done, total)))
    assert updates == [(0, 3), (3, 3)]
```

```python
def test_task_pipeline_persists_translation_item_progress(tmp_path, monkeypatch):
    async def translate(_, __, progress_callback, **___):
        progress_callback(2, 3)
        progress_callback(3, 3)
        return translated_segments
    # execute a saved-source task and assert translate_completed == translate_total == 3
```

- [ ] **Step 2: Run focused tests and verify they fail**

Run: `uv run pytest tests/test_translate.py tests/test_task_manager.py -k 'progress' -q`

Expected: FAIL because callbacks and persisted counters do not exist.

- [ ] **Step 3: Implement callbacks**

Have VAD publish `(0, chunk_count)` before work and `(i + 1, chunk_count)` after every attempted chunk. Have translation publish `(0, source_item_count)` before batches and its count of non-null translated strings after each successful batch. Task manager callbacks update counters and the coarse percent without launching concurrent work.

- [ ] **Step 4: Run focused tests and verify they pass**

Run: `uv run pytest tests/test_translate.py tests/test_task_manager.py -k 'progress' -q`

Expected: PASS.

### Task 3: Render counters in active task UI

**Files:**
- Modify: `static/admin.html`
- Test: `tests/test_admin_html.py`

**Interfaces:**
- Consumes task counter fields from `/api/tasks`.
- Produces `识别 N/M` while `stage === 'asr'` and `翻译 N/M` while `stage === 'translating'`.

- [ ] **Step 1: Write the failing test**

```python
def test_admin_task_progress_renders_asr_and_translation_item_counts():
    html = _admin_html()
    assert "function taskProgressDetail(task)" in html
    assert "识别 ${task.asr_completed}/${task.asr_total}" in html
    assert "翻译 ${task.translate_completed}/${task.translate_total}" in html
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `uv run pytest tests/test_admin_html.py::test_admin_task_progress_renders_asr_and_translation_item_counts -q`

Expected: FAIL because task cards only render a percentage.

- [ ] **Step 3: Implement the task detail formatter and use it in active/history cards**

```javascript
function taskProgressDetail(task) {
  if (task.stage === 'asr' && Number(task.asr_total) > 0) return `识别 ${task.asr_completed}/${task.asr_total}`;
  if (task.stage === 'translating' && Number(task.translate_total) > 0) return `翻译 ${task.translate_completed}/${task.translate_total}`;
  return '';
}
```

- [ ] **Step 4: Run the test and verify it passes**

Run: `uv run pytest tests/test_admin_html.py::test_admin_task_progress_renders_asr_and_translation_item_counts -q`

Expected: PASS.

### Task 4: Full verification

- [ ] Run: `uv run pytest -q && git diff --check`
- [ ] Expected: all non-integration tests PASS and no whitespace errors.
