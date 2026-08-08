# Task History UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the admin task-history drawer easier to scan and act on by presenting progress, status, metadata, errors, and pagination with a consistent visual hierarchy.

**Architecture:** Keep the existing `/api/tasks` contract and vanilla-JavaScript renderer. Replace inline drawer presentation with semantic task-history classes, add status/type/time formatting helpers, and render a concise history summary plus richer task rows. CSS owns all layout and responsive styling to preserve the project’s light, rounded-card visual system.

**Tech Stack:** HTML, CSS, browser JavaScript, pytest, uv.

## Global Constraints

- Do not change task API payloads or authentication behavior.
- Escape all server-provided task fields before placing them in `innerHTML`.
- Preserve retry, delete, bulk delete, filters, pagination, and automatic task refresh.

---

### Task 1: Specify the task-history presentation contract

**Files:**
- Modify: `tests/test_admin_html.py`
- Modify: `static/admin.html:106-160,936-985`

**Interfaces:**
- Consumes: task objects from `GET /api/tasks`, including `id`, `status`, `pipeline_type`, `stage`, `progress`, `created_at`, `video_path`, `item_name`, and `error_message`.
- Produces: `renderTasks(tasks, total)` renders semantic summary and task-row elements with escaped untrusted content.

- [x] **Step 1: Write the failing test**

```python
def test_admin_task_history_uses_scannable_semantic_task_rows():
    html = _admin_html()
    assert 'class="task-history-summary"' in html
    assert 'class="task-history-row"' in html
    assert 'class="task-status-label"' in html
    assert 'class="task-history-path"' in html
```

- [x] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_admin_html.py::test_admin_task_history_uses_scannable_semantic_task_rows -q`

Expected: FAIL because the existing drawer only renders compact generic task cards.

- [x] **Step 3: Write minimal implementation**

Add the summary container and replace the generic card template with a task row containing a status label, stage, time, path, progress, optional failure message, and existing actions. Add `taskStatusLabel`, `pipelineLabel`, and `formatTaskTime` helpers; every task string remains wrapped in `escHtml`.

- [x] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_admin_html.py -q`

- [ ] **Step 5: Commit**

Run: `git add static/admin.html tests/test_admin_html.py && git commit -m "feat: improve task history hierarchy"`

### Task 2: Style the responsive task-history drawer

**Files:**
- Modify: `static/style.css:100-240`
- Test: `tests/test_admin_html.py`

**Interfaces:**
- Consumes: the task-history semantic classes created in Task 1.
- Produces: a responsive history summary, filter toolbar, task-row timeline, status accents, readable progress, and error/action treatment without inline layout styles.

- [x] **Step 1: Write the failing test**

```python
def test_admin_task_history_avoids_inline_layout_for_task_list():
    html = _admin_html()
    assert 'id="task-grid" class="task-history-list"' in html
    assert 'id="task-pagination" class="pagination task-history-pagination"' in html
```

- [x] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_admin_html.py::test_admin_task_history_avoids_inline_layout_for_task_list -q`

Expected: FAIL because the list and pagination currently rely on inline layout styles.

- [x] **Step 3: Write minimal implementation**

Move drawer, filter, selection, list, and pagination layout declarations to `style.css`. Add mobile rules that stack task metadata and actions without truncating the task title or path.

- [x] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_admin_html.py -q`

- [ ] **Step 5: Commit**

Run: `git add static/admin.html static/style.css tests/test_admin_html.py && git commit -m "style: refine task history drawer"`
