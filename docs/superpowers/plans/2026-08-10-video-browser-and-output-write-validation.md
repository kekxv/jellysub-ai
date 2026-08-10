# Video Browser and Output Write Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Display only the selected monitored-directory contents and reject subtitle jobs before queueing if their output directory is unwritable.

**Architecture:** The client-side tree will identify a selected monitored root and filter every video to that root before it derives direct child folders/files. The server will perform a create-and-delete probe in each destination directory after path authorization and before it creates a task.

**Tech Stack:** FastAPI, Python pathlib, pytest, inline JavaScript, Node.js VM.

## Global Constraints

- Preserve existing authenticated API routes and response conventions.
- Do not queue a subtitle task when its media directory cannot accept a probe file.
- Remove every probe file after a successful or failed check.
- Write and observe failing regression tests before implementation.

---

### Task 1: Scope the video tree to the selected monitored root

**Files:**
- Modify: `static/admin.html:669-740`
- Test: `tests/test_admin_html.py`

**Interfaces:**
- Consumes: `buildDirTree(videos, currentPath, rootDirs)`.
- Produces: direct `subdirs` and `files` only from the selected root/path.

- [ ] **Step 1: Write the failing test**

```python
def test_admin_video_tree_scopes_a_monitored_root_to_its_contents():
    result = _run_admin_tree(
        [
            {"name": "episode.mkv", "path": "/media/tvdrama/season-1/episode.mkv"},
            {"name": "movie.mkv", "path": "/media/anime/movie.mkv"},
            {"name": "film.mkv", "path": "/media/media/film.mkv"},
        ],
        "tvdrama",
        ["/media/anime", "/media/media", "/media/tvdrama"],
    )
    assert result == {"subdirs": ["season-1"], "files": []}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_admin_html.py::test_admin_video_tree_scopes_a_monitored_root_to_its_contents -q`

Expected: FAIL because the selected root is treated as an empty relative path across all monitored roots.

- [ ] **Step 3: Write minimal implementation**

```javascript
const selectedRoot = rootDirs.find(root =>
    currentParts[0] === root.replace(/\/+$/, '').split('/').pop()
);
const effectiveParts = selectedRoot ? currentParts.slice(1) : currentParts;
```

Derive a path from the selected root for every video and skip videos that are outside it.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_admin_html.py::test_admin_video_tree_scopes_a_monitored_root_to_its_contents -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add static/admin.html tests/test_admin_html.py
git commit -m "fix: scope video browser to selected root"
```

### Task 2: Reject unwritable subtitle destinations before task creation

**Files:**
- Modify: `main.py:585-656`
- Test: `tests/test_api.py`

**Interfaces:**
- Consumes: a validated absolute video path.
- Produces: `_ensure_subtitle_output_writable(video_path: str) -> None`, raising `HTTPException(409, "Subtitle output directory is not writable")` when a temporary probe cannot be written.

- [ ] **Step 1: Write the failing tests**

```python
def test_subtitle_generation_rejects_an_unwritable_output_directory(client, video_path, monkeypatch):
    _configure_allowed_video_directory(monkeypatch, video_path.parent)
    monkeypatch.setattr(
        "main._ensure_subtitle_output_writable",
        lambda _path: (_ for _ in ()).throw(
            HTTPException(409, "Subtitle output directory is not writable")
        ),
    )
    _authenticated_client(client)
    response = client.post("/api/videos/subtitle", json={"video_path": str(video_path)})
    assert response.status_code == 409
    assert response.json() == {"detail": "Subtitle output directory is not writable"}
```

Add the same behavioral assertion for `/api/videos/subtitle/batch`; assert no tasks were created in each failure case.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_api.py -k 'unwritable_output' -q`

Expected: FAIL because the helper and its route calls do not exist.

- [ ] **Step 3: Write minimal implementation**

```python
def _ensure_subtitle_output_writable(video_path: str) -> None:
    output_dir = Path(video_path).parent
    probe_path = output_dir / f".jellysub-write-test-{os.getpid()}-{time.time_ns()}"
    try:
        with probe_path.open("x", encoding="utf-8"):
            pass
    except OSError as exc:
        raise HTTPException(409, "Subtitle output directory is not writable") from exc
    finally:
        probe_path.unlink(missing_ok=True)
```

Call it once per accepted single or batch path immediately after `_validate_video_path`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_api.py -k 'unwritable_output' -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add main.py tests/test_api.py
git commit -m "fix: validate subtitle output writability"
```

### Task 3: Verify the integrated behavior

**Files:**
- Verify: `tests/test_admin_html.py`, `tests/test_api.py`, `tests/test_task_manager.py`

**Interfaces:**
- Consumes: completed Tasks 1 and 2.
- Produces: fresh evidence for browser navigation and task startup behavior.

- [ ] **Step 1: Run focused regressions**

Run: `uv run pytest tests/test_admin_html.py tests/test_api.py -q`

Expected: PASS.

- [ ] **Step 2: Run the full non-integration suite**

Run: `uv run pytest -q`

Expected: PASS; integration tests are deselected by the project configuration.

- [ ] **Step 3: Inspect the final diff**

Run: `git diff --check && git status --short`

Expected: no whitespace errors and only intended source, test, and plan changes.

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/plans/2026-08-10-video-browser-and-output-write-validation.md
git commit -m "docs: plan video browser and write validation fixes"
```
