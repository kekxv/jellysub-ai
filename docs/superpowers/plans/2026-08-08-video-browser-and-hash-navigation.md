# Video Browser and Hash Navigation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve the local-video browser and preserve the selected directory in the URL hash so reload, browser navigation, and copied URLs reopen the same view.

**Architecture:** Keep the existing client-side directory tree built from `/api/videos`. A small hash adapter serializes the relative browser path with `encodeURIComponent`, restores it before the video API result renders, and re-renders when browser hash navigation changes. The video list receives semantic toolbar, count, directory-row, and media-row classes styled in the existing light dashboard system.

**Tech Stack:** HTML, CSS, browser JavaScript, pytest, uv.

## Global Constraints

- Do not change `/api/videos` or subtitle-generation API contracts.
- Encode the full relative directory path before placing it in `location.hash`.
- Invalid or malformed hashes must fall back safely to the monitored-directory root.
- Preserve escaping for every file or directory name used in `innerHTML` and inline event handlers.

---

### Task 1: Add durable directory hash navigation

**Files:**
- Modify: `tests/test_admin_html.py`
- Modify: `static/admin.html:575-820,1146`

**Interfaces:**
- Consumes: `currentDirPath` and a `#videos=<encoded-relative-path>` browser hash.
- Produces: `videoPathFromHash()`, `syncVideoPathHash(path)`, and a `hashchange` listener that restore the selected directory and retain search/selection behavior.

- [x] **Step 1: Write the failing test**

```python
def test_admin_video_browser_restores_directory_from_encoded_hash():
    html = _admin_html()
    assert "function videoPathFromHash()" in html
    assert "encodeURIComponent(relativePath)" in html
    assert "window.addEventListener('hashchange', restoreVideoPathFromHash)" in html
```

- [x] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_admin_html.py::test_admin_video_browser_restores_directory_from_encoded_hash -q`

Expected: FAIL because directory navigation currently only lives in memory.

- [x] **Step 3: Write minimal implementation**

Add the three named helpers. `videoPathFromHash` accepts only the `#videos=` prefix and catches decode failures. `navigateTo` writes the encoded hash after changing the directory; startup and `hashchange` restore it without re-writing the hash.

- [x] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_admin_html.py -q`

- [ ] **Step 5: Commit**

Run: `git add static/admin.html tests/test_admin_html.py && git commit -m "feat: preserve video directory in browser hash"`

### Task 2: Improve the video browser visual hierarchy

**Files:**
- Modify: `tests/test_admin_html.py`
- Modify: `static/admin.html:65-100,709-820`
- Modify: `static/style.css:526-680`

**Interfaces:**
- Consumes: `renderVideoPage()` tree data and breadcrumb state.
- Produces: a `video-library-toolbar`, live result count, semantic directory and media list rows, and responsive controls.

- [x] **Step 1: Write the failing test**

```python
def test_admin_video_browser_has_library_toolbar_and_semantic_rows():
    html = _admin_html()
    assert 'class="video-library-toolbar"' in html
    assert 'id="video-library-count"' in html
    assert 'class="dir-row video-directory-row"' in html
    assert 'class="video-media-row"' in html
```

- [x] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_admin_html.py::test_admin_video_browser_has_library_toolbar_and_semantic_rows -q`

Expected: FAIL because the current table has no video-library summary or semantic row treatments.

- [x] **Step 3: Write minimal implementation**

Add a toolbar containing search and a live count. Render directory rows as browsable folder cards within the table and media rows with a compact metadata line. Update the count for folders and files in the current view and add responsive CSS for the upgraded layout.

- [x] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_admin_html.py -q`

- [ ] **Step 5: Commit**

Run: `git add static/admin.html static/style.css tests/test_admin_html.py && git commit -m "style: improve local video browser"`
