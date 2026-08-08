"""Regression checks for untrusted values rendered in the admin page."""

from pathlib import Path


def _admin_html() -> str:
    return Path("static/admin.html").read_text()


def test_admin_task_renderer_escapes_webhook_item_name():
    """Webhook names cannot become task-card markup."""
    html = _admin_html()

    assert "${escHtml(t.item_name || '任务')}" in html
    assert "${t.item_name || '任务'}" not in html


def test_admin_subtitle_overlay_escapes_segment_text():
    """Subtitle text remains data while the bilingual layout markup stays static."""
    html = _admin_html()

    assert "escHtml(t.text)" in html
    assert "escHtml(s.text)" in html
    assert "innerHTML = overlayHtml" in html


def test_admin_inline_handlers_escape_paths_and_names():
    """A quote or line separator in a path cannot terminate an inline handler."""
    html = _admin_html()

    assert "function escapeInlineHandler(value)" in html
    assert "escapeInlineHandler(v.path)" in html
    assert "escapeInlineHandler(v.name)" in html
    assert ".replace(/\\\\/g, '\\\\\\\\')" in html
    assert r"replace(/\u2028/g" in html
    assert r"'\\u2028'" in html


def test_admin_has_task_first_operational_overview():
    html = _admin_html()

    assert 'id="queue-summary"' in html
    assert 'id="active-task-list"' in html
    assert 'id="task-stat-processing"' in html


def test_admin_renders_task_stage_progress_and_refreshes_active_work():
    html = _admin_html()

    assert "function stageLabel(stage)" in html
    assert 'class="progress-bar"' in html
    assert "setInterval(refreshOperationalOverview, 5000)" in html


def test_admin_task_history_uses_scannable_semantic_task_rows():
    """History rows must expose state, progress, and media identity at a glance."""
    html = _admin_html()

    assert 'class="task-history-summary"' in html
    assert 'task-history-row task-history-${status.className}' in html
    assert 'task-status-label ${status.className}' in html
    assert 'class="task-history-path"' in html


def test_admin_task_history_avoids_inline_layout_for_task_list():
    """The redesigned history list needs reusable responsive layout hooks."""
    html = _admin_html()

    assert 'id="task-grid" class="task-history-list"' in html
    assert 'id="task-pagination" class="pagination task-history-pagination"' in html


def test_admin_video_browser_restores_directory_from_encoded_hash():
    """Refreshing a directory URL must restore its encoded relative path."""
    html = _admin_html()

    assert "function videoPathFromHash()" in html
    assert "encodeURIComponent(relativePath)" in html
    assert "window.addEventListener('hashchange', restoreVideoPathFromHash)" in html


def test_admin_video_browser_has_library_toolbar_and_semantic_rows():
    """The media browser must distinguish folder navigation from video actions."""
    html = _admin_html()

    assert 'class="video-library-toolbar"' in html
    assert 'id="video-library-count"' in html
    assert 'class="dir-row video-directory-row"' in html
    assert 'class="video-media-row"' in html
