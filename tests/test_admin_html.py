"""Regression checks for untrusted values rendered in the admin page."""

import json
import subprocess
from pathlib import Path


def _admin_html() -> str:
    return Path("static/admin.html").read_text()


def _run_admin_tree(videos: list[dict], current_path: str, root_dirs: list[str]) -> dict:
    """Execute the browser's real directory-tree functions without a browser."""
    script = """
const fs = require('fs');
const vm = require('vm');
const html = fs.readFileSync(process.argv[1], 'utf8');
const start = html.indexOf('        function getRelativePath(');
const end = html.indexOf('        function renderBreadcrumb()', start);
const context = {};
vm.runInNewContext(html.slice(start, end), context);
console.log(JSON.stringify(context.buildDirTree(
    JSON.parse(process.argv[2]), process.argv[3], JSON.parse(process.argv[4])
)));
"""
    result = subprocess.run(
        [
            "node",
            "-e",
            script,
            "static/admin.html",
            json.dumps(videos),
            current_path,
            json.dumps(root_dirs),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def _run_admin_subtitle_request(function_name: str, response: dict) -> dict:
    """Run the real single/batch request function with a failing API response."""
    script = """
const fs = require('fs');
const vm = require('vm');
const html = fs.readFileSync(process.argv[1], 'utf8');
const functionName = process.argv[2];
const response = JSON.parse(process.argv[3]);
const startMarker = functionName === 'batch' ? '        async function batchGenSub(' : '        async function genSub(';
const endMarker = functionName === 'batch' ? '        function toggleAllTasks(' : '        let pendingGenPath';
const start = html.indexOf(startMarker);
const end = html.indexOf(endMarker, start);
const source = html.slice(start, end);
const messages = [];
let refreshes = 0;
const context = {
    selectedVideos: new Set(['/media/episode.mkv']),
    fetch: async () => ({ ok: false, json: async () => response }),
    alert: message => messages.push(message),
    scanVideos: async () => { refreshes += 1; },
    loadTasks: () => { refreshes += 1; },
};
vm.runInNewContext(source, context);
const run = functionName === 'batch'
    ? context.batchGenSub(false, 'auto')
    : context.genSub('/media/episode.mkv', false, 'auto');
Promise.resolve(run).then(() => {
    console.log(JSON.stringify({ messages, refreshes }));
});
"""
    result = subprocess.run(
        ["node", "-e", script, "static/admin.html", function_name, json.dumps(response)],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


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


def test_admin_video_tree_scopes_a_monitored_root_to_its_contents():
    """Selecting tvdrama must not show direct children of other roots."""
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


def test_admin_single_subtitle_request_shows_an_api_write_error():
    """A failed write probe must be reported instead of refreshing as if queued."""
    result = _run_admin_subtitle_request(
        "single",
        {"detail": "Subtitle output directory is not writable", "task_ids": [], "skipped": 0},
    )

    assert result == {
        "messages": ["Subtitle output directory is not writable"],
        "refreshes": 0,
    }


def test_admin_batch_subtitle_request_shows_an_api_write_error():
    """A failed batch write probe must not claim that empty tasks were created."""
    result = _run_admin_subtitle_request(
        "batch",
        {"detail": "Subtitle output directory is not writable", "task_ids": [], "skipped": 0},
    )

    assert result == {
        "messages": ["Subtitle output directory is not writable"],
        "refreshes": 0,
    }
