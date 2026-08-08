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
