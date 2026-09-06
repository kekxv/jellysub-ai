"""Translation behavior regressions."""

import pytest

from core.translate import TranslateEngine, translate_segments


class FailingEngine(TranslateEngine):
    """An engine that cannot produce a translation."""

    def translate_batch(self, *args, **kwargs):
        return None


@pytest.mark.asyncio
async def test_translate_segments_returns_none_when_a_batch_never_translates(monkeypatch):
    """Failed translations must not be silently replaced with source subtitles."""
    monkeypatch.setattr("core.translate.get_translate_engine", lambda **_: FailingEngine())

    result = await translate_segments(
        [{"start": 0.0, "end": 1.0, "text": "Hello"}],
        "zh-CN",
        mode="online",
        source_lang="en",
    )

    assert result is None
