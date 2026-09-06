"""Translation behavior regressions."""

import pytest

from core.translate import TranslateEngine, translate_segments
from core.translate.openai_api import _translate_batch_online


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


def test_online_translation_uses_only_standard_chat_completions_parameters(monkeypatch):
    """Disabling thinking must not add provider-specific fields to the SDK request."""
    request = {}

    class FakeCompletions:
        def create(self, **kwargs):
            request.update(kwargs)
            message = type("Message", (), {"content": '["你好"]'})()
            choice = type("Choice", (), {"message": message})()
            return type("Response", (), {"choices": [choice]})()

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = type("Chat", (), {"completions": FakeCompletions()})()

    monkeypatch.setattr("core.translate.openai_api.OpenAI", FakeOpenAI)

    result = _translate_batch_online(
        ["Hello"], "zh-CN", "https://example.test/v1", "test-key", "test-model", thinking=False,
    )

    assert result == ["你好"]
    assert "extra_body" not in request
    assert "thinking" not in request
