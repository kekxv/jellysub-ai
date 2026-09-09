"""Translation behavior regressions."""

import threading
import time

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


def test_online_translation_requests_are_serialized_across_threads(monkeypatch):
    """A provider that rejects concurrent requests receives one request at a time."""
    active_requests = 0
    max_active_requests = 0
    active_lock = threading.Lock()
    start = threading.Barrier(2)

    class FakeCompletions:
        def create(self, **kwargs):
            nonlocal active_requests, max_active_requests
            with active_lock:
                active_requests += 1
                max_active_requests = max(max_active_requests, active_requests)
            time.sleep(0.05)
            with active_lock:
                active_requests -= 1
            message = type("Message", (), {"content": '["你好"]'})()
            choice = type("Choice", (), {"message": message})()
            return type("Response", (), {"choices": [choice]})()

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = type("Chat", (), {"completions": FakeCompletions()})()

    monkeypatch.setattr("core.translate.openai_api.OpenAI", FakeOpenAI)

    def translate():
        start.wait(timeout=1)
        return _translate_batch_online(
            ["Hello"], "zh-CN", "https://example.test/v1", "test-key", "test-model", thinking=False,
        )

    threads = [threading.Thread(target=translate) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=1)

    assert all(not thread.is_alive() for thread in threads)
    assert max_active_requests == 1
