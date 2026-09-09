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
async def test_translate_segments_uses_source_when_a_batch_never_translates(monkeypatch):
    """A permanently failed item falls back to its source text."""
    monkeypatch.setattr("core.translate.get_translate_engine", lambda **_: FailingEngine())

    result = await translate_segments(
        [{"start": 0.0, "end": 1.0, "text": "Hello"}],
        "zh-CN",
        mode="online",
        source_lang="en",
    )

    assert result == [{"start": 0.0, "end": 1.0, "text": "Hello"}]


@pytest.mark.asyncio
async def test_translate_segments_reports_completed_items_after_each_serial_batch(monkeypatch):
    """Translation reports cumulative item counts without parallelizing batches."""
    class SuccessfulEngine(TranslateEngine):
        def translate_batch(self, texts, *args, **kwargs):
            return [f"译文 {text}" for text in texts]

    updates = []
    monkeypatch.setattr("core.translate.get_translate_engine", lambda **_: SuccessfulEngine())
    segments = [
        {"start": float(index), "end": float(index + 1), "text": f"line {index}"}
        for index in range(6)
    ]

    result = await translate_segments(
        segments,
        "zh-CN",
        progress_callback=lambda completed, total: updates.append((completed, total)),
    )

    assert len(result) == 6
    assert updates == [(0, 6), (5, 6), (6, 6)]


@pytest.mark.asyncio
async def test_translate_segments_retries_invalid_item_before_next_batch_and_falls_back_to_source(monkeypatch):
    """A bad item is retried immediately and cannot restart completed batches."""
    calls = []

    class PartiallyInvalidEngine(TranslateEngine):
        def translate_batch(self, texts, *args, **kwargs):
            calls.append(list(texts))
            if texts == ["line 4"]:
                return ["line 4"]
            return ["line 4" if text == "line 4" else f"译文 {text}" for text in texts]

    monkeypatch.setattr("core.translate.get_translate_engine", lambda **_: PartiallyInvalidEngine())
    segments = [
        {"start": float(index), "end": float(index + 1), "text": f"line {index}"}
        for index in range(11)
    ]

    result = await translate_segments(segments, "zh-CN", source_lang="en")

    assert calls == [
        ["line 0", "line 1", "line 2", "line 3", "line 4"],
        ["line 4"],
        ["line 5", "line 6", "line 7", "line 8", "line 9"],
        ["line 10"],
    ]
    assert [segment["text"] for segment in result] == [
        "译文 line 0", "译文 line 1", "译文 line 2", "译文 line 3", "line 4",
        "译文 line 5", "译文 line 6", "译文 line 7", "译文 line 8", "译文 line 9",
        "译文 line 10",
    ]


@pytest.mark.asyncio
async def test_translate_segments_splits_failed_batches_before_falling_back_to_items(monkeypatch):
    """A failed batch is bisected so usable sub-batches are not retried item by item."""
    calls = []

    class SplitRecoveringEngine(TranslateEngine):
        def translate_batch(self, texts, *args, **kwargs):
            calls.append(list(texts))
            if len(texts) in {5, 3}:
                return None
            return [f"译文 {text}" for text in texts]

    monkeypatch.setattr("core.translate.get_translate_engine", lambda **_: SplitRecoveringEngine())
    segments = [
        {"start": float(index), "end": float(index + 1), "text": f"line {index}"}
        for index in range(5)
    ]

    result = await translate_segments(segments, "zh-CN", source_lang="en")

    assert calls == [
        ["line 0", "line 1", "line 2", "line 3", "line 4"],
        ["line 0", "line 1"],
        ["line 2", "line 3", "line 4"],
        ["line 2"],
        ["line 3", "line 4"],
    ]
    assert [segment["text"] for segment in result] == [
        "译文 line 0", "译文 line 1", "译文 line 2", "译文 line 3", "译文 line 4",
    ]


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
