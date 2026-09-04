"""Task queue concurrency regressions."""

import threading
from concurrent.futures import ThreadPoolExecutor

from config import AppConfig
from core.task_manager import TaskManager


def test_active_task_creation_is_atomic_across_manager_instances(tmp_path):
    """Separate connections racing on one video cannot create two active rows."""
    db_path = str(tmp_path / "tasks.db")
    managers = [TaskManager(db_path), TaskManager(db_path)]
    start_barrier = threading.Barrier(2)

    def create(manager):
        start_barrier.wait(timeout=5)
        return manager.create_task_if_no_active("/media/movie.mp4")

    with ThreadPoolExecutor(max_workers=2) as executor:
        task_ids = list(executor.map(create, managers))

    assert sum(task_id is not None for task_id in task_ids) == 1
    assert managers[0].count_tasks(status="pending") == 1


def test_retrying_failed_task_with_an_active_task_returns_already_running(tmp_path):
    """Retry must not reactivate a failed task when its video is already active."""
    manager = TaskManager(str(tmp_path / "tasks.db"))
    failed_task_id = manager.create_task("/media/movie.mp4")
    manager._update_task(failed_task_id, status="failed", stage="failed")
    active_task_id = manager.create_task("/media/movie.mp4")

    result = manager.retry_task(failed_task_id)

    assert result == "already_running"
    assert manager.get_task(failed_task_id)["status"] == "failed"
    assert manager.get_task(active_task_id)["status"] == "pending"


def test_pipeline_subtitle_source_translates_existing_subtitle(tmp_path, monkeypatch):
    """source_type=subtitle 的任务：跳过 ASR，改为读取并翻译已有字幕文件。"""
    import json

    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    video = video_dir / "m.mkv"
    video.write_text("x")
    srt = video_dir / "m.en.srt"
    srt.write_text("1\n00:00:01,000 --> 00:00:02,000\nThis is the first subtitle line.\n", encoding="utf-8")

    cfg = AppConfig(temp_dir=str(tmp_path / "tmp"), target_language="zh-CN")
    monkeypatch.setattr("core.task_manager.get_config", lambda: cfg)
    monkeypatch.setattr("env_config.MODEL_IDLE_TIMEOUT", 0)

    source_segments = [{"start": 1.0, "end": 2.0, "text": "This is the first subtitle line."}]
    translate_source_lang = {}

    async def fake_resolve(video_path, source_path=None, source_index=None, work_dir=None):
        return {
            "kind": "external", "path": str(srt), "lang": "en",
            "source_path": str(srt), "source_index": None, "display": "m.en.srt",
        }

    async def fake_translate(segments, target_lang, **kwargs):
        translate_source_lang["lang"] = kwargs.get("source_lang", "")
        return [{"start": s["start"], "end": s["end"], "text": "你好"} for s in segments]

    monkeypatch.setattr("core.subtitle_source.resolve_subtitle_source", fake_resolve)
    monkeypatch.setattr("core.subtitle_source.read_subtitle_segments", lambda p: source_segments)
    monkeypatch.setattr("core.translate.translate_segments", fake_translate)
    monkeypatch.setattr("core.translate.set_translate_busy", lambda v: None)
    monkeypatch.setattr("core.utils.check_memory_limit", lambda: None)
    monkeypatch.setattr("core.subtitle_writer.generate_srt", lambda segs, path: True)
    monkeypatch.setattr("core.subtitle_writer.generate_bilingual_srt", lambda src, tgt, path: True)

    manager = TaskManager(str(tmp_path / "tasks.db"))
    task_id = manager.create_task(
        str(video), pipeline_type="video_subtitle",
        source_type="subtitle", source_path=str(srt), source_lang="auto",
    )
    task = manager.get_task(task_id)
    manager._execute_pipeline(task)

    task = manager.get_task(task_id)
    assert task["status"] == "done"
    assert task["source_segments"] is not None
    assert task["translated_segments"] is not None
    assert json.loads(task["translated_segments"])[0]["text"] == "你好"
    # 自动识别到英语（英文文本 + 文件名 hint 'en'）
    assert translate_source_lang["lang"] == "en"


def test_pipeline_subtitle_source_uses_explicit_source_lang(tmp_path, monkeypatch):
    """用户显式指定源语言时，不识别字幕内容，直接传给翻译引擎。"""
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    video = video_dir / "m.mkv"
    video.write_text("x")
    srt = video_dir / "m.srt"
    srt.write_text("1\n00:00:01,000 --> 00:00:02,000\nBonjour le monde aujourd'hui\n", encoding="utf-8")

    cfg = AppConfig(temp_dir=str(tmp_path / "tmp"), target_language="zh-CN")
    monkeypatch.setattr("core.task_manager.get_config", lambda: cfg)
    monkeypatch.setattr("env_config.MODEL_IDLE_TIMEOUT", 0)

    translate_source_lang = {}

    async def fake_resolve(video_path, source_path=None, source_index=None, work_dir=None):
        return {
            "kind": "external", "path": str(srt), "lang": "", "source_path": str(srt),
            "source_index": None, "display": "m.srt",
        }

    async def fake_translate(segments, target_lang, **kwargs):
        translate_source_lang["lang"] = kwargs.get("source_lang", "")
        return [{"start": s["start"], "end": s["end"], "text": "你好"} for s in segments]

    monkeypatch.setattr("core.subtitle_source.resolve_subtitle_source", fake_resolve)
    monkeypatch.setattr("core.subtitle_source.read_subtitle_segments", lambda p: [{"start": 1.0, "end": 2.0, "text": "Bonjour le monde aujourd'hui"}])
    monkeypatch.setattr("core.translate.translate_segments", fake_translate)
    monkeypatch.setattr("core.translate.set_translate_busy", lambda v: None)
    monkeypatch.setattr("core.utils.check_memory_limit", lambda: None)
    monkeypatch.setattr("core.subtitle_writer.generate_srt", lambda segs, path: True)
    monkeypatch.setattr("core.subtitle_writer.generate_bilingual_srt", lambda src, tgt, path: True)

    manager = TaskManager(str(tmp_path / "tasks.db"))
    task_id = manager.create_task(
        str(video), pipeline_type="video_subtitle",
        source_type="subtitle", source_path=str(srt), source_lang="fr",
    )
    manager._execute_pipeline(manager.get_task(task_id))

    assert translate_source_lang["lang"] == "fr"
