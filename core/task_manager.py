"""SQLite-backed task queue with single worker thread for subtitle generation pipelines."""

import json
import logging
import os
import sqlite3
import threading
import time
from pathlib import Path

from config import get_config

logger = logging.getLogger("uvicorn.error")

# Pipeline stage order
_PIPELINE_STAGES = [
    "extracting_audio",
    "asr",
    "translating",
    "writing_srt",
]

_STAGE_PROGRESS = {
    "extracting_audio": 20,
    "asr": 50,
    "translating": 80,
    "writing_srt": 95,
    "done": 100,
}


def _cleanup_tmp_files(tmp_dir: Path, audio_path: str):
    """清理任务相关的临时文件（音频 + VAD 分块）。"""
    try:
        # 删除主临时音频
        Path(audio_path).unlink(missing_ok=True)
    except Exception:
        pass
    # 清理同一批次的 VAD 分块文件（同 SHA256 前缀）
    try:
        prefix = Path(audio_path).stem  # e.g. "task_abc123def456_task"
        for f in tmp_dir.glob(f"{prefix}*"):
            f.unlink(missing_ok=True)
    except Exception:
        pass


def cleanup_all_tmp(tmp_dir: str = "./tmp"):
    """清理整个 tmp 目录（应用关闭时调用）。"""
    path = Path(tmp_dir)
    if not path.exists():
        return
    for f in path.iterdir():
        try:
            if f.is_file():
                f.unlink()
        except Exception:
            pass
    try:
        path.rmdir()
    except OSError:
        pass  # 目录非空时忽略
    logger.info("Tmp directory cleaned up: %s", tmp_dir)


class TaskManager:
    """SQLite-backed task queue with single worker thread."""

    def __init__(self, db_path: str = "tasks.db"):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._running = False
        self._worker_thread: threading.Thread | None = None
        self._init_db()

    # ------------------------------------------------------------------ #
    #  DB helpers
    # ------------------------------------------------------------------ #

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _dict_row(self, row: sqlite3.Row) -> dict:
        return dict(row)

    def _init_db(self):
        conn = self._get_conn()
        cur = conn.cursor()

        # Check if tasks table exists
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='tasks'")
        table_exists = cur.fetchone() is not None

        if not table_exists:
            # Fresh install: create full schema
            cur.execute("""
                CREATE TABLE tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_path TEXT NOT NULL,
                    item_id TEXT,
                    item_type TEXT,
                    item_name TEXT,
                    status TEXT DEFAULT 'pending',
                    progress INTEGER DEFAULT 0,
                    error_message TEXT,
                    pipeline_type TEXT DEFAULT 'webhook',
                    stage TEXT DEFAULT 'pending',
                    retry_count INTEGER DEFAULT 0,
                    max_retries INTEGER DEFAULT 1,
                    source_segments TEXT,
                    translated_segments TEXT,
                    started_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
        else:
            # Migration: add missing columns
            columns_to_add = [
                ("pipeline_type", "TEXT DEFAULT 'webhook'"),
                ("stage", "TEXT DEFAULT 'pending'"),
                ("retry_count", "INTEGER DEFAULT 0"),
                ("max_retries", "INTEGER DEFAULT 1"),
                ("source_segments", "TEXT"),
                ("translated_segments", "TEXT"),
                ("started_at", "TIMESTAMP"),
                ("asr_language", "TEXT DEFAULT 'auto'"),
            ]
            for col_name, col_def in columns_to_add:
                try:
                    cur.execute(f"ALTER TABLE tasks ADD COLUMN {col_name} {col_def}")
                except sqlite3.OperationalError:
                    pass  # Column already exists

        # Create task_stages table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS task_stages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id INTEGER NOT NULL,
                stage TEXT NOT NULL,
                started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP,
                error_message TEXT,
                FOREIGN KEY (task_id) REFERENCES tasks(id) ON DELETE CASCADE
            )
        """)

        conn.commit()
        conn.close()

    # ------------------------------------------------------------------ #
    #  CRUD
    # ------------------------------------------------------------------ #

    def create_task(
        self,
        video_path: str,
        item_id: str = "",
        item_type: str = "",
        item_name: str = "",
        pipeline_type: str = "webhook",
        asr_language: str = "auto",
    ) -> int:
        with self._lock:
            conn = self._get_conn()
            cur = conn.cursor()
            cur.execute(
                """INSERT INTO tasks
                   (video_path, item_id, item_type, item_name, pipeline_type, asr_language, status, stage)
                   VALUES (?, ?, ?, ?, ?, ?, 'pending', 'pending')""",
                (video_path, item_id, item_type, item_name, pipeline_type, asr_language),
            )
            task_id = cur.lastrowid
            conn.commit()
            conn.close()
        logger.info("Task %d created: %s (%s)", task_id, video_path, pipeline_type)
        return task_id

    def get_task(self, task_id: int) -> dict | None:
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute("SELECT * FROM tasks WHERE id = ?", (task_id,))
        row = cur.fetchone()
        conn.close()
        return self._dict_row(row) if row else None

    def list_tasks(
        self,
        status: str | None = None,
        pipeline_type: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict]:
        conn = self._get_conn()
        cur = conn.cursor()
        query = "SELECT * FROM tasks WHERE 1=1"
        params: list = []
        if status:
            query += " AND status = ?"
            params.append(status)
        if pipeline_type:
            query += " AND pipeline_type = ?"
            params.append(pipeline_type)
        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        cur.execute(query, params)
        rows = cur.fetchall()
        conn.close()
        return [self._dict_row(r) for r in rows]

    def count_tasks(
        self,
        status: str | None = None,
        pipeline_type: str | None = None,
    ) -> int:
        conn = self._get_conn()
        cur = conn.cursor()
        query = "SELECT COUNT(*) FROM tasks WHERE 1=1"
        params: list = []
        if status:
            query += " AND status = ?"
            params.append(status)
        if pipeline_type:
            query += " AND pipeline_type = ?"
            params.append(pipeline_type)
        cur.execute(query, params)
        count = cur.fetchone()[0]
        conn.close()
        return count

    def get_latest_by_video_path(self, path: str) -> dict | None:
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM tasks WHERE video_path = ? ORDER BY created_at DESC LIMIT 1",
            (path,),
        )
        row = cur.fetchone()
        conn.close()
        return self._dict_row(row) if row else None

    def get_latest_by_type(self, pipeline_type: str) -> dict | None:
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM tasks WHERE pipeline_type = ? ORDER BY created_at DESC LIMIT 1",
            (pipeline_type,),
        )
        row = cur.fetchone()
        conn.close()
        return self._dict_row(row) if row else None

    def get_pending_tasks(self) -> list[dict]:
        """Get pending tasks (retry_count <= max_retries), oldest first."""
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM tasks WHERE status = 'pending' AND retry_count <= max_retries ORDER BY created_at ASC"
        )
        rows = cur.fetchall()
        conn.close()
        return [self._dict_row(r) for r in rows]

    def set_source_segments(self, task_id: int, segments: list[dict]):
        with self._lock:
            conn = self._get_conn()
            cur = conn.cursor()
            cur.execute(
                "UPDATE tasks SET source_segments = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (json.dumps(segments, ensure_ascii=False), task_id),
            )
            conn.commit()
            conn.close()

    def set_translated_segments(self, task_id: int, segments: list[dict]):
        with self._lock:
            conn = self._get_conn()
            cur = conn.cursor()
            cur.execute(
                "UPDATE tasks SET translated_segments = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (json.dumps(segments, ensure_ascii=False), task_id),
            )
            conn.commit()
            conn.close()

    def delete_task(self, task_id: int):
        with self._lock:
            conn = self._get_conn()
            cur = conn.cursor()
            cur.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
            conn.commit()
            conn.close()

    def delete_tasks(self, task_ids: list[int]):
        """批量删除任务记录，不删除字幕文件。"""
        if not task_ids:
            return
        with self._lock:
            conn = self._get_conn()
            cur = conn.cursor()
            cur.executemany("DELETE FROM tasks WHERE id = ?", [(tid,) for tid in task_ids])
            conn.commit()
            conn.close()

    def create_tasks_batch(
        self,
        video_paths: list[str],
        pipeline_type: str = "video_subtitle",
        asr_language: str = "auto",
    ) -> tuple[list[int], int]:
        """批量创建字幕任务，跳过已有 pending/processing 状态的视频。
        返回 (创建的任务 ID 列表, 跳过的数量)。
        """
        created = []
        skipped = 0
        for path in video_paths:
            existing = self.get_latest_by_video_path(path)
            if existing and existing["status"] in ("pending", "processing"):
                skipped += 1
                continue
            task_id = self.create_task(video_path=path, pipeline_type=pipeline_type, asr_language=asr_language)
            created.append(task_id)
        return created, skipped

    def retry_task(self, task_id: int):
        """Reset a failed task to pending for manual retry."""
        with self._lock:
            conn = self._get_conn()
            cur = conn.cursor()
            cur.execute(
                """UPDATE tasks SET
                    status = 'pending', stage = 'pending',
                    retry_count = 0, error_message = NULL,
                    progress = 0, updated_at = CURRENT_TIMESTAMP
                   WHERE id = ?""",
                (task_id,),
            )
            conn.commit()
            conn.close()
        logger.info("Task %d reset for retry", task_id)

    # ------------------------------------------------------------------ #
    #  Internal update helpers (called from worker thread, safe without lock)
    # ------------------------------------------------------------------ #

    def _update_task(self, task_id: int, **kwargs):
        if not kwargs:
            return
        fields = ", ".join(f"{k} = ?" for k in kwargs)
        values = list(kwargs.values()) + [task_id]
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute(
            f"UPDATE tasks SET {fields}, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            values,
        )
        conn.commit()
        conn.close()

    def _record_stage(self, task_id: int, stage: str, error: str | None = None):
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO task_stages (task_id, stage, completed_at, error_message)
               VALUES (?, ?, CURRENT_TIMESTAMP, ?)""",
            (task_id, stage, error),
        )
        conn.commit()
        conn.close()

    # ------------------------------------------------------------------ #
    #  Worker thread
    # ------------------------------------------------------------------ #

    def start_worker(self, preload_hook=None):
        self._preload_hook = preload_hook
        self._running = True
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()
        logger.info("Task worker started")

    def stop_worker(self):
        self._running = False
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=30)
        logger.info("Task worker stopped")

    def _worker_loop(self):
        # 在 worker 线程内先预加载模型，再处理任务
        if self._preload_hook:
            self._preload_hook()
        while self._running:
            tasks = self.get_pending_tasks()
            if not tasks:
                time.sleep(1)
                continue
            task = tasks[0]  # FIFO
            self._execute_pipeline(task)

    def _execute_pipeline(self, task: dict):
        task_id = task["id"]
        video_path = task["video_path"]
        cfg = get_config()
        media_dir = str(Path(video_path).parent)
        media_name = Path(video_path).stem
        tmp_dir = Path(cfg.temp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        # 用时间戳+路径SHA256作为临时文件名，避免中文/特殊字符问题
        import hashlib as _hashlib
        name_hash = _hashlib.sha256(video_path.encode("utf-8")).hexdigest()[:12]
        audio_path = str(tmp_dir / f"task_{name_hash}_task.wav")

        logger.info("Executing task %d: %s", task_id, video_path)
        self._update_task(task_id, status="processing", started_at=time.strftime("%Y-%m-%d %H:%M:%S"))

        try:
            # --- Stage 1: Extract audio ---
            self._update_task(task_id, stage="extracting_audio", progress=_STAGE_PROGRESS["extracting_audio"])
            self._record_stage(task_id, "extracting_audio")

            import asyncio
            from core.audio import extract_audio
            from core.utils import check_memory_limit

            check_memory_limit()
            ok = asyncio.new_event_loop().run_until_complete(
                extract_audio(video_path, audio_path)
            )
            if not ok:
                raise RuntimeError("Audio extraction failed")

            # --- Stage 2: ASR ---
            # 重试时检查是否已有 ASR 结果，跳过耗时的 ASR 步骤
            segments = None
            detected_lang = ""
            if task.get("source_segments"):
                import json as _json
                segments = _json.loads(task["source_segments"])
                if not segments:
                    logger.info("Task %d: previous ASR found no speech, skipping", task_id)
                    self._update_task(task_id, stage="done", progress=100, status="done")
                    self._record_stage(task_id, "done")
                    return
                logger.info("Task %d: ASR result already exists, skipping ASR (%d segments)", task_id, len(segments))
            else:
                self._update_task(task_id, stage="asr", progress=_STAGE_PROGRESS["asr"])
                from core.asr import run_asr, set_asr_busy
                from core.utils import check_memory_limit

                check_memory_limit()
                set_asr_busy(True)
                try:
                    segments, detected_lang = run_asr(
                        audio_path,
                        mode=cfg.asr_mode,
                        engine=cfg.asr_engine,
                        model_name=cfg.asr_model,
                        asr_language=task["asr_language"] if task.get("asr_language") else cfg.asr_language,
                        api_url=cfg.asr_api_url,
                        api_key=cfg.asr_api_key,
                        model_online=cfg.asr_model_online,
                        use_vad=True,
                        vad_min_silence_ms=500,
                    )

                finally:
                    set_asr_busy(False)

                if not segments:
                    logger.info("Task %d: no speech detected, skipping ASR and translation", task_id)
                    self._update_task(task_id, stage="done", progress=100, status="done")
                    self._record_stage(task_id, "done")
                    return  # 无语音，直接结束
                self.set_source_segments(task_id, segments)

            # --- 阶段间隙：强制释放 ASR 占用的显存，为翻译留出空间 ---
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()

            # --- Stage 3: Translate ---
            # 重试时检查是否已有翻译结果
            translated = None
            if task.get("translated_segments"):
                import json as _json
                translated = _json.loads(task["translated_segments"])
                logger.info("Task %d: translation result already exists (%d segments)", task_id, len(translated))
            else:
                self._update_task(task_id, stage="translating", progress=_STAGE_PROGRESS["translating"])
                from core.translate import translate_segments, set_translate_busy
                from core.utils import check_memory_limit

                check_memory_limit()
                set_translate_busy(True)
                try:
                    translated = asyncio.new_event_loop().run_until_complete(
                        translate_segments(
                            segments,
                            cfg.target_language,
                            mode=cfg.translate_mode,
                            api_url=cfg.translate_api_url,
                            api_key=cfg.translate_api_key,
                            model=cfg.translate_model,
                            model_local=cfg.translate_model_local,
                            thinking=cfg.translate_thinking,
                            prompt_format=cfg.translate_prompt_format,
                            source_lang=detected_lang,
                        )
                    )
                finally:
                    set_translate_busy(False)

                if not translated:
                    raise RuntimeError("Translation failed")
                self.set_translated_segments(task_id, translated)

            # --- Stage 4: Write SRT ---
            self._update_task(task_id, stage="writing_srt", progress=_STAGE_PROGRESS["writing_srt"])

            from core.subtitle_writer import generate_srt, generate_bilingual_srt

            target_ext = f".{cfg.target_language}.srt"
            target_path = os.path.join(media_dir, f"{media_name}.default{target_ext}")
            generate_srt(translated, target_path)

            bilingual_path = os.path.join(media_dir, f"{media_name}.bilingual{target_ext}")
            generate_bilingual_srt(segments, translated, bilingual_path)

            # --- Done ---
            self._update_task(task_id, stage="done", progress=_STAGE_PROGRESS["done"], status="done")
            self._record_stage(task_id, "done")

            if task["pipeline_type"] == "webhook":
                # Refresh Jellyfin
                import asyncio
                from core.jellyfin_api import JellyfinClient
                client = JellyfinClient(cfg.jellyfin_url, cfg.jellyfin_api_key)
                asyncio.new_event_loop().run_until_complete(client.refresh_item(task["item_id"]))

            logger.info("Task %d completed: %d segments", task_id, len(segments))

        except Exception as e:
            self._record_stage(task_id, task.get("stage", "unknown"), error=str(e))
            logger.exception("Task %d failed at stage %s", task_id, task.get("stage"))
            retry_count = task.get("retry_count", 0) + 1
            if retry_count <= task.get("max_retries", 1):
                self._update_task(
                    task_id,
                    status="pending",
                    stage="pending",
                    retry_count=retry_count,
                    progress=0,
                    error_message=str(e),
                )
                logger.info("Task %d auto-retry (attempt %d)", task_id, retry_count)
            else:
                self._update_task(
                    task_id,
                    status="failed",
                    stage="failed",
                    retry_count=retry_count,
                    error_message=str(e),
                )

        finally:
            _cleanup_tmp_files(tmp_dir, audio_path)
            # 检查模型空闲超时并释放，降低长时运行内存压力
            if cfg := get_config():
                from env_config import MODEL_IDLE_TIMEOUT
                if MODEL_IDLE_TIMEOUT > 0:
                    from core.asr import check_asr_idle
                    from core.translate import check_model_idle
                    check_asr_idle(MODEL_IDLE_TIMEOUT)
                    check_model_idle(MODEL_IDLE_TIMEOUT)
