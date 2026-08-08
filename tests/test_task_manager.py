"""Task queue concurrency regressions."""

import threading
from concurrent.futures import ThreadPoolExecutor

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
