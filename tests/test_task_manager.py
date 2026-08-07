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
