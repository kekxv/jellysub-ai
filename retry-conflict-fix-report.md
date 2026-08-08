# Retry conflict fix report

- Regression tests added for `TaskManager.retry_task()` and `POST /api/tasks/{task_id}/retry` when a failed task shares a video path with an active task.
- The task manager now returns `already_running` after rolling back the unique-index conflict; the API returns HTTP 409 with a clear detail message.
- Verification: `uv run pytest tests/test_env_config.py tests/test_task_manager.py tests/test_api.py tests/test_admin_html.py tests/test_dockerfiles.py -q` — 45 passed.
- Implementation commit: `73f1061 fix retry conflict for active videos`.
