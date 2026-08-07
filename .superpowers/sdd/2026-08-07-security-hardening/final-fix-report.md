# Final review fix report

Date: 2026-08-07

Base commit: `41df4c4` (`docs: record security hardening verification`)

## Findings resolved

1. Published credential placeholders
   - `validate_security_config()` now rejects every exact administrator and session placeholder previously published in `.env.example` and `README.md`.
   - `.env.example` leaves required credentials empty, and README commands generate secrets at deployment time rather than publishing accepted literals.
   - Regression coverage includes all three published username/password pairs and both published session-secret strings.

2. Atomic active-task creation
   - SQLite now has a partial unique index on `video_path` for `pending`/`processing` rows. Existing duplicate active rows are migrated deterministically by retaining the oldest and marking later rows failed before the index is created.
   - `create_task_if_no_active()` turns only that uniqueness conflict into an `already_running` result; webhook, manual, test, and batch creation paths use the atomic API.
   - Tests cover synchronized concurrent webhook requests and two independent `TaskManager` instances sharing one database.

3. Stable container identity
   - CPU and GPU images both create `app` with explicit UID/GID `10001:10001`.
   - A regression test extracts and compares both numeric identities.
   - README documents `/data` and writable media ownership for named-volume/bind-mount use and CPU/GPU migration.

4. Disabled-webhook guidance
   - The unset-`WEBHOOK_SECRET` startup log now states that the endpoint is disabled.
   - README now matches the enforced HTTP 503 behavior and the implemented raw-body HMAC-SHA256 contract.

## Test-first evidence

- Initial focused RED command:
  `uv run pytest tests/test_env_config.py tests/test_api.py::test_concurrent_webhooks_create_only_one_active_task tests/test_dockerfiles.py::test_cpu_and_gpu_images_use_the_same_explicit_app_uid_and_gid -q`
  Result: 7 failed, 1 passed. Failures showed accepted published placeholders, two concurrent `accepted` responses, and missing explicit Docker IDs.
- Cross-connection RED command:
  `uv run pytest tests/test_task_manager.py -q`
  Result: 1 failed because `create_task_if_no_active` did not exist.
- Focused GREEN command:
  `uv run pytest tests/test_env_config.py tests/test_task_manager.py tests/test_api.py::test_webhook_skips_duplicate_active_task tests/test_api.py::test_concurrent_webhooks_create_only_one_active_task tests/test_dockerfiles.py::test_cpu_and_gpu_images_use_the_same_explicit_app_uid_and_gid -q`
  Result: 10 passed.

## Final verification evidence

- Review-focused suite:
  `uv run pytest tests/test_env_config.py tests/test_task_manager.py tests/test_api.py tests/test_admin_html.py tests/test_dockerfiles.py -q`
  Result: 43 passed.
- Full suite (no model tests executed):
  `uv run pytest -q`
  Result: 68 passed, 3 failed, 1 deselected. The three failures are the documented unchanged baseline/environment cases: two missing-`ffmpeg` real-media failures and `test_find_existing_subtitle_found`.
- Full suite excluding only those three documented failures:
  `uv run pytest -q --deselect=tests/test_real_media.py::test_extract_audio_real_mp4 --deselect=tests/test_real_media.py::test_task_pipeline_with_real_mp4 --deselect=tests/test_subtitle_checker.py::test_find_existing_subtitle_found`
  Result: 68 passed, 4 deselected.
- Docker static validation:
  `docker build --check -f Dockerfile .`
  Result: exit 0, no warnings.
  `docker build --check -f Dockerfile-Gpu .`
  Result: exit 0, no warnings.
- Lock validation:
  `uv lock --check` and `(cd docker/gpu && uv lock --check)` both exited 0.
- Patch validation:
  `git diff --check` exited 0.

No model pipeline or model download was run during this fix wave.
