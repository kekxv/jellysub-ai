# Residual Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `subagent-driven-development` to implement this plan task-by-task.

**Goal:** Close remaining deployment, model-loading, and test-isolation risks without changing subtitle generation behavior.

**Architecture:** Put security policy in environment configuration, limit model code to reviewed revision-pinned models, and make configuration persistence crash-safe. Tests exercise rate limits, origins, startup policy, atomic files, and forbid test model downloads.

**Tech Stack:** FastAPI, Pydantic, pytest, uv.

## Global Constraints

- Use `uv` for every Python command.
- Automated tests must not load or download models; explicit real-model tests use `MODEL_SOURCE=modelscope`.
- New behavior starts with a focused failing test before production edits.

### Task 1: Authentication and browser boundary hardening

**Files:** Modify `env_config.py`, `main.py`, `tests/test_env_config.py`, `tests/test_api.py`.

- [ ] Write tests requiring TOTP in production, returning HTTP 429 after six failed logins, and denying unlisted CORS origins.
- [ ] Run `uv run pytest tests/test_env_config.py tests/test_api.py -q` and observe RED.
- [ ] Add a lock-protected login limiter keyed by client address/username, explicit CORS origin parsing, and production TOTP policy with an explicit development override.
- [ ] Run focused tests until GREEN and commit `fix: harden login and browser boundaries`.

### Task 2: Trusted local model loading

**Files:** Modify `env_config.py`, `core/translate/local.py`, `core/asr/sensevoice.py`, relevant ASR loaders, `tests/test_model_security.py`.

- [ ] Write tests rejecting unapproved model IDs/revisions and requiring `trust_remote_code=False`.
- [ ] Run `uv run pytest tests/test_model_security.py -q` and observe RED.
- [ ] Add allowlisted immutable model revisions and disable remote model code, preserving ModelScope for explicit real-model tests.
- [ ] Run focused tests until GREEN and commit `fix: constrain local model loading`.

### Task 3: Atomic configuration and test isolation

**Files:** Modify `config.py`, `main.py`, `tests/test_config.py`, `tests/test_api.py`.

- [ ] Write tests asserting atomic config replacement and no preload thread/model request from API config saves in tests.
- [ ] Run `uv run pytest tests/test_config.py tests/test_api.py -q` and observe RED.
- [ ] Write same-directory temporary configs, fsync, and replace atomically; inject or guard preload so tests cannot fetch models.
- [ ] Run focused tests until GREEN and commit `fix: make configuration saves atomic`.

### Task 4: Verification

- [ ] Run `uv run pytest -q`, Docker static checks, and `git diff --check`.
- [ ] Record host-ffmpeg failures separately from regressions and commit the plan verification update.
