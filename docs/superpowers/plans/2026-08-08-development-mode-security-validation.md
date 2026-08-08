# Development Mode Security Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let explicit local development mode start without production credentials or related warnings, while retaining production validation.

**Architecture:** Keep policy in `validate_security_config`: explicit development mode returns before credential, session-secret, and TOTP checks. The module-level warning logic uses the same boundary.

**Tech Stack:** Python 3.10+, FastAPI, pytest, uv.

## Global Constraints

- Development mode is explicit: `DEVELOPMENT_MODE=true`.
- Production still rejects default credentials, missing or weak session secrets, and missing TOTP.
- Run tests through `uv run pytest`.

---

### Task 1: Define the development-mode validation boundary

**Files:**
- Modify: `tests/test_env_config.py`
- Modify: `env_config.py:74-92`

**Interfaces:**
- Consumes: `validate_security_config(username, password, session_secret, *, totp_secret, development_mode)`.
- Produces: no exception for local defaults with `development_mode=True`; `RuntimeError` for the same values in production mode.

- [x] **Step 1: Write the failing test**

```python
def test_validate_security_config_allows_insecure_values_with_development_override():
    validate_security_config("admin", "admin", "", totp_secret="", development_mode=True)
```

- [x] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_env_config.py::test_validate_security_config_allows_insecure_values_with_development_override -q`

Expected: FAIL because development mode currently validates default credentials and the session secret.

- [x] **Step 3: Write minimal implementation**

Place `if development_mode: return` at the beginning of `validate_security_config`.

- [x] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_env_config.py -q`

- [x] **Step 5: Commit**

Run: `git add env_config.py tests/test_env_config.py && git commit -m "fix: relax security checks in explicit development mode"`

### Task 2: Suppress expected development warnings

**Files:**
- Modify: `tests/test_env_config.py`
- Modify: `env_config.py:113-123`
- Modify: `.env.example:1-22`

**Interfaces:**
- Consumes: module-level `DEVELOPMENT_MODE`, `ADMIN_USERNAME`, `ADMIN_PASSWORD`, `TOTP_SECRET`, and `SESSION_SECRET`.
- Produces: no credential, TOTP, or session-secret warning in development mode; local-only environment flags documented.

- [x] **Step 1: Write the failing test**

Reload `env_config` with `DEVELOPMENT_MODE=true` and missing credentials/secrets. Capture logger records and assert that the three production-security warnings are absent.

- [x] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_env_config.py -q`

- [x] **Step 3: Write minimal implementation**

Guard the three warning conditions with `if not DEVELOPMENT_MODE`, and document `DEVELOPMENT_MODE=true` plus `SESSION_HTTPS_ONLY=false` as local-only in `.env.example`.

- [x] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_env_config.py -q`

- [x] **Step 5: Commit**

Run: `git add env_config.py .env.example tests/test_env_config.py && git commit -m "docs: clarify local development environment settings"`
