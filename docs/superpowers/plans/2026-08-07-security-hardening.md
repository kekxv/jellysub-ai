# Security Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the remotely exploitable default deployment paths, make webhook/task inputs safe, and make the production images reproducible and cache-efficient.

**Architecture:** Security settings remain environment-driven, but production startup rejects missing or known-unsafe credentials. Webhooks become opt-in authenticated requests, with their body protected by HMAC and their mapped path constrained to configured media directories. The admin UI treats every task, filename, configuration, and subtitle string as untrusted text. The Docker runtime receives only application code and an immutable uv-resolved dependency environment; mutable state belongs in `/data`.

**Tech Stack:** Python 3.12, FastAPI/Starlette, Pydantic v2, pytest, uv, Docker BuildKit.

## Global Constraints

- Manage local Python environments and commands with `uv`; do not use `pip` or a global virtualenv.
- Real-model tests, if explicitly run, use `MODEL_SOURCE=modelscope`; regular tests mock model/network work.
- Production Docker images must not contain credentials, `config.json`, test media, tests, or a root runtime process.
- Webhook verification uses `HMAC-SHA256(WEBHOOK_SECRET, raw_request_body)` in `X-Jellyfin-Signature`; an unset secret disables the endpoint.
- Every production behavior change starts with a focused failing test and ends with its focused test plus the full suite.

---

### Task 1: Enforce safe runtime credentials and repair authentication tests

**Files:**
- Modify: `env_config.py`, `main.py`, `Dockerfile`, `Dockerfile-Gpu`, `.env.example`, `README.md`
- Modify: `tests/test_api.py`
- Create: `tests/test_env_config.py`

**Interfaces:**
- Produces `validate_security_config(username: str, password: str, session_secret: str) -> None` in `env_config.py`.
- `main.lifespan()` calls `validate_security_config` before starting the worker.
- `main._credential_hash(username, password)` remains the only browser-compatible login proof format.

- [ ] **Step 1: Write failing validation and login tests**

```python
def test_validate_security_config_rejects_default_admin_credentials():
    with pytest.raises(RuntimeError, match="ADMIN_PASSWORD"):
        validate_security_config("admin", "admin", "a" * 32)

def test_login_accepts_browser_credential_hash(client):
    response = client.post("/login", json={
        "username": "tester",
        "password": _credential_hash("tester", "strong-password"),
        "totp_code": "",
    })
    assert response.json() == {"status": "ok"}
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `uv run pytest tests/test_env_config.py tests/test_api.py::test_login_accepts_browser_credential_hash -q`

Expected: FAIL because no validation helper exists and the API test fixture currently submits plaintext.

- [ ] **Step 3: Implement the smallest safe configuration contract**

```python
def validate_security_config(username: str, password: str, session_secret: str) -> None:
    if not username or not password or (username == "admin" and password == "admin"):
        raise RuntimeError("Set non-default ADMIN_USERNAME and ADMIN_PASSWORD")
    if len(session_secret) < 32 or session_secret == "change_me_in_production":
        raise RuntimeError("Set a random SESSION_SECRET of at least 32 characters")
```

Remove credential values from Docker `ENV`, call the helper in lifespan, set secure cookie behavior from an explicit environment flag, and update tests to generate the browser hash.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `uv run pytest tests/test_env_config.py tests/test_api.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add env_config.py main.py Dockerfile Dockerfile-Gpu .env.example README.md tests/test_env_config.py tests/test_api.py
git commit -m "fix: require secure runtime credentials"
```

### Task 2: Authenticate and constrain webhook work

**Files:**
- Modify: `main.py`, `core/task_manager.py`, `tests/test_api.py`

**Interfaces:**
- Produces `_verify_webhook_signature(raw_body: bytes, signature: str) -> bool` in `main.py`.
- Produces `TaskManager.has_active_task(video_path: str) -> bool`.
- `/webhook` returns HTTP 503 without `WEBHOOK_SECRET`, HTTP 401 for an invalid HMAC, 404 for a nonexistent mapped file, and 403 for a mapped path outside `video_dirs`.

- [ ] **Step 1: Write failing webhook security tests**

```python
def test_webhook_rejects_when_secret_is_unset(client, payload):
    with patch("main.WEBHOOK_SECRET", ""):
        assert client.post("/webhook", json=payload).status_code == 503

def test_webhook_accepts_only_body_hmac(client, payload):
    raw = json.dumps(payload, separators=(",", ":")).encode()
    signature = hmac.new(b"test-secret", raw, hashlib.sha256).hexdigest()
    with patch("main.WEBHOOK_SECRET", "test-secret"):
        assert client.post("/webhook", content=raw, headers={"Content-Type": "application/json", "X-Jellyfin-Signature": signature}).status_code == 200
```

Add cases for a changed body, a path outside configured video roots, a nonexistent file, and a duplicate active task.

- [ ] **Step 2: Run focused tests and verify RED**

Run: `uv run pytest tests/test_api.py -k webhook -q`

Expected: FAIL because the endpoint currently accepts empty secrets and compares a fixed SHA-256 value.

- [ ] **Step 3: Implement HMAC, path validation, and duplicate prevention**

Read the cached raw body, compare `hmac.new(secret.encode(), raw_body, hashlib.sha256).hexdigest()` with `compare_digest`, then validate the mapped local path before `ffprobe` or task creation. Query pending/processing tasks by video path before inserting a webhook task.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `uv run pytest tests/test_api.py -k webhook -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add main.py core/task_manager.py tests/test_api.py
git commit -m "fix: harden webhook authentication and task creation"
```

### Task 3: Eliminate untrusted HTML execution and bound API input

**Files:**
- Modify: `static/admin.html`, `main.py`, `tests/test_api.py`
- Create: `tests/test_admin_html.py`

**Interfaces:**
- `escHtml(value)` escapes `&`, `<`, `>`, `"`, and `'` before every HTML interpolation.
- `escapeInlineHandler(value)` also escapes backslashes and line separators before constructing an inline JavaScript string.
- Task list/query parameters accept `1 <= limit <= 100`, `offset >= 0`; batch task and subtitle requests accept at most 100 IDs/paths.

- [ ] **Step 1: Write failing XSS and bounds tests**

```python
def test_admin_task_renderer_escapes_webhook_item_name():
    html = Path("static/admin.html").read_text()
    assert "${escHtml(t.item_name || '任务')}" in html
    assert "${t.item_name || '任务'}" not in html

def test_tasks_limit_is_bounded(client):
    response = client.get("/api/tasks?limit=101")
    assert response.status_code == 422
```

Add source checks covering escaped subtitle overlay text and backslash-safe inline handler paths.

- [ ] **Step 2: Run focused tests and verify RED**

Run: `uv run pytest tests/test_admin_html.py tests/test_api.py -k 'limit or batch' -q`

Expected: FAIL because item names and player subtitles reach `innerHTML` unescaped and limits are unrestricted.

- [ ] **Step 3: Implement escaping and Pydantic bounds**

Escape task fields, configuration-backed values, subtitle overlay text, file paths/names in inline handlers, and use `Query`/`Field` constraints for pagination and batches. Keep the intentional layout markup (`<br>` and `<small>`) static; only data is escaped.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `uv run pytest tests/test_admin_html.py tests/test_api.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add static/admin.html main.py tests/test_admin_html.py tests/test_api.py
git commit -m "fix: escape admin content and bound task APIs"
```

### Task 4: Make Docker images immutable, non-root, and cache-correct

**Files:**
- Modify: `Dockerfile`, `Dockerfile-Gpu`, `.dockerignore`, `config.py`, `core/task_manager.py`, `.github/workflows/docker.yml`, `README.md`
- Create: `tests/test_dockerfiles.py`

**Interfaces:**
- `config._CONFIG_PATH` honors `CONFIG_PATH`, defaulting to the repository `config.json` for non-container development.
- `TaskManager()` honors `TASK_DB_PATH`, defaulting to `tasks.db`.
- Container runtime exposes `/data` for config, SQLite, temporary audio, and model cache and runs as `app` (non-root).

- [ ] **Step 1: Write failing Docker/config tests**

```python
def test_cpu_dockerfile_has_no_credential_defaults():
    dockerfile = Path("Dockerfile").read_text()
    assert "ADMIN_PASSWORD=" not in dockerfile
    assert "USER app" in dockerfile
    assert "uv sync --locked --no-dev --no-install-project" in dockerfile

def test_config_path_can_be_overridden(monkeypatch, tmp_path):
    monkeypatch.setenv("CONFIG_PATH", str(tmp_path / "config.json"))
    module = importlib.reload(config)
    assert module._CONFIG_PATH == tmp_path / "config.json"
```

- [ ] **Step 2: Run focused tests and verify RED**

Run: `uv run pytest tests/test_dockerfiles.py -q`

Expected: FAIL because credentials are embedded, runtime runs as root, and dependency installation ignores the lock.

- [ ] **Step 3: Implement Docker and persistence hardening**

Use a dedicated virtual environment populated by `uv sync --locked --no-dev --no-install-project`, copy it into the runtime stage, pin the uv image tag, copy only runtime source files, exclude config/tests/assets from the build context, create/chown `/data`, and run as `app`. Repair the GPU dependency-copy location by copying the dedicated virtual environment rather than Python's distribution-specific site-packages. Build CPU and GPU images in CI, with GitHub Actions cache retained.

- [ ] **Step 4: Run focused tests, static Docker checks, and image builds**

Run:

```bash
uv run pytest tests/test_dockerfiles.py -q
docker build --check -f Dockerfile .
docker build --check -f Dockerfile-Gpu .
docker build -t jellysub-ai:security-check .
```

Expected: all commands exit 0; Docker checks report no embedded-secret warnings.

- [ ] **Step 5: Commit**

```bash
git add Dockerfile Dockerfile-Gpu .dockerignore config.py core/task_manager.py .github/workflows/docker.yml README.md tests/test_dockerfiles.py
git commit -m "fix: harden reproducible container runtime"
```

### Task 5: Full regression verification

**Files:**
- Modify: `docs/superpowers/plans/2026-08-07-security-hardening.md`

- [ ] **Step 1: Run the non-model suite**

Run: `uv run pytest -q`

Expected: PASS with integration/model tests excluded by the project marker configuration.

Result (2026-08-07): `uv run pytest -q` exited 1: 60 passed, 3 failed, 1 deselected. The three failures are reproducible on `main`: two require unavailable `ffmpeg`; the subtitle-discovery assertion also fails unchanged on `main`. The task-pipeline failure in this branch is the expected downstream consequence of the unavailable `ffmpeg`.

- [x] **Step 2: Run Docker validation**

Run:

```bash
docker build --check -f Dockerfile .
docker build --check -f Dockerfile-Gpu .
git status --short
```

Expected: Docker checks exit 0 without security warnings; only intended tracked files are modified.

Result (2026-08-07): both CPU and GPU `docker build --check` commands exited 0 and reported `Check complete, no warnings found.` `git diff --check` exited 0. The only worktree status entry before recording this result was the intended, previously untracked plan under `docs/`.

- [x] **Step 3: Record verification results and commit the plan**

```bash
git add docs/superpowers/plans/2026-08-07-security-hardening.md
git commit -m "docs: record security hardening verification"
```

Result (2026-08-07): recorded despite the non-model suite's environmental/baseline failures. See `.superpowers/sdd/2026-08-07-security-hardening/task-5-report.md` for the complete command summary and comparison.

## Self-Review

- Coverage: Tasks 1–3 address default credentials, session integrity, webhook authentication/path abuse/duplication, XSS, and request bounds. Task 4 addresses plaintext image configuration, root runtime, the GPU copy-path risk, lockfile usage, source-only context, and CI coverage. Task 5 requires end-to-end verification.
- No placeholder scan: each task names concrete files, commands, interfaces, expected failure/pass behavior, and code direction.
- Type consistency: `validate_security_config`, `_verify_webhook_signature`, and `TaskManager.has_active_task` are defined once and consumed by the stated callers/tests.
