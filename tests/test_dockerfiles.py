import importlib
from pathlib import Path

import config
from core.task_manager import TaskManager


DOCKERFILES = (Path("Dockerfile"), Path("Dockerfile-Gpu"))


def test_cpu_dockerfile_has_no_credential_defaults():
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")

    assert "ADMIN_PASSWORD=" not in dockerfile
    assert "USER app" in dockerfile
    assert "uv sync --locked --no-dev --no-install-project" in dockerfile


def test_production_dockerfiles_use_reproducible_non_root_runtime():
    for path in DOCKERFILES:
        dockerfile = path.read_text(encoding="utf-8")

        assert "ghcr.io/astral-sh/uv:latest" not in dockerfile
        assert "uv sync --locked --no-dev --no-install-project" in dockerfile
        assert "COPY --from=builder /app/.venv /app/.venv" in dockerfile
        assert "COPY . ." not in dockerfile
        assert "USER app" in dockerfile


def test_production_dockerfiles_keep_mutable_state_in_data_volume():
    for path in DOCKERFILES:
        dockerfile = path.read_text(encoding="utf-8")

        assert "CONFIG_PATH=/data/config.json" in dockerfile
        assert "TASK_DB_PATH=/data/tasks.db" in dockerfile
        assert "MODEL_CACHE=/data/model_cache" in dockerfile
        assert "TEMP_DIR=/data/tmp" in dockerfile
        assert "VOLUME [\"/data\"]" in dockerfile
        assert "WORKDIR /data" in dockerfile


def test_docker_context_excludes_development_and_mutable_files():
    ignored = {
        line.strip().rstrip("/")
        for line in Path(".dockerignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert {"config.json", "tests", "assets"} <= ignored


def test_config_path_can_be_overridden(monkeypatch, tmp_path):
    config_path = tmp_path / "config.json"

    with monkeypatch.context() as context:
        context.setenv("CONFIG_PATH", str(config_path))
        module = importlib.reload(config)
        assert module._CONFIG_PATH == config_path

    importlib.reload(config)


def test_config_path_defaults_to_repository_config(monkeypatch):
    monkeypatch.delenv("CONFIG_PATH", raising=False)

    module = importlib.reload(config)

    assert module._CONFIG_PATH == Path(config.__file__).parent / "config.json"


def test_temp_path_can_be_overridden(monkeypatch):
    monkeypatch.setenv("TEMP_DIR", "/data/tmp")

    assert config.AppConfig().temp_dir == "/data/tmp"


def test_task_manager_path_can_be_overridden(monkeypatch, tmp_path):
    db_path = tmp_path / "tasks.db"
    monkeypatch.setenv("TASK_DB_PATH", str(db_path))

    manager = TaskManager()

    assert manager.db_path == str(db_path)
    assert db_path.is_file()


def test_task_manager_path_defaults_to_tasks_db(monkeypatch, tmp_path):
    monkeypatch.delenv("TASK_DB_PATH", raising=False)
    monkeypatch.chdir(tmp_path)

    manager = TaskManager()

    assert manager.db_path == "tasks.db"
    assert (tmp_path / "tasks.db").is_file()


def test_ci_builds_cpu_and_gpu_images_with_separate_caches():
    workflow = Path(".github/workflows/docker.yml").read_text(encoding="utf-8")

    assert "dockerfile: Dockerfile\n" in workflow
    assert "dockerfile: Dockerfile-Gpu\n" in workflow
    assert "file: ${{ matrix.dockerfile }}" in workflow
    assert "scope=${{ matrix.variant }}" in workflow
