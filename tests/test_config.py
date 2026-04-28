"""配置模块测试。"""

import json
import tempfile
from unittest.mock import patch

from pathlib import Path

from config import AppConfig, load_config, save_config, get_config


def test_app_config_defaults():
    """配置模型应有合理默认值。"""
    cfg = AppConfig()
    assert cfg.jellyfin_url == "http://localhost:8096"
    assert cfg.asr_model == "Qwen/Qwen3-ASR-0.6B"
    assert cfg.translate_api_url == "https://api.openai.com/v1"
    assert cfg.target_language == "zh-CN"
    assert cfg.path_mappings == {}
    assert cfg.temp_dir == "./tmp"


def test_save_and_load_config(tmp_path):
    """配置应能正确保存和加载。"""
    config_file = tmp_path / "config.json"
    cfg = AppConfig(
        jellyfin_url="http://test:8096",
        jellyfin_api_key="secret",
        path_mappings={"/media": "/mnt/data"},
    )

    with patch("config._CONFIG_PATH", config_file):
        save_config(cfg)

    assert config_file.exists()
    data = json.loads(config_file.read_text(encoding="utf-8"))
    assert data["jellyfin_url"] == "http://test:8096"
    assert data["jellyfin_api_key"] == "secret"
    assert data["path_mappings"] == {"/media": "/mnt/data"}


def test_load_config_missing_file():
    """缺失配置文件时应返回默认配置。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("config._CONFIG_PATH", Path(tmpdir) / "nonexistent.json"):
            cfg = load_config()
        assert cfg.jellyfin_url == "http://localhost:8096"
