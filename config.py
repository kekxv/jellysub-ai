"""配置模块 — Pydantic 模型 + JSON 文件读写。"""

import json
import logging
from pathlib import Path

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parent / "config.json"


class AppConfig(BaseModel):
    jellyfin_url: str = "http://localhost:8096"
    jellyfin_api_key: str = ""

    # ASR 配置
    asr_mode: str = "local"           # "local" | "online"
    asr_model: str = "Qwen/Qwen3-ASR-0.6B"   # 本地模型名
    asr_api_url: str = ""             # 在线 API 地址
    asr_api_key: str = ""             # 在线 API 密钥
    asr_model_online: str = ""        # 在线模型名

    # 翻译配置
    translate_mode: str = "local"     # "online" | "local"
    translate_api_url: str = "https://api.openai.com/v1"
    translate_api_key: str = ""
    translate_model: str = "gpt-4o"
    translate_model_local: str = "Qwen/Qwen3-0.6B"
    translate_prompt_format: str = "json"  # "json" | "numbered"
    translate_thinking: bool = False  # 在线模式是否开启思考

    target_language: str = "zh-CN"
    path_mappings: dict[str, str] = Field(default_factory=dict)
    temp_dir: str = "./tmp"
    video_dirs: list[str] = Field(default_factory=list)  # 本地视频目录列表


_config: AppConfig | None = None


def load_config() -> AppConfig:
    global _config
    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        _config = AppConfig(**data)
        logger.info("Config loaded from %s", _CONFIG_PATH)
    else:
        _config = AppConfig()
        logger.info("Using default config")
    return _config


def save_config(cfg: AppConfig) -> None:
    global _config
    _config = cfg
    with open(_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg.model_dump(), f, indent=2, ensure_ascii=False)
    logger.info("Config saved to %s", _CONFIG_PATH)


def get_config() -> AppConfig:
    global _config
    if _config is None:
        return load_config()
    return _config
