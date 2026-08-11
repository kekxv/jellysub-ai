"""配置模块 — Pydantic 模型 + JSON 文件读写。"""

import json
import logging
import os
import tempfile
from pathlib import Path

from pydantic import BaseModel, Field

logger = logging.getLogger("uvicorn.error")

_CONFIG_PATH = Path(os.environ.get("CONFIG_PATH", Path(__file__).parent / "config.json"))


class AppConfig(BaseModel):
    jellyfin_url: str = "http://localhost:8096"
    jellyfin_api_key: str = ""

    # ASR 配置
    asr_engine: str = "qwen3-asr"     # "qwen3-asr" | "sensevoice" | "openai"
    asr_mode: str = "local"             # 保留向后兼容: "local" | "online"
    asr_model: str = "Qwen/Qwen3-ASR-0.6B"   # 本地模型名
    asr_language: str = "auto"          # "auto" | "zh" | "en" | "ja" | "ko" | "yue"
    asr_api_url: str = ""               # 在线 API 地址
    asr_api_key: str = ""               # 在线 API 密钥
    asr_model_online: str = ""          # 在线模型名

    # VAD 配置（用于修复"小声语音识别不全"）
    vad_threshold: float = 0.3           # Silero 语音概率阈值，越低越敏感（默认 0.5 会漏低 SNR 语音）
    vad_speech_pad_ms: int = 300         # 语音段前后 padding(ms)，保住词首词尾（默认 30ms 太小）
    vad_min_silence_ms: int = 500        # 最小静音(ms)，用于切分
    vad_min_speech_ms: int = 100         # 最小语音(ms)，过滤瞬态噪声
    max_subtitle_sec: float = 7.0        # 单条字幕最大时长(秒)，超限按句重切
    audio_normalize: bool = True         # 提取音频时 loudnorm 响度归一化

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
    temp_dir: str = Field(default_factory=lambda: os.environ.get("TEMP_DIR", "./tmp"))
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
    _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=_CONFIG_PATH.parent,
            prefix=f".{_CONFIG_PATH.name}.",
            suffix=".tmp",
            delete=False,
        ) as f:
            temp_path = f.name
            json.dump(cfg.model_dump(), f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, _CONFIG_PATH)
    except Exception:
        if temp_path:
            Path(temp_path).unlink(missing_ok=True)
        raise
    _config = cfg
    logger.info("Config saved to %s", _CONFIG_PATH)


def get_config() -> AppConfig:
    global _config
    if _config is None:
        return load_config()
    return _config
