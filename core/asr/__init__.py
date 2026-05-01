"""ASR 语音识别引擎 — 统一入口。"""

import logging
import threading
import time

logger = logging.getLogger("uvicorn.error")

from core.asr.base import (
    AsrEngine,
    join_words,
    add_silence_gaps,
    group_into_segments,
)
from core.asr.qwen3 import Qwen3AsrEngine
from core.asr.sensevoice import SenseVoiceAsrEngine
from core.asr.openai_api import OpenaiAsrEngine

# 向后兼容别名（测试用）
_add_silence_gaps = add_silence_gaps

__all__ = [
    "AsrEngine",
    "Qwen3AsrEngine",
    "SenseVoiceAsrEngine",
    "OpenaiAsrEngine",
    "join_words",
    "add_silence_gaps",
    "group_into_segments",
    "_add_silence_gaps",  # 向后兼容别名
    "get_asr_engine",
    "run_asr",
    "set_asr_busy",
    "release_all_engines",
    "check_asr_idle",
]

# =========================================================================== #
#  工厂函数与引擎缓存
# =========================================================================== #

_engine_instances: dict[str, AsrEngine] = {}
_engine_last_access: float = 0.0
_engine_busy: bool = False


def set_asr_busy(busy: bool):
    global _engine_busy
    _engine_busy = busy
    if busy:
        _touch_engine()


def _touch_engine():
    global _engine_last_access
    _engine_last_access = time.time()


def get_asr_engine(
    engine: str = "qwen3-asr",
    model_name: str = "Qwen/Qwen3-ASR-0.6B",
    api_url: str = "",
    api_key: str = "",
    model_online: str = "",
    device: str | None = None,
) -> AsrEngine:
    """创建并缓存 ASR 引擎实例。

    engine: "qwen3-asr" | "sensevoice" | "openai"
    """
    # 唯一标识一个引擎配置
    cache_key = f"{engine}:{model_name}:{api_url}:{model_online}"
    
    global _engine_instances
    if cache_key in _engine_instances:
        _touch_engine()
        return _engine_instances[cache_key]

    # 如果切换了模型，释放之前的本地模型
    if engine != "openai":
        release_all_engines()

    if engine == "qwen3-asr":
        eng = Qwen3AsrEngine(model_name=model_name, device=device)
    elif engine == "sensevoice":
        eng = SenseVoiceAsrEngine(model_name=model_name, device=device)
    elif engine == "openai":
        eng = OpenaiAsrEngine(api_url=api_url, api_key=api_key, model=model_online or "whisper-1")
    else:
        raise ValueError(f"Unknown ASR engine: {engine}. Available: {list(_engine_registry.keys())}")

    if engine != "openai":
        _engine_instances[cache_key] = eng
    _touch_engine()
    return eng


def release_all_engines():
    """释放所有缓存的本地 ASR 引擎。"""
    global _engine_instances, _engine_last_access
    for eng in _engine_instances.values():
        if hasattr(eng, "release"):
            eng.release()
    _engine_instances.clear()
    _engine_last_access = 0.0


def check_asr_idle(timeout: int):
    """检查 ASR 引擎空闲超时并释放。"""
    global _engine_last_access, _engine_busy
    if _engine_instances and timeout > 0:
        if _engine_busy:
            _touch_engine()  # 正在使用中，自动续期
            return
        elapsed = time.time() - _engine_last_access
        if elapsed > timeout:
            logger.info("ASR engine idle timeout (%ds), releasing...", timeout)
            release_all_engines()


# =========================================================================== #
#  向后兼容入口
# =========================================================================== #

_qwen_engine: Qwen3AsrEngine | None = None


def _load_model(model_name: str, device: str | None = None):
    """兼容旧接口：加载 Qwen3-ASR 模型并返回内部模型实例。"""
    global _qwen_engine
    if _qwen_engine is None:
        _qwen_engine = Qwen3AsrEngine(model_name=model_name, device=device)
    elif _qwen_engine.model_name != model_name:
        _qwen_engine.release()
        _qwen_engine = Qwen3AsrEngine(model_name=model_name, device=device)
    return _qwen_engine._load_model()


def run_asr(
    audio_path: str,
    mode: str = "local",
    model_name: str = "Qwen/Qwen3-ASR-0.6B",
    asr_language: str = "auto",
    api_url: str = "",
    api_key: str = "",
    model_online: str = "",
    engine: str = "qwen3-asr",
    use_vad: bool = False,
    vad_min_silence_ms: int = 500,
) -> tuple[list[dict], str]:
    """
    对音频进行语音识别，返回 (片段列表, 检测到的语言代码)。

    engine: "qwen3-asr" | "sensevoice" | "openai"
    mode: 保留向后兼容，当 engine="openai" 或 mode="online" 时使用 API
    asr_language: "auto" | "zh" | "en" | "ja" | "ko" | "yue"
    use_vad: 是否使用 Silero VAD 分块处理长音频，防止死循环
    vad_min_silence_ms: VAD 最小静音持续时间（毫秒），用于切分
    """
    # 向后兼容：mode="online" 等价于 engine="openai"
    if mode == "online" or engine == "openai":
        eng = OpenaiAsrEngine(api_url=api_url, api_key=api_key, model=model_online or "whisper-1")
        return eng.transcribe(audio_path, language=asr_language)

    # 本地模式
    eng = get_asr_engine(engine, model_name=model_name, api_url=api_url, api_key=api_key, model_online=model_online)

    if use_vad and eng.need_vad():
        from core.asr.vad_wrapper import transcribe_with_vad
        return transcribe_with_vad(eng, audio_path, min_silence_ms=vad_min_silence_ms, language=asr_language)

    return eng.transcribe(audio_path, language=asr_language)
