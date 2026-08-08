"""环境变量配置 — 从 .env 文件或系统环境变量读取。"""

import logging
import os

from dotenv import load_dotenv

logger = logging.getLogger("uvicorn.error")

# 加载 .env 文件（系统环境变量优先）
load_dotenv()


def parse_cors_origins(raw_origins: str) -> tuple[str, ...]:
    """Parse a comma-separated CORS allowlist without permitting wildcards."""
    origins = tuple(origin.strip() for origin in raw_origins.split(",") if origin.strip())
    if "*" in origins:
        raise ValueError("CORS origins must not include wildcard '*'")
    return origins


# --- 认证 ---
ADMIN_USERNAME: str = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD: str = os.getenv("ADMIN_PASSWORD", "admin")
TOTP_SECRET: str = os.getenv("TOTP_SECRET", "")
SESSION_SECRET: str = os.getenv("SESSION_SECRET", "")
SESSION_HTTPS_ONLY: bool = os.getenv("SESSION_HTTPS_ONLY", "true").lower() in {"1", "true", "yes"}
DEVELOPMENT_MODE: bool = os.getenv("DEVELOPMENT_MODE", "false").lower() in {"1", "true", "yes"}
CORS_ORIGINS: tuple[str, ...] = parse_cors_origins(os.getenv("CORS_ORIGINS", ""))

# --- Webhook ---
WEBHOOK_SECRET: str = os.getenv("WEBHOOK_SECRET", "")

# --- 模型下载源 ---
# "huggingface" | "modelscope" | "" (默认 huggingface)
MODEL_SOURCE: str = os.getenv("MODEL_SOURCE", "")
MODEL_PRELOAD_ENABLED: bool = os.getenv("MODEL_PRELOAD_ENABLED", "true").lower() in {"1", "true", "yes"}


_PUBLISHED_ADMIN_USERNAMES = frozenset({
    "your-admin-username",
    "your_admin_username",
    "myuser",
})
_PUBLISHED_ADMIN_PASSWORDS = frozenset({
    "replace-with-a-strong-password",
    "your_secure_password",
    "mypassword",
})
_PUBLISHED_SESSION_SECRETS = frozenset({
    "change_me_in_production",
    "replace-with-a-random-32-character-minimum-secret",
    "generate-a-random-secret-with-at-least-32-characters",
})

_APPROVED_LOCAL_MODELS = frozenset({
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-ASR-0.6B",
    "Qwen/Qwen3-ASR-1.7B",
    "Qwen/Qwen3-ForcedAligner-0.6B",
    "iic/SenseVoiceSmall",
    "FunAudioLLM/SenseVoiceSmall",
    "Helsinki-NLP/opus-mt-en-zh",
})


def validate_local_model(model_name: str) -> str:
    """Allow only repositories whose model code and weights are reviewed."""
    if model_name not in _APPROVED_LOCAL_MODELS:
        raise ValueError(f"Local model repository is not approved: {model_name}")
    return model_name


def validate_security_config(
    username: str,
    password: str,
    session_secret: str,
    *,
    totp_secret: str = TOTP_SECRET,
    development_mode: bool = DEVELOPMENT_MODE,
) -> None:
    """Reject credentials and session keys that are unsafe for a running service."""
    if development_mode:
        return
    if (
        not username
        or not password
        or (username == "admin" and password == "admin")
        or username in _PUBLISHED_ADMIN_USERNAMES
        or password in _PUBLISHED_ADMIN_PASSWORDS
    ):
        raise RuntimeError("Set non-default ADMIN_USERNAME and ADMIN_PASSWORD")
    if len(session_secret) < 32 or session_secret in _PUBLISHED_SESSION_SECRETS:
        raise RuntimeError("Set a random SESSION_SECRET of at least 32 characters")
    if not development_mode and not totp_secret:
        raise RuntimeError("Set TOTP_SECRET outside explicit development mode")

# --- 模型空闲超时 ---
# 空闲 N 秒后自动释放模型，降低内存压力。0 表示不释放（默认不释放）。
try:
    MODEL_IDLE_TIMEOUT: int = int(os.getenv("MODEL_IDLE_TIMEOUT", "0"))
except ValueError:
    MODEL_IDLE_TIMEOUT = 0

# --- 内存保护 ---
# 最大内存占用上限（GB），0 表示不限制。
try:
    MAX_MEMORY_GB: float = float(os.getenv("MAX_MEMORY_GB", "0"))
except ValueError:
    MAX_MEMORY_GB = 0.0

# --- 启动检查 ---
_warnings: list[str] = []

if not DEVELOPMENT_MODE:
    if ADMIN_USERNAME == "admin" and ADMIN_PASSWORD == "admin":
        _warnings.append("使用默认管理员凭据 (admin/admin)。请通过环境变量 ADMIN_USERNAME/ADMIN_PASSWORD 修改。")

    if not TOTP_SECRET:
        _warnings.append("未设置 TOTP_SECRET，生产启动将被拒绝。建议使用 pyotp.random_base32() 生成。")

    if not SESSION_SECRET:
        _warnings.append("未设置 SESSION_SECRET，session 安全性较低。建议设置为一个随机字符串。")

if not WEBHOOK_SECRET:
    logger.info("未设置 WEBHOOK_SECRET，Webhook 端点已禁用。")

if MODEL_SOURCE == "modelscope":
    import os as _os
    _os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    logger.info("使用 ModelScope 镜像源下载模型（HF_ENDPOINT=https://hf-mirror.com）")

for w in _warnings:
    logger.warning(w)
