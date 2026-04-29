"""环境变量配置 — 从 .env 文件或系统环境变量读取。"""

import logging
import os

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# 加载 .env 文件（系统环境变量优先）
load_dotenv()

# --- 认证 ---
ADMIN_USERNAME: str = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD: str = os.getenv("ADMIN_PASSWORD", "admin")
TOTP_SECRET: str = os.getenv("TOTP_SECRET", "")
SESSION_SECRET: str = os.getenv("SESSION_SECRET", "")

# --- Webhook ---
WEBHOOK_SECRET: str = os.getenv("WEBHOOK_SECRET", "")

# --- 模型下载源 ---
# "huggingface" | "modelscope" | "" (默认 huggingface)
MODEL_SOURCE: str = os.getenv("MODEL_SOURCE", "")

# --- 模型空闲超时 ---
# 空闲 N 秒后自动释放模型，降低内存压力。0 表示不释放（默认不释放）。
try:
    MODEL_IDLE_TIMEOUT: int = int(os.getenv("MODEL_IDLE_TIMEOUT", "0"))
except ValueError:
    MODEL_IDLE_TIMEOUT = 0

# --- 启动检查 ---
_warnings: list[str] = []

if ADMIN_USERNAME == "admin" and ADMIN_PASSWORD == "admin":
    _warnings.append("使用默认管理员凭据 (admin/admin)。请通过环境变量 ADMIN_USERNAME/ADMIN_PASSWORD 修改。")

if not TOTP_SECRET:
    _warnings.append("未设置 TOTP_SECRET，TOTP 验证将被跳过。建议使用 pyotp.random_base32() 生成。")

if not SESSION_SECRET:
    _warnings.append("未设置 SESSION_SECRET，session 安全性较低。建议设置为一个随机字符串。")

if not WEBHOOK_SECRET:
    logger.info("未设置 WEBHOOK_SECRET，Webhook 将不校验签名（开发模式）。")

if MODEL_SOURCE == "modelscope":
    import os as _os
    _os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    logger.info("使用 ModelScope 镜像源下载模型（HF_ENDPOINT=https://hf-mirror.com）")

for w in _warnings:
    logger.warning(w)
