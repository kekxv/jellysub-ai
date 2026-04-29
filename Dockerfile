# ---- 阶段 1: 构建阶段 ----
FROM python:3.12-slim AS builder

# 1. 安装 uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# 2. 安装编译所需的系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    libsox-dev \
    && rm -rf /var/lib/apt/lists/*

# 3. 先拷贝依赖定义文件
COPY pyproject.toml uv.lock ./

# 4. 分层安装依赖
# 第一步：只安装 PyTorch (利用 pyproject.toml 里的索引配置)
# 只要 pyproject.toml 里的 torch 版本不变，这一层 400MB 永远被缓存
RUN UV_HTTP_TIMEOUT=300 uv pip install --system --no-cache-dir \
    "torch>=2.0" "torchaudio" \
    --index-url https://download.pytorch.org/whl/cpu

# 第二步：安装剩余的所有依赖
# 即使改了业务代码，只要没改依赖，这一层也会命中缓存
RUN UV_HTTP_TIMEOUT=300 uv pip install --system --no-cache-dir \
    -r pyproject.toml

# 注意：这里【不要】运行 uv pip install .
# 因为安装 . 会把 core 文件夹拷贝进 site-packages，导致缓存失效。

# ---- 阶段 2: 运行阶段 ----
FROM python:3.12-slim

# 设置环境变量
# 将 /app 添加到 python 路径，这样不需要安装项目也能 import core
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    ADMIN_USERNAME=admin \
    ADMIN_PASSWORD=admin \
    MODEL_SOURCE="" \
    PYTHONPATH=/app

WORKDIR /app

# 5. 安装运行时必要的系统库 (FFmpeg 和 libsox)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsox3 \
    && rm -rf /var/lib/apt/lists/*

# 6. 从构建阶段拷贝已经装好的“纯净”依赖 (约 800MB-1GB)
# 这一层在推送/拉取时，只要依赖没变，就是 Layer already exists
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# 7. 最后一步：拷贝业务代码 (通常只有几百 KB)
# 以后改代码，Docker 只会重新构建和推送这一层
COPY . .

EXPOSE 8000

# 直接启动，因为 PYTHONPATH 已经包含了 /app，Python 能自动找到 core 文件夹
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]