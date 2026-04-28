# ---- 阶段 1: 构建阶段 ----
FROM python:3.12-slim AS builder

# 安装 uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# 1. 安装编译所需的系统依赖 (修复 Building sox 失败的问题)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    libsox-dev \
    && rm -rf /var/lib/apt/lists/*

# 2. 预先拷贝依赖文件
COPY pyproject.toml uv.lock ./

# 3. 先安装依赖 (不安装项目本身)
# 注意：我们先不运行 "uv pip install ." 因为 core 文件夹还没考进来
# 我们利用 uv 的特性直接从 lock 文件安装依赖
RUN UV_HTTP_TIMEOUT=300 uv pip install --system --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple \
    -r pyproject.toml

# 4. 现在拷贝所有源码 (包括 core 文件夹)
COPY . .

# 5. 最后“正式安装”项目本身 (解决 package directory 'core' does not exist)
RUN uv pip install --system --no-cache-dir --no-deps .

# ---- 阶段 2: 运行阶段 ----
FROM python:3.12-slim

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    ADMIN_USERNAME=admin \
    ADMIN_PASSWORD=admin \
    MODEL_SOURCE=""

WORKDIR /app

# 安装运行时必要的系统库 (FFmpeg 和 libsox)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsox3 \
    && rm -rf /var/lib/apt/lists/*

# 从构建阶段拷贝环境
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# 拷贝代码
COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]