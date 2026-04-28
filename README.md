# JellySub-AI

Jellyfin 自动化 AI 字幕生成旁路服务。

## 功能

当 Jellyfin 有新媒体入库时，通过 Webhook 通知本服务，自动完成：

1. 检查是否已有中文字幕（含 UTF-8 编码校验）
2. 使用 FFmpeg 提取音频
3. 使用 Qwen3-ASR 进行语音识别
4. 使用大模型 API 翻译为目标语言
5. 生成标准 SRT 字幕并通知 Jellyfin 刷新

## 技术栈

- **后端**: Python 3.10+ / FastAPI
- **前端**: 纯 HTML + JS + CSS（由 FastAPI 静态挂载）
- **音频处理**: FFmpeg (subprocess)
- **ASR**: ModelScope `Qwen/Qwen3-ASR-0.6B`
- **翻译**: OpenAI 兼容格式的大模型 API
- **依赖管理**: uv

## 快速开始

### 环境要求

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) 包管理器
- FFmpeg / ffprobe（系统路径中可用）

### 安装依赖

```bash
uv sync
```

### 启动服务

```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8000
```

访问 `http://localhost:8000/` 打开配置页面。

### 配置 Jellyfin Webhook

在 Jellyfin 仪表盘中安装 **Webhook** 插件，添加一个通知：

- **触发事件**: `Item Added`
- **请求类型**: POST
- **URL**: `http://<your-server>:8000/webhook`
- **请求格式**: JSON

插件发送的 payload 包含 `ItemId`, `Path`, `ItemType`, `Name` 等字段，本服务会自动解析并处理。

### 路径映射

如果 Jellyfin 和本服务运行在不同的容器或路径下，需在配置页面设置路径映射：

```
/media → /mnt/data
```

表示 Jellyfin 报告的 `/media/movie.mp4` 实际对应本地的 `/mnt/data/movie.mp4`。

## 项目结构

```
JellySub-AI/
├── main.py              # FastAPI 入口，Webhook 路由 + 后台任务
├── config.py            # 配置模型 + JSON 读写
├── pyproject.toml       # uv 依赖管理
├── core/
│   ├── audio.py         # FFmpeg 音频提取 + ffprobe 字幕流检查
│   ├── asr.py           # Qwen3-ASR 推理（自动检测 GPU）
│   ├── translate.py     # 大模型翻译字幕（严格锁定时间轴）
│   ├── jellyfin_api.py  # Jellyfin REST API 客户端
│   ├── subtitle_checker.py  # 已有字幕检查 + UTF-8 校验
│   └── subtitle_writer.py   # SRT 文件生成（UTF-8）
├── static/
│   ├── index.html       # WebUI 配置页
│   └── style.css        # 样式
└── tests/               # 测试用例
```

## 运行测试

```bash
uv run pytest -v
```

## API 端点

| 方法     | 路径            | 说明                  |
|--------|---------------|---------------------|
| `GET`  | `/`           | 配置页面                |
| `POST` | `/webhook`    | 接收 Jellyfin Webhook |
| `GET`  | `/api/config` | 获取当前配置              |
| `PUT`  | `/api/config` | 保存配置                |
| `GET`  | `/static/*`   | 静态资源                |

## 开发

```bash
# 安装开发依赖
uv sync

# 运行测试
uv run pytest -v

# 启动开发服务器（自动重载）
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Docker如何运行

由于涉及模型、数据库和环境变量，建议使用以下方式运行：

> 在线镜像： ghcr.io/kekxv/jellysub-ai:main

#### 方案 A：Docker Run (快速测试)

```bash
docker build -t jellysub-ai .

docker run -d -p 8000:8000 \
  -e ADMIN_PASSWORD="your_secure_password" \
  -e MODEL_SOURCE="modelscope" \
  -v $(pwd)/model_cache:/app/model_cache \
  -v $(pwd)/tasks.db:/app/tasks.db \
  --name jellysub jellysub-ai
```

#### 方案 B：Docker Compose (推荐)

创建 `docker-compose.yml`:

```yaml
version: '3.8'
services:
  jellysub:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ADMIN_USERNAME=myuser
      - ADMIN_PASSWORD=mypassword
      - SESSION_SECRET=random_string_here
      - MODEL_SOURCE=modelscope
      - MODEL_IDLE_TIMEOUT=300
    volumes:
      - ./model_cache:/app/model_cache
      - ./tasks.db:/app/tasks.db
      - ./assets:/app/assets
    restart: always
```
