# JellySub-AI

Jellyfin 自动化 AI 字幕生成旁路服务。

## 功能

### 模式一：Webhook 自动触发

当 Jellyfin 有新媒体入库时，通过 Webhook 通知本服务，自动完成：

1. 检查是否已有中文字幕（含 UTF-8 编码校验）
2. 使用 FFmpeg 提取音频
3. 使用 Qwen3-ASR 进行语音识别
4. 使用大模型 API 翻译为目标语言
5. 生成标准 SRT 字幕并通知 Jellyfin 刷新

### 模式二：本地视频手动扫描

1. 在配置页面添加本地视频目录路径
2. 服务自动递归扫描视频文件（支持中文路径、含空格路径、最深 5 级子目录）
3. 在 WebUI 中选择单个或批量视频生成字幕

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

#### 1. 安装 Jellyfin Webhook 插件

在 Jellyfin 仪表盘中进入 **插件 → 目录**，搜索并安装 **Webhook** 插件，安装后重启 Jellyfin。

#### 2. 添加通知目标

进入 **仪表板 → 插件 → Webhook → 添加**，配置如下：

| 配置项             | 值                                               |
|-------------------|--------------------------------------------------|
| **名称**           | JellySub-AI（随意填写）                           |
| **Webhook 类型**   | `Generic Webhook`                                |
| **Enabled**        | ✅ 勾选                                            |
| **URI**            | `http://<jellysub-ai服务IP>:8000/webhook`         |
| **Method**         | `POST`                                           |
| **Format**         | `JSON`                                           |

#### 3. 配置签名校验（推荐）

Jellyfin Webhook 插件支持通过 `X-Jellyfin-Signature` 请求头发送签名。本服务使用 SHA256 哈希校验来验证请求来源，防止未授权调用。

**步骤：**

1. **选择一个共享密钥**（任意字符串，例如 `my-secret-key-2026`）
2. **计算该密钥的 SHA256 哈希值**：
   ```bash
   echo -n "my-secret-key-2026" | sha256sum
   # 输出示例: a1b2c3d4e5f6...（64位十六进制字符串）
   ```
3. **在 Jellyfin Webhook 插件中**，找到 **Shared Secret** 或 **Custom Headers** 字段：
   - 如果插件有 **Shared Secret** 字段：直接填入你选择的原始密钥（`my-secret-key-2026`），插件会自动计算签名
   - 如果插件只有 **Custom Headers** 字段：手动添加一个 Header：
     - **Name**: `X-Jellyfin-Signature`
     - **Value**: 上一步算出的 SHA256 哈希值
4. **在本服务的 `.env` 文件中**设置相同的密钥：
   ```bash
   WEBHOOK_SECRET=my-secret-key-2026
   ```

**校验原理：** 本服务收到请求后，会计算 `sha256(WEBHOOK_SECRET)` 并与请求头中的 `X-Jellyfin-Signature` 值对比。两者必须完全一致。

> **注意：** 如果 `WEBHOOK_SECRET` 为空（默认），签名校验将被跳过。开发测试时可不设置，但生产环境强烈建议启用。

#### 4. 配置触发事件

在 Webhook 插件的 **Triggers** 选项卡中，添加以下触发条件：

| 触发事件            | 说明                           |
|--------------------|--------------------------------|
| `Item Added`       | 新媒体文件入库时触发（**必需**）  |
| `Item Marked Played` | 播放时触发（可选）             |

通常只需配置 `Item Added`，这样每当有新视频入库时就会自动检查并生成字幕。

#### 5. 可选：自定义通知内容

Jellyfin Webhook 插件默认发送的 JSON payload 包含以下字段，本服务会自动解析：

```json
{
  "ItemType": "Movie",
  "ItemId": "abc123",
  "Name": "电影名称",
  "Path": "/media/movies/电影名称/电影名称.mkv",
  "ServerName": "MyJellyfin",
  "ServerUrl": "http://localhost:8096"
}
```

电视剧还会额外发送 `SeriesName`、`SeasonNumber00`、`EpisodeNumber00` 字段。本服务也兼容小写字段名（`item_id`、`path`、`item_type`）。

#### 6. 完整配置示例

以下是一个从密钥设置到 Jellyfin 配置的完整流程：

```bash
# --- 第一步：生成密钥的 SHA256 ---
echo -n "jellysub-webhook-secret" | sha256sum
# 输出: e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855

# --- 第二步：在 .env 中配置 ---
cat >> .env << 'EOF'
WEBHOOK_SECRET=jellysub-webhook-secret
EOF

# --- 第三步：重启本服务 ---
uv run uvicorn main:app --host 0.0.0.0 --port 8000
```

然后在 Jellyfin Webhook 插件中：
- **Shared Secret** 填写 `jellysub-webhook-secret`（插件会自动计算 SHA256）
- 或者 **Custom Headers** 添加 `X-Jellyfin-Signature: e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`

#### 7. Webhook 响应说明

本服务收到请求后会返回以下状态之一：

| 响应                                  | 含义                                   |
|--------------------------------------|----------------------------------------|
| `{"status": "accepted", "item": "..."}` | 任务已创建，开始处理                      |
| `{"status": "skipped", "reason": "unsupported type"}` | 非视频类型（音乐、图片等），跳过 |
| `{"status": "skipped", "reason": "subtitle already exists"}` | 已有有效中文字幕，跳过 |
| `{"status": "skipped", "reason": "internal subtitle"}` | 视频自带内置字幕，跳过 |
| `{"status": "error", "reason": "invalid signature"}` | 签名校验失败，请检查 `WEBHOOK_SECRET` |
| `{"status": "error", "reason": "missing Path or ItemId"}` | 请求缺少必要字段 |

### 本地视频扫描

在配置页面（`http://localhost:8000/`）中设置 **视频目录**（`video_dirs`），添加你存放视频的本地路径。服务会自动扫描这些目录中的视频文件。

**扫描特性：**

- **支持格式**: mp4、mkv、avi、mov、wmv、flv、webm
- **中文路径**: 完全支持文件名和目录名包含中文
- **含空格路径**: 路径中包含空格也能正常解析
- **递归子目录**: 自动深入子目录查找视频（默认最深 5 级），例如：
  ```
  /data/movies/
    ├── Movie A/
    │   └── Movie A (2024).mkv          ← 扫描到
    └── Series B/
        ├── Season 1/
        │   ├── S01E01.mp4              ← 扫描到
        │   └── S01E02.mp4              ← 扫描到
        └── Season 2/                   ← 超过 5 级则不再深入
  ```

扫描结果会通过 `GET /api/videos` 接口返回，你可以在 WebUI 中查看并选择单个或批量生成字幕。

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
  -v $(pwd)/config.json:/app/config.json \
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
      - ./config.json:/app/config.json
      - ./tasks.db:/app/tasks.db
      - ./assets:/app/assets
    restart: always
```
