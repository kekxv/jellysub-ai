# 项目：JellySub-AI (Jellyfin 自动化 AI 字幕生成旁路服务)

## 1. 项目概述
本项目是一个为 Jellyfin 媒体服务器设计的外部旁路服务。当 Jellyfin 有新媒体入库时，通过 Webhook 通知本项目。本项目将自动检查字幕、提取音频、使用 Qwen3-ASR 进行语音识别、使用大模型进行翻译，最终生成标准字幕并让 Jellyfin 刷新加载。

**核心限制：禁止使用 C# 开发。纯 Python + HTML 实现。**

## 2. 技术栈
- **后端框架**: Python 3.10+ / FastAPI (用于接收 Webhook 和提供配置 API)
- **前端配置页**: 纯 HTML + JS (Fetch API) + CSS，由 FastAPI 提供静态挂载。
- **音频处理**: FFmpeg (通过 `ffmpeg-python` 或原生 `subprocess` 调用)
- **ASR 模型**: ModelScope `Qwen/Qwen3-ASR-0.6B`
- **翻译服务**: 基于兼容 OpenAI 格式的大模型 API 端点进行翻译。
- **通信方式**: Jellyfin Webhook 插件 -> Python POST 接口; Python -> Jellyfin REST API (刷新媒体库)。

## 3. 核心业务工作流 (Workflow)

1. **接收事件**: 接收来自 Jellyfin Webhook 的 `ItemAdded` 或自定义事件，获取媒体的物理路径和 ItemID。
2. **存在性检查**: 
   - 检查媒体文件同级目录是否已存在同名 `.srt`/`.vtt`/`.ass` 文件。
   - (可选) 使用 `ffprobe` 检查视频文件内部是否包含有效的字幕流。
   - 如果已存在，则中止任务并记录日志。
3. **音频提取**:
   - 判断输入文件是独立音频还是视频文件。
   - 如果是单视频文件：使用 ffmpeg 提取出供 ASR 使用的标准格式音频（推荐 `wav, 16000Hz, 单声道`）。
   - 如果已经是独立音频文件且格式兼容：直接使用。
4. **ASR 语音识别 (源语言)**:
   - 加载 ModelScope `Qwen3-ASR-1.7B` 模型。
   - 对音频进行推断，获取包含精确时间戳 (`start`, `end`) 的文本片段。
   - 生成源语言字幕文件 (如 `movie.zh.srt` 或 `movie.en.srt`)。
5. **AI 接口翻译**:
   - 解析上一步生成的 SRT 文件。
   - **【核心约束：时间轴严格一致】**: 必须剥离时间轴，仅提取文本构成 JSON 数组发送给 AI，要求 AI 按原数组长度返回翻译后的 JSON 数组。
   - 接收翻译后，将翻译文本严格填回原有的时间轴结构中。
   - 生成目标语言字幕文件 (如 `movie.zh-CN.srt`)。
6. **字幕入库与通知**:
   - 将生成的字幕保存在原视频同一目录下，命名符合 Jellyfin 规范。
   - 调用 Jellyfin 官方 REST API (如 `/Items/{Id}/Refresh`) 强制扫描该视频，使得字幕立即可用。

## 4. 目录结构规范
```text
JellySub-AI/
├── main.py              # FastAPI 启动入口，Webhook 路由
├── config.py            # 配置文件读取与保存逻辑 (JSON/YAML)
├── core/
│   ├── audio.py         # FFmpeg 音频提取逻辑
│   ├── asr.py           # ModelScope Qwen3-ASR 交互逻辑
│   ├── translate.py     # 大模型 API 翻译与时间轴锁定逻辑
│   └── jellyfin_api.py  # 与 Jellyfin API 通信的工具类
├── static/
│   ├── index.html       # WebUI 配置页面前端
│   └── style.css        # 前端样式
└── requirements.txt     # 依赖包列表
```

## 5. 开发准则与提示 (AI 助手必读)
- **异步处理**: Webhook 必须立刻返回 `200 OK` 响应，不能阻塞。字幕处理的完整流程需放在后台任务 (BackgroundTasks 或 asyncio.create_task) 中执行。
- **路径转换**: 由于 Jellyfin 和本服务可能运行在不同的 Docker 容器或宿主机中，需在 WebUI 中提供“路径映射”配置 (例如将 Jellyfin 的 `/media` 映射到本地的 `/mnt/data`)。
- **时间轴安全**: 翻译逻辑中严禁让 AI 直接处理 `00:01:23,456 --> 00:01:25,789` 这样的字符串，极易发生格式破坏。应采用数组/字典映射策略。
- **容错处理**: 如果 ASR 生成失败、翻译 API 超时或 FFmpeg 报错，需捕获异常并记录详细日志，不要让整个服务崩溃。
