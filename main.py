"""JellySub-AI — FastAPI 入口，认证 + Webhook 路由 + 后台字幕处理。"""

import asyncio
import hashlib
import hmac
import json
import logging
import os
import threading
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from starlette.middleware.sessions import SessionMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import pyotp

from config import AppConfig, get_config, load_config, save_config
from core.task_manager import TaskManager
from core.audio import has_internal_subtitle
from env_config import (
    ADMIN_USERNAME,
    ADMIN_PASSWORD,
    TOTP_SECRET,
    WEBHOOK_SECRET,
    SESSION_SECRET,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="JellySub-AI")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Session 中间件
session_secret = SESSION_SECRET or os.urandom(32).hex()
app.add_middleware(SessionMiddleware, secret_key=session_secret, max_age=86400)

# 挂载静态文件
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# 挂载资源目录（测试视频）
assets_dir = Path(__file__).parent / "assets"
if assets_dir.exists():
    app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")


# --- Task Manager ---
task_manager = TaskManager()


# --- 认证辅助函数 ---

def require_auth(request: Request) -> bool:
    return request.session.get("authenticated", False)


def check_credentials(username: str, password: str, totp_code: str) -> bool:
    if username != ADMIN_USERNAME or password != ADMIN_PASSWORD:
        return False
    if TOTP_SECRET and totp_code:
        return pyotp.TOTP(TOTP_SECRET).verify(totp_code)
    return True


def _require_auth(request: Request):
    if not require_auth(request):
        from fastapi import HTTPException
        raise HTTPException(status_code=401, detail="Not authenticated")


def _validate_video_path(path_str: str):
    """验证路径是否在配置的视频目录中，防止路径穿越。"""
    if not path_str:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="Path is required")
    
    cfg = get_config()
    try:
        target_path = Path(path_str).resolve()
    except Exception:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="Invalid path format")

    allowed = False
    # Also allow assets directory for testing
    if assets_dir.exists() and target_path.is_relative_to(assets_dir.resolve()):
        allowed = True
    
    if not allowed:
        for d in cfg.video_dirs:
            try:
                if target_path.is_relative_to(Path(d).resolve()):
                    allowed = True
                    break
            except (ValueError, Exception):
                continue
            
    if not allowed:
        from fastapi import HTTPException
        logger.warning("Access denied to path: %s", path_str)
        raise HTTPException(status_code=403, detail="Access denied: path is outside allowed video directories")


# --- 模型预加载 ---

def _preload_asr_model(model_name: str):
    """后台线程预加载 ASR 模型。"""
    try:
        from core.asr import _load_model
        _load_model(model_name)
        logger.info("ASR model preloaded: %s", model_name)
    except Exception:
        logger.exception("ASR model preload failed")


def _preload_translate_model(model_name: str):
    """后台线程预加载翻译模型。"""
    try:
        from core.translate import _load_local_model
        _load_local_model(model_name)
        logger.info("Translation model preloaded: %s", model_name)
    except Exception:
        logger.exception("Translation model preload failed")


def _preload_models(cfg):
    """串行加载模型（应在 worker 线程内调用）。"""
    if cfg.asr_mode == "local" and cfg.asr_model:
        _preload_asr_model(cfg.asr_model)
    if cfg.translate_mode == "local" and cfg.translate_model_local:
        _preload_translate_model(cfg.translate_model_local)


def _start_idle_checker():
    """后台线程定期检查模型空闲超时并释放。"""
    from env_config import MODEL_IDLE_TIMEOUT
    from core.asr import _check_model_idle as _check_asr
    from core.translate import _check_model_idle as _check_translate

    def _check_loop():
        import time
        timeout = MODEL_IDLE_TIMEOUT
        while True:
            time.sleep(30)
            if timeout > 0:
                _check_asr(timeout)
                _check_translate(timeout)

    threading.Thread(target=_check_loop, daemon=True).start()


@app.on_event("startup")
def startup():
    _start_idle_checker()
    cfg = get_config()
    task_manager.start_worker(preload_hook=lambda: _preload_models(cfg))


@app.on_event("shutdown")
def shutdown():
    task_manager.stop_worker()


# --- 首页 ---

@app.get("/")
async def index():
    index_path = static_dir / "home.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "JellySub-AI is running"}


# --- 登录 ---

@app.get("/login")
async def login_page(request: Request):
    if require_auth(request):
        return RedirectResponse(url="/admin", status_code=302)
    index_path = static_dir / "login.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "Login page not found"}


class LoginRequest(BaseModel):
    username: str = ""
    password: str = ""
    totp_code: str = ""


@app.post("/login")
async def login(body: LoginRequest, request: Request):
    if check_credentials(body.username, body.password, body.totp_code):
        request.session["authenticated"] = True
        return {"status": "ok"}
    return {"status": "error", "detail": "用户名、密码或验证码错误"}


@app.post("/logout")
async def logout(request: Request):
    request.session.clear()
    return {"status": "ok"}


# --- 管理后台 ---

@app.get("/admin")
async def admin_page(request: Request):
    if not require_auth(request):
        return RedirectResponse(url="/login", status_code=302)
    index_path = static_dir / "admin.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "Admin page not found"}


# --- 脱敏状态 API ---

@app.get("/api/status")
async def api_status(request: Request):
    _require_auth(request)
    cfg = get_config()
    return {
        "asr_mode": cfg.asr_mode,
        "asr_model": cfg.asr_model if cfg.asr_mode == "local" else cfg.asr_model_online,
        "target_language": cfg.target_language,
        "translate_mode": cfg.translate_mode,
        "translate_model": cfg.translate_model if cfg.translate_mode == "online" else cfg.translate_model_local,
        "webhook_enabled": bool(WEBHOOK_SECRET),
    }


# --- 配置 API（需认证）---

class ConfigResponse(BaseModel):
    jellyfin_url: str = ""
    jellyfin_api_key: str = ""
    asr_mode: str
    asr_model: str
    asr_api_url: str
    asr_api_key: str
    asr_model_online: str
    translate_mode: str
    translate_api_url: str
    translate_api_key: str
    translate_model: str
    translate_model_local: str
    translate_prompt_format: str
    translate_thinking: bool
    target_language: str
    path_mappings: dict[str, str]
    temp_dir: str
    video_dirs: list[str]


@app.get("/api/config", response_model=ConfigResponse)
async def api_get_config(request: Request):
    _require_auth(request)
    cfg = get_config()
    return ConfigResponse(**cfg.model_dump())


@app.put("/api/config")
async def api_save_config(body: ConfigResponse, request: Request):
    _require_auth(request)
    cfg = AppConfig(**body.model_dump())
    save_config(cfg)
    # 在后台线程加载模型，不阻塞 HTTP 响应
    threading.Thread(target=_preload_models, args=(cfg,), daemon=True).start()
    return {"status": "saved"}


# --- 任务管理 API（需认证）---

@app.get("/api/tasks")
async def api_list_tasks(
    request: Request,
    status: str = None,
    pipeline_type: str = None,
    limit: int = 50,
    offset: int = 0,
):
    """列出所有任务，支持分页和过滤。"""
    _require_auth(request)
    tasks = task_manager.list_tasks(status=status, pipeline_type=pipeline_type, limit=limit, offset=offset)
    total = task_manager.count_tasks(status=status, pipeline_type=pipeline_type)
    # Parse JSON segments for frontend
    for t in tasks:
        if t.get("source_segments"):
            t["source_segments"] = json.loads(t["source_segments"])
        if t.get("translated_segments"):
            t["translated_segments"] = json.loads(t["translated_segments"])
    return {"tasks": tasks, "total": total}


@app.get("/api/tasks/{task_id}")
async def api_get_task(request: Request, task_id: int):
    """获取单个任务详情。"""
    _require_auth(request)
    task = task_manager.get_task(task_id)
    if not task:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Task not found")
    if task.get("source_segments"):
        task["source_segments"] = json.loads(task["source_segments"])
    if task.get("translated_segments"):
        task["translated_segments"] = json.loads(task["translated_segments"])
    return task


@app.post("/api/tasks/{task_id}/retry")
async def api_retry_task(request: Request, task_id: int):
    """手动重试失败任务。"""
    _require_auth(request)
    task = task_manager.get_task(task_id)
    if not task:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Task not found")
    task_manager.retry_task(task_id)
    return {"status": "queued"}


@app.delete("/api/tasks/{task_id}")
async def api_delete_task(request: Request, task_id: int):
    """删除任务。"""
    _require_auth(request)
    task_manager.delete_task(task_id)
    return {"status": "deleted"}


class BatchDeleteRequest(BaseModel):
    task_ids: list[int] = []


@app.post("/api/tasks/batch/delete")
async def api_batch_delete_tasks(body: BatchDeleteRequest, request: Request):
    """批量删除任务记录，不删除字幕文件。"""
    _require_auth(request)
    if not body.task_ids:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="task_ids is required and must not be empty")
    task_manager.delete_tasks(body.task_ids)
    return {"deleted": len(body.task_ids)}


# --- 测试 API（需认证）---

class TestTranslationRequest(BaseModel):
    texts: list[str] = []


@app.post("/api/test/translate")
async def api_test_translate(body: TestTranslationRequest, request: Request):
    """测试翻译效果。"""
    _require_auth(request)
    if not body.texts:
        body.texts = ["Hello world", "This is a test subtitle", "How are you doing today"]

    cfg = get_config()
    segments = [{"start": i * 2.0, "end": (i + 1) * 2.0, "text": t} for i, t in enumerate(body.texts)]

    from core.translate import translate_segments
    translated = await translate_segments(
        segments,
        cfg.target_language,
        mode=cfg.translate_mode,
        api_url=cfg.translate_api_url,
        api_key=cfg.translate_api_key,
        model=cfg.translate_model,
        model_local=cfg.translate_model_local,
        thinking=cfg.translate_thinking,
        prompt_format=cfg.translate_prompt_format,
    )

    result = []
    for orig, trans in zip(body.texts, translated):
        result.append({
            "original": orig,
            "translated": trans["text"],
        })
    return {"results": result}


@app.get("/api/test/status")
async def api_test_status(request: Request):
    """兼容端点：返回最新的 test 类型任务状态。"""
    _require_auth(request)
    task = task_manager.get_latest_by_type("test")
    if not task:
        return {"running": False, "progress": "idle", "percent": 0, "segments": [], "translated": []}
    return {
        "running": task["status"] in ("pending", "processing"),
        "progress": task.get("stage", "idle"),
        "percent": task.get("progress", 0),
        "segments": json.loads(task["source_segments"]) if task.get("source_segments") else [],
        "translated": json.loads(task["translated_segments"]) if task.get("translated_segments") else [],
    }


@app.post("/api/test/run")
async def api_test_run(request: Request):
    """启动测试流程：对 assets/en.mp4 进行 ASR + 翻译。"""
    _require_auth(request)
    # Check if there's already a running test task
    task = task_manager.get_latest_by_type("test")
    if task and task["status"] in ("pending", "processing"):
        return {"status": "already_running"}

    if not assets_dir.exists():
        return {"status": "error", "reason": "assets directory not found"}

    test_mp4 = assets_dir / "en.mp4"
    if not test_mp4.exists():
        return {"status": "error", "reason": "en.mp4 not found in assets"}

    task_id = task_manager.create_task(
        video_path=str(test_mp4),
        item_id="test",
        item_type="test",
        item_name="test",
        pipeline_type="test",
    )
    return {"status": "started", "task_id": task_id}


# --- 本地视频管理 API（需认证）---

_VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm"}


def _scan_videos(dirs: list[str]) -> list[dict]:
    """扫描目录中的视频文件。"""
    # Get currently processing video paths from DB
    processing_tasks = task_manager.list_tasks(status="processing", limit=100)
    processing_paths = {t["video_path"] for t in processing_tasks}

    videos = []
    for d in dirs:
        dir_path = Path(d)
        if not dir_path.is_dir():
            continue
        for f in sorted(dir_path.iterdir()):
            if f.is_file() and f.suffix.lower() in _VIDEO_EXTS:
                has_sub = any(
                    f.with_suffix(f".default.{lang}.srt").exists()
                    for lang in ["zh-CN", "zh-TW", "en", "ja", "ko"]
                )
                videos.append({
                    "id": str(f),
                    "name": f.name,
                    "path": str(f),
                    "size": f.stat().st_size,
                    "has_subtitle": has_sub,
                    "processing": str(f) in processing_paths,
                })
    return videos


@app.get("/api/videos")
async def api_list_videos(request: Request):
    """列出所有视频文件。"""
    _require_auth(request)
    cfg = get_config()
    videos = _scan_videos(cfg.video_dirs)
    return {"dirs": cfg.video_dirs, "videos": videos}


@app.get("/api/videos/stream")
async def api_stream_video(request: Request, path: str):
    """代理视频文件流，解决跨域问题。"""
    _require_auth(request)
    _validate_video_path(path)
    video_path = Path(path)
    if not video_path.exists() or not video_path.is_file():
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(str(video_path))


class SubtitleJobRequest(BaseModel):
    video_path: str


class BatchSubtitleRequest(BaseModel):
    video_paths: list[str] = []


@app.post("/api/videos/subtitle")
async def api_generate_subtitle(body: SubtitleJobRequest, request: Request):
    """开始生成字幕。"""
    _require_auth(request)
    video_path = body.video_path
    _validate_video_path(video_path)
    # Check if already running
    task = task_manager.get_latest_by_video_path(video_path)
    if task and task["status"] in ("pending", "processing"):
        return {"status": "already_running"}

    task_id = task_manager.create_task(
        video_path=video_path,
        pipeline_type="video_subtitle",
    )
    return {"status": "started", "task_id": task_id}


@app.post("/api/videos/subtitle/batch")
async def api_batch_generate_subtitle(body: BatchSubtitleRequest, request: Request):
    """批量生成字幕，跳过已在处理中的视频。"""
    _require_auth(request)
    if not body.video_paths:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="video_paths is required and must not be empty")
    for path in body.video_paths:
        _validate_video_path(path)
    task_ids, skipped = task_manager.create_tasks_batch(body.video_paths)
    return {"status": "started", "task_ids": task_ids, "skipped": skipped}


@app.get("/api/videos/subtitle/status")
async def api_subtitle_status(request: Request, path: str):
    """兼容端点：返回指定视频的最新任务状态。"""
    _require_auth(request)
    _validate_video_path(path)
    task = task_manager.get_latest_by_video_path(path)
    if not task:
        return {"status": "not_found"}
    return {
        "running": task["status"] in ("pending", "processing"),
        "progress": task.get("stage", "idle"),
        "percent": task.get("progress", 0),
        "segments": json.loads(task["source_segments"]) if task.get("source_segments") else [],
        "translated": json.loads(task["translated_segments"]) if task.get("translated_segments") else [],
    }


# --- Webhook 路由 ---

class WebhookPayload(BaseModel):
    """Jellyfin Webhook 通用 payload 模型。"""
    ServerName: str = ""
    ServerUrl: str = ""
    ItemType: str = ""
    Name: str = ""
    ItemId: str = ""
    Path: str = ""
    Username: str = ""
    UserId: str = ""
    SeriesName: str = ""
    SeasonNumber00: str = ""
    EpisodeNumber00: str = ""
    item_id: str = ""
    path: str = ""
    item_type: str = ""


@app.post("/webhook")
async def webhook(payload: WebhookPayload, request: Request):
    """接收 Jellyfin Webhook，校验签名后创建字幕任务。"""
    if WEBHOOK_SECRET:
        signature = request.headers.get("X-Jellyfin-Signature", "")
        expected = hashlib.sha256(WEBHOOK_SECRET.encode()).hexdigest()
        if not hmac.compare_digest(signature, expected):
            logger.warning("Webhook signature mismatch from %s", request.client.host if request.client else "unknown")
            return {"status": "error", "reason": "invalid signature"}

    item_id = payload.ItemId or payload.item_id
    item_path = payload.Path or payload.path
    item_type = payload.ItemType or payload.item_type
    item_name = payload.Name or "Unknown"

    logger.info("Webhook received: %s [%s] path=%s", item_name, item_type, item_path)

    if item_type not in ("Movie", "Episode"):
        logger.info("Skipping non-video item type: %s", item_type)
        return {"status": "skipped", "reason": f"unsupported type: {item_type}"}

    if not item_path or not item_id:
        logger.warning("Webhook missing Path or ItemId")
        return {"status": "error", "reason": "missing Path or ItemId"}

    # Apply path mapping for local path
    cfg = get_config()
    local_path = _apply_path_mapping(item_path, cfg.path_mappings)

    # Check existing subtitle / internal stream before creating task
    media_dir = str(Path(local_path).parent)
    media_name = Path(local_path).stem
    from core.subtitle_checker import find_existing_subtitle, is_valid_subtitle
    from core.audio import has_internal_subtitle

    existing = find_existing_subtitle(media_dir, media_name, cfg.target_language)
    if existing and is_valid_subtitle(existing):
        logger.info("Valid UTF-8 subtitle exists: %s, skipping", existing)
        return {"status": "skipped", "reason": "subtitle already exists"}

    if await has_internal_subtitle(local_path):
        logger.info("Media has internal subtitle stream, skipping")
        return {"status": "skipped", "reason": "internal subtitle"}

    task_manager.create_task(
        video_path=local_path,
        item_id=item_id,
        item_type=item_type,
        item_name=item_name,
        pipeline_type="webhook",
    )
    return {"status": "accepted", "item": item_name}


# --- 后台处理 ---

def _apply_path_mapping(jellyfin_path: str, mappings: dict[str, str]) -> str:
    """将 Jellyfin 路径映射为本地路径。"""
    for src, dst in mappings.items():
        if jellyfin_path.startswith(src):
            return jellyfin_path.replace(src, dst, 1)
    return jellyfin_path


# --- 启动 ---

load_config()
