"""内存监控工具。"""

import logging
import os
import psutil

logger = logging.getLogger("uvicorn.error")

class MemoryExceededError(RuntimeError):
    """内存占用超过设定上限异常。"""
    pass

def get_current_memory_gb() -> float:
    """获取当前进程及其子进程的总内存占用 (GB)。"""
    try:
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        rss = mem_info.rss
        
        # 加上子进程内存 (ffmpeg 等)
        for child in process.children(recursive=True):
            try:
                rss += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
                
        return rss / (1024 ** 3)
    except Exception:
        return 0.0

def check_memory_limit():
    """检查内存是否超过上限，超过则抛出异常。"""
    from env_config import MAX_MEMORY_GB
    if MAX_MEMORY_GB <= 0:
        return

    current_mem = get_current_memory_gb()
    if current_mem > MAX_MEMORY_GB:
        msg = f"Memory limit exceeded: {current_mem:.2f}GB > {MAX_MEMORY_GB:.2f}GB"
        logger.error(msg)
        raise MemoryExceededError(msg)
