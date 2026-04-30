"""Jellyfin REST API 客户端。"""

import logging

import httpx

logger = logging.getLogger("uvicorn.error")


class JellyfinClient:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._headers = {"X-Emby-Token": api_key}

    async def get_item_info(self, item_id: str) -> dict | None:
        """获取媒体项信息。"""
        url = f"{self.base_url}/Items/{item_id}"
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(url, headers=self._headers, timeout=10.0)
                resp.raise_for_status()
                return resp.json()
        except Exception:
            logger.exception("Failed to get item info for %s", item_id)
            return None

    async def refresh_item(self, item_id: str) -> bool:
        """强制刷新指定媒体项。"""
        url = f"{self.base_url}/Items/{item_id}/Refresh"
        payload = {
            "MetadataRefreshMode": "FullRefresh",
            "ImageRefreshMode": "FullRefresh",
            "ReplaceAllMetadata": False,
            "ReplaceAllImages": False,
        }
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    url, headers=self._headers, json=payload, timeout=30.0
                )
                resp.raise_for_status()
                logger.info("Refreshed Jellyfin item %s", item_id)
                return True
        except Exception:
            logger.exception("Failed to refresh item %s", item_id)
            return False
