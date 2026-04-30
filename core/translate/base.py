"""翻译引擎 — 基础类型 + 共享工具。"""

import json
import logging
import re

logger = logging.getLogger("uvicorn.error")


class TranslateEngine:
    """翻译引擎基类。"""

    def translate_batch(self, texts: list[str], target_lang: str,
                        prompt_format: str = "json", thinking: bool = False,
                        context: str = "", source_lang: str = "") -> list[str] | None:
        """翻译一批文本，返回翻译后的字符串列表。
        context: 可选上下文文本，帮助模型理解上下文含义。
        source_lang: 源语言代码，用于 prompt 中指明语言。
        """
        raise NotImplementedError

    def preferred_format(self) -> str:
        """返回引擎偏好的输出格式：'numbered'（本地）或 'json'（在线 API）。"""
        return "json"


# =========================================================================== #
#  共享常量 / 工具
# =========================================================================== #

_LANG_GROUP = {
    "zh-CN": {"zh"},
    "zh-TW": {"zh"},
    "zh": {"zh"},
    "en": {"en"},
    "ja": {"ja"},
    "ko": {"ko"},
    "fr": {"fr"},
    "de": {"de"},
    "es": {"es"},
    "ru": {"ru"},
}

_BATCH_SIZE = 5

_INSTRUCTION = """You are a professional subtitle translator. Translate the given text from English to Chinese.

Rules:
1. Return ONLY a JSON array of translated strings, same length as input.
2. No explanations, no markdown, no extra text.
3. Keep translations concise and natural for subtitles.

Input JSON array:"""

_JSON_ARRAY_PATTERN = re.compile(r"\[[\s\S]*\]")


def parse_json_output(content: str, expected_count: int) -> list[str] | None:
    """解析 JSON 数组格式输出。"""
    # 移除 markdown 代码块包裹
    if "```" in content:
        parts = content.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("[") and "]" in part:
                content = part
                break

    # 尝试寻找第一个 [ 和最后一个 ] 之间的内容
    start_idx = content.find("[")
    end_idx = content.rfind("]")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        potential_json = content[start_idx : end_idx + 1]
        try:
            translated = json.loads(potential_json)
            if isinstance(translated, list):
                return [str(t) for t in translated]
        except json.JSONDecodeError:
            pass

    # 尝试正则提取所有引号内的字符串（兜底方案）
    results = re.findall(r'"((?:[^"\\]|\\.)*)"', content)
    if results:
        # 如果数量匹配，或者比预期多，尝试筛选
        if len(results) == expected_count:
            return results
        if len(results) > expected_count:
            # 排除掉可能出现在 prompt 中的原文字符串
            return results[-expected_count:]

    try:
        translated = json.loads(content)
        if not isinstance(translated, list):
            return None
        return [str(t) for t in translated]
    except json.JSONDecodeError:
        # 模型输出被截断，尝试从残缺 JSON 中提取字符串
        return _extract_strings_from_truncated(content, expected_count)


def _extract_strings_from_truncated(content: str, expected_count: int) -> list[str] | None:
    """从被截断的 JSON 数组中提取翻译文本。"""
    results = re.findall(r'"((?:[^"\\]|\\.)*)"', content)
    if not results:
        return None
    # 去掉可能的第一个非翻译项（如 prompt 残留）
    if len(results) > expected_count:
        results = results[-expected_count:]
    return [s for s in results] if results else None


def parse_numbered_output(content: str, expected_count: int) -> list[str] | None:
    """解析编号格式输出: "1. 你好世界\n2. 这是测试"。"""
    results = []
    for line in content.split("\n"):
        line = line.strip()
        if not line:
            continue
        cleaned = re.sub(r"^\d+[.、\s]*\s*", "", line)
        if cleaned:
            results.append(cleaned)
    if len(results) == expected_count:
        return results
    # 数量不匹配：取前 N 个或后 N 个
    if len(results) >= expected_count:
        return results[-expected_count:]
    if results:
        # 不足时用 JSON 格式兜底
        return parse_json_output(content, expected_count)
    return None
