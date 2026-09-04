"""轻量级字幕语言识别 — 无外部依赖。

优先级：
1. 脚本区间判断（中日韩 / 俄 / 阿拉伯 / 泰 / 希腊 等，可靠）。
2. 拉丁字母语言用常见词投票（en/fr/de/es/it/pt/nl，尽力而为）。
3. 仍不确定时回退到调用方提供的 hint（如文件名语言标签）。

返回 ISO 639-1 代码（如 'en'、'zh'、'ja'），无法判断时返回 ''。
"""

import re
from collections import Counter

# 拼音/拉丁字母字符
_LATIN_RE = re.compile(r"[A-Za-z]")

# 各语言的高区分度常见词（拉丁字母语言）。尽量避开跨语言通用的短词。
_LATIN_WORDS: dict[str, set[str]] = {
    "en": {
        "the", "this", "that", "with", "have", "from", "you", "are", "what",
        "will", "would", "there", "their", "been", "because", "about", "people",
        "they", "when", "your", "were", "should", "could", "into",
    },
    "fr": {
        "les", "des", "dans", "pour", "avec", "sont", "mais", "elle", "elle",
        "être", "nous", "vous", "cette", "qui", "aux", "est", "une", "pas",
    },
    "de": {
        "und", "nicht", "mit", "von", "auch", "sind", "wir", "aber", "den",
        "dem", "auf", "ist", "ein", "eine", "das", "die",
    },
    "es": {
        "para", "pero", "como", "más", "este", "esta", "todo", "sobre", "los",
        "las", "hay", "está", "son", "una", "por", "que",
    },
    "it": {
        "sono", "perché", "anche", "della", "questo", "questa", "non", "con",
        "per", "una", "che", "come", "più",
    },
    "pt": {
        "não", "mais", "também", "isso", "são", "você", "muito", "para", "por",
        "uma", "que", "com", "os", "as",
    },
    "nl": {
        "het", "een", "niet", "maar", "zijn", "wat", "deze", "naar", "bij",
        "van", "voor", "ze", "je",
    },
}


def _script_lang(text: str) -> str:
    """根据字符脚本判断非拉丁语言。"""
    han = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    hiragana = sum(1 for c in text if "\u3040" <= c <= "\u309f")
    katakana = sum(1 for c in text if "\u30a0" <= c <= "\u30ff")
    hangul = sum(1 for c in text if "\uac00" <= c <= "\ud7af")
    cyrillic = sum(1 for c in text if "\u0400" <= c <= "\u04ff")
    greek = sum(1 for c in text if "\u0370" <= c <= "\u03ff")
    arabic = sum(1 for c in text if "\u0600" <= c <= "\u06ff")
    thai = sum(1 for c in text if "\u0e00" <= c <= "\u0e7f")
    devanagari = sum(1 for c in text if "\u0900" <= c <= "\u097f")

    if han >= 2:
        # 汉字为主；若混入假名则更可能是日语
        if hiragana or katakana > han * 0.1:
            return "ja"
        return "zh"
    if hangul >= 2:
        return "ko"
    if hiragana or katakana:
        return "ja"
    if cyrillic >= 2:
        return "ru"
    if arabic >= 2:
        return "ar"
    if thai >= 2:
        return "th"
    if greek >= 2:
        return "el"
    if devanagari >= 2:
        return "hi"
    return ""


def _latin_vote(text: str) -> str:
    """拉丁字母语言常见词投票，返回票数最高的语言代码。"""
    words = re.findall(r"[A-Za-zÀ-ÿ]+", text.lower())
    if not words:
        return ""
    counts: Counter[str] = Counter()
    for lang, vocab in _LATIN_WORDS.items():
        hits = sum(1 for w in set(words) if w in vocab)
        if hits:
            counts[lang] = hits
    if not counts:
        return ""
    best, best_count = counts.most_common(1)[0]
    # 只要有一个高区分度词命中就给出判断，用户可再手动覆盖
    if best_count >= 1:
        return best
    return ""


def detect_language(text: str, hint: str = "", sample_chars: int = 20000) -> str:
    """识别字幕文本的语言，返回 ISO 639-1 代码。

    text: 字幕文本（可拼接多条）。hint: 文件名推断的语言代码，作为兜底。
    """
    if not text:
        return hint or ""
    sample = text[:sample_chars]

    script_lang = _script_lang(sample)
    if script_lang:
        return script_lang

    vote = _latin_vote(sample)
    if vote:
        return vote

    # 拉丁文本且无法确定：若 hint 是已知语言，用 hint；否则默认英语
    if hint:
        hint_norm = hint.lower()[:2]
        if hint_norm in _LATIN_WORDS or hint_norm in {"zh", "ja", "ko", "ru", "ar", "th", "el"}:
            return hint_norm
    return "en"
