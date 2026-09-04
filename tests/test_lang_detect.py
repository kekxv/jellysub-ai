"""语言识别模块单元测试。"""

from core.lang_detect import detect_language


def test_detect_english():
    text = ("Hello everyone, this is the first subtitle. What are you doing today? "
            "We will talk about the things that happened.")
    assert detect_language(text) == "en"


def test_detect_chinese():
    assert detect_language("大家晚上好，这是第一条字幕。我们今天要聊一些事情。") == "zh"


def test_detect_japanese():
    assert detect_language("こんにちは、最初の字幕です。今日は何かについて話します。") == "ja"


def test_detect_korean():
    assert detect_language("안녕하세요, 첫 번째 자막입니다. 오늘은 무언가에 대해 이야기합니다.") == "ko"


def test_detect_russian():
    assert detect_language("Привет всем, это первый субтитр. Мы будем говорить о чём-то.") == "ru"


def test_detect_arabic():
    assert detect_language("مرحبا بالجميع، هذا هو الترجمة الأولى.") == "ar"


def test_detect_french():
    assert detect_language("Bonjour à tous, ceci est le premier sous-titre. Nous allons parler.") == "fr"


def test_detect_german():
    assert detect_language("Hallo zusammen, dies ist der erste Untertitel. Wir werden darüber sprechen.") == "de"


def test_detect_empty_text_falls_back_to_hint():
    assert detect_language("", hint="ja") == "ja"


def test_detect_uses_hint_for_ambiguous_latin():
    # 无脚本、无可识别特征词时，使用 hint
    assert detect_language("12345", hint="fr") == "fr"
