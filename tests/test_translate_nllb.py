"""NLLB + fastText translation engine regressions."""

from unittest.mock import MagicMock, patch
import sys

import pytest

from core.translate import get_translate_engine
from core.translate.nllb import (
    NLLBTranslateEngine,
    fasttext_label_to_nllb,
    normalize_fasttext_label,
    _nllb_generate,
    load_fasttext_model,
    load_nllb_model,
)


def test_fasttext_labels_normalize_to_nllb_language_tags():
    assert normalize_fasttext_label("__label__eng_Latn") == "eng_latn"
    assert fasttext_label_to_nllb("__label__eng_Latn") == "eng_Latn"
    assert fasttext_label_to_nllb("__label__zho") == "zho_Hans"
    assert fasttext_label_to_nllb("fr") == "fra_Latn"


def test_nllb_generate_sets_source_and_target_prefix():
    tokenizer = MagicMock()
    tokenizer.return_value = {"input_ids": [[1, 2], [3, 4]]}
    tokenizer.convert_ids_to_tokens.side_effect = [["eng_Latn", "Hello", "</s>"], ["eng_Latn", "Hi", "</s>"]]
    tokenizer.unk_token_id = 0
    tokenizer.convert_tokens_to_ids.side_effect = lambda value: 1 if isinstance(value, str) else [9]
    tokenizer.decode.side_effect = ["你好", "嗨"]
    model = MagicMock()
    model.translate_batch.return_value = [
        type("Result", (), {"hypotheses": [["zho_Hans", "你好"]]})(),
        type("Result", (), {"hypotheses": [["zho_Hans", "嗨"]]})(),
    ]
    result = _nllb_generate({"tokenizer": tokenizer, "model": model},
                            ["Hello", "Hi"], "eng_Latn", "zho_Hans")
    assert result == ["你好", "嗨"]
    assert tokenizer.src_lang == "eng_Latn"
    assert model.translate_batch.call_args.kwargs["target_prefix"] == [["zho_Hans"], ["zho_Hans"]]


def test_factory_selects_nllb_for_nllb_model_name():
    engine = get_translate_engine(model_name="facebook/nllb-200-distilled-600M")
    assert isinstance(engine, NLLBTranslateEngine)


def test_fasttext_offline_mode_reports_download_instructions(tmp_path, monkeypatch):
    monkeypatch.setenv("OFFLINE_MODELS", "true")
    with pytest.raises(FileNotFoundError, match=r"OFFLINE_MODELS.*lid\.176\.ftz"):
        load_fasttext_model(str(tmp_path / "lid.176.ftz"))


def test_fasttext_downloads_missing_model_when_online(tmp_path, monkeypatch):
    target = tmp_path / "lid.176.ftz"
    monkeypatch.delenv("OFFLINE_MODELS", raising=False)
    fake_fasttext = MagicMock()
    with patch.dict(sys.modules, {"fasttext": fake_fasttext}), patch("core.translate.nllb.urlretrieve") as download:
        load_fasttext_model(str(target))
    download.assert_called_once()
    assert download.call_args.args[1] == str(target)


def test_nllb_loader_downloads_only_int8_snapshot(monkeypatch, tmp_path):
    import core.translate.nllb as nllb
    monkeypatch.setattr(nllb, "_nllb_model", None)
    monkeypatch.setenv("MODEL_CACHE", str(tmp_path))
    monkeypatch.setenv("OFFLINE_MODELS", "true")
    with patch("huggingface_hub.snapshot_download", return_value=str(tmp_path)) as download, \
         patch("transformers.AutoTokenizer.from_pretrained") as tokenizer, \
         patch("ctranslate2.Translator") as translator:
        loaded = load_nllb_model(device="cpu")
    assert loaded["model"] is translator.return_value
    assert download.call_args.kwargs["repo_id"].endswith("ct2-int8")
    assert download.call_args.kwargs["local_files_only"] is True
    assert "pytorch_model.bin" not in download.call_args.kwargs["allow_patterns"]
    assert translator.call_args.kwargs["compute_type"] == "int8"
    assert tokenizer.call_args.kwargs["local_files_only"] is True


def test_nllb_offline_missing_cache_has_actionable_error(monkeypatch):
    import core.translate.nllb as nllb
    monkeypatch.setattr(nllb, "_nllb_model", None)
    monkeypatch.setenv("OFFLINE_MODELS", "true")
    with patch("huggingface_hub.snapshot_download", side_effect=OSError("missing")):
        with pytest.raises(RuntimeError, match="offline cache"):
            load_nllb_model(device="cpu")


def test_fasttext_detection_uses_batch_api_for_numpy2_compatibility():
    from core.translate.nllb import detect_fasttext_language

    class Detector:
        def predict(self, texts, k):
            if not isinstance(texts, list):
                raise ValueError('single-string prediction is incompatible with NumPy 2')
            return [['__label__en']], [[0.99]]

    assert detect_fasttext_language('Hello\nworld', Detector()) == 'en'


def test_default_fasttext_uses_compact_ftz_in_configured_cache(tmp_path, monkeypatch):
    from core.translate.nllb import FASTTEXT_MODEL_DEFAULT, FASTTEXT_MODEL_URL, _fasttext_path

    monkeypatch.setenv('MODEL_CACHE', str(tmp_path))
    assert FASTTEXT_MODEL_URL.endswith('/lid.176.ftz')
    assert _fasttext_path(FASTTEXT_MODEL_DEFAULT) == tmp_path / 'lid.176.ftz'
