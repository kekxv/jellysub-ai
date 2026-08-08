"""Tests for local model loading safety."""

import inspect
import sys
import types


def test_local_translation_model_accepts_configured_modelscope_repository(
    monkeypatch, tmp_path
):
    """Configured ModelScope repositories must not be rejected before loading."""
    from core.translate import local
    import env_config

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            assert model_name == "/models/Qwen3-1.7B"
            return cls()

    class FakeModel:
        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            assert model_name == "/models/Qwen3-1.7B"
            return cls()

        def to(self, device):
            assert device == "cpu"
            return self

    fake_snapshot = types.ModuleType("modelscope.hub.snapshot_download")
    fake_snapshot.snapshot_download = lambda name, cache_dir: "/models/Qwen3-1.7B"
    fake_hub = types.ModuleType("modelscope.hub")
    fake_modelscope = types.ModuleType("modelscope")
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = FakeTokenizer
    fake_transformers.AutoModelForCausalLM = FakeModel

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(env_config, "MODEL_SOURCE", "modelscope")
    monkeypatch.setattr(local, "_local_model", None)
    monkeypatch.setitem(sys.modules, "modelscope", fake_modelscope)
    monkeypatch.setitem(sys.modules, "modelscope.hub", fake_hub)
    monkeypatch.setitem(sys.modules, "modelscope.hub.snapshot_download", fake_snapshot)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    loaded = local.load_local_model("Qwen/Qwen3-1.7B", device="cpu")

    assert isinstance(loaded["tokenizer"], FakeTokenizer)
    assert isinstance(loaded["model"], FakeModel)


def test_transformers_loader_never_enables_remote_code():
    from core.translate import local

    assert "trust_remote_code=False" in inspect.getsource(local.load_local_model)
