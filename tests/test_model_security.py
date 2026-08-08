"""Tests for the local-model trust boundary."""

import inspect

import pytest

from env_config import validate_local_model


def test_local_model_allowlist_rejects_unknown_repository():
    with pytest.raises(ValueError, match="not approved"):
        validate_local_model("attacker/unreviewed-model")


def test_local_model_allowlist_accepts_supported_models():
    assert validate_local_model("Qwen/Qwen3-0.6B") == "Qwen/Qwen3-0.6B"
    assert validate_local_model("iic/SenseVoiceSmall") == "iic/SenseVoiceSmall"


def test_transformers_loader_never_enables_remote_code():
    from core.translate import local

    assert "trust_remote_code=False" in inspect.getsource(local.load_local_model)
