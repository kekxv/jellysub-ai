"""Security configuration validation tests."""

import importlib
import logging

import pytest

import env_config
from env_config import parse_cors_origins, validate_security_config


def test_validate_security_config_rejects_default_admin_credentials():
    """Default admin credentials must not be accepted at application startup."""
    with pytest.raises(RuntimeError, match="ADMIN_PASSWORD"):
        validate_security_config("admin", "admin", "a" * 32)


@pytest.mark.parametrize(
    ("username", "password"),
    [
        ("your-admin-username", "replace-with-a-strong-password"),
        ("your_admin_username", "your_secure_password"),
        ("myuser", "mypassword"),
    ],
)
def test_validate_security_config_rejects_published_admin_placeholders(username, password):
    """Credentials copied from published examples must not pass startup validation."""
    with pytest.raises(RuntimeError, match="ADMIN_USERNAME.*ADMIN_PASSWORD"):
        validate_security_config(username, password, "s" * 32)


@pytest.mark.parametrize(
    "session_secret",
    [
        "replace-with-a-random-32-character-minimum-secret",
        "generate-a-random-secret-with-at-least-32-characters",
    ],
)
def test_validate_security_config_rejects_published_session_placeholders(session_secret):
    """Session secrets copied from published examples must not sign sessions."""
    with pytest.raises(RuntimeError, match="SESSION_SECRET"):
        validate_security_config("operator", "a genuinely strong password", session_secret)


def test_validate_security_config_requires_totp_outside_development():
    """Production startup must not silently accept password-only authentication."""
    with pytest.raises(RuntimeError, match="TOTP_SECRET"):
        validate_security_config(
            "operator",
            "a genuinely strong password",
            "s" * 32,
            totp_secret="",
            development_mode=False,
        )


def test_validate_security_config_allows_missing_totp_with_development_override():
    """Only an explicit development setting may permit password-only authentication."""
    validate_security_config(
        "operator",
        "a genuinely strong password",
        "s" * 32,
        totp_secret="",
        development_mode=True,
    )


def test_validate_security_config_allows_local_defaults_with_development_override():
    """Explicit development mode must permit an empty local security setup."""
    validate_security_config(
        "admin",
        "admin",
        "",
        totp_secret="",
        development_mode=True,
    )


def test_development_mode_omits_production_security_warnings(monkeypatch, caplog):
    """Intentional local defaults must not be reported as production risks."""
    with monkeypatch.context() as environment:
        environment.setenv("DEVELOPMENT_MODE", "true")
        environment.setenv("ADMIN_USERNAME", "admin")
        environment.setenv("ADMIN_PASSWORD", "admin")
        environment.setenv("TOTP_SECRET", "")
        environment.setenv("SESSION_SECRET", "")
        caplog.set_level(logging.WARNING, logger="uvicorn.error")
        caplog.clear()
        importlib.reload(env_config)

        messages = [record.getMessage() for record in caplog.records]
        assert not any(
            marker in message
            for message in messages
            for marker in ("默认管理员凭据", "TOTP", "SESSION_SECRET")
        )

    importlib.reload(env_config)


def test_parse_cors_origins_returns_only_explicit_origins():
    """Comma-separated CORS configuration must not grant every browser origin."""
    assert parse_cors_origins("https://admin.example, , https://app.example ") == (
        "https://admin.example",
        "https://app.example",
    )


def test_parse_cors_origins_rejects_wildcard_origin():
    """A wildcard environment value would reopen the browser boundary."""
    with pytest.raises(ValueError, match="wildcard"):
        parse_cors_origins("https://admin.example,*")
