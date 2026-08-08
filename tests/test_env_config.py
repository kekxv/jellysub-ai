"""Security configuration validation tests."""

import pytest

from env_config import validate_security_config


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
