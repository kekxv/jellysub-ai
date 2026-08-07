"""Security configuration validation tests."""

import pytest

from env_config import validate_security_config


def test_validate_security_config_rejects_default_admin_credentials():
    """Default admin credentials must not be accepted at application startup."""
    with pytest.raises(RuntimeError, match="ADMIN_PASSWORD"):
        validate_security_config("admin", "admin", "a" * 32)
