from __future__ import annotations

from pathlib import Path

import pytest


def test_public_output_root_is_configured_by_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from oepnai_image import mcp_server

    public_root = tmp_path / "public" / "images"
    monkeypatch.setenv("OPENAI_IMAGE_PUBLIC_IMAGES_ROOT", str(public_root))

    assert mcp_server._default_output_dir(None) == public_root / "openai-image-mcp"


def test_public_output_integration_is_disabled_without_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from oepnai_image import mcp_server

    monkeypatch.delenv("OPENAI_IMAGE_PUBLIC_IMAGES_ROOT", raising=False)

    assert mcp_server._public_images_root() is None
    assert mcp_server._default_output_dir(None) is None


def test_streamable_http_settings_are_environment_driven(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from oepnai_image import mcp_server

    monkeypatch.setenv("OPENAI_IMAGE_MCP_HOST", "0.0.0.0")
    monkeypatch.setenv("OPENAI_IMAGE_MCP_PORT", "8011")
    monkeypatch.setenv("OPENAI_IMAGE_MCP_PATH", "/mcp")
    monkeypatch.setenv("OPENAI_IMAGE_MCP_ALLOWED_HOSTS", "image-sidecar:8011")

    settings = mcp_server._streamable_http_settings()

    assert settings.host == "0.0.0.0"
    assert settings.port == 8011
    assert settings.streamable_http_path == "/mcp"
    assert "image-sidecar:8011" in settings.transport_security.allowed_hosts


def test_internal_provider_requires_explicit_insecure_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from oepnai_image import mcp_server

    monkeypatch.setenv("OPENAI_BASE_URL", "http://image-provider:8080/v1")
    monkeypatch.setenv("OPENAI_IMAGE_ALLOW_INSECURE_BASE_URL", "true")

    assert mcp_server._request_env({"headers": {"x-openai-base-url": "http://image-provider:8080/v1"}})[
        "base_url"
    ] == "http://image-provider:8080/v1"


def test_streamable_http_path_must_start_with_slash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from oepnai_image import mcp_server

    monkeypatch.setenv("OPENAI_IMAGE_MCP_PATH", "mcp")

    with pytest.raises(RuntimeError, match="must start with '/'"):
        mcp_server._streamable_http_settings()


def test_cli_output_directory_uses_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from oepnai_image.cli import default_output_root

    monkeypatch.setenv("OPENAI_IMAGE_OUTPUT_DIR", str(tmp_path / "images"))

    assert default_output_root() == tmp_path / "images"
