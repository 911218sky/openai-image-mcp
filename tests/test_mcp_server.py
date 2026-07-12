from __future__ import annotations

from pathlib import Path

from oepnai_image.mcp_server import list_image_providers


def test_list_image_providers_reports_configured_targets_without_credentials(
    tmp_path: Path,
    monkeypatch,
) -> None:
    # Given: a local provider registry with a credential name, but no credential value.
    config_path = tmp_path / "providers.toml"
    config_path.write_text(
        "\n".join(
            [
                "schema_version = 1",
                'default_provider = "gemini"',
                "[providers.gemini]",
                'protocol = "gemini-native"',
                'default_model = "gemini-image"',
                "[providers.gemini.targets.primary]",
                'base_url = "https://generativelanguage.example/v1beta"',
                'api_key_env = "GEMINI_API_KEY"',
                "priority = 1",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENAI_IMAGE_CONFIG", str(config_path))

    # When: MCP asks for available image providers.
    result = list_image_providers()

    # Then: routing metadata is visible while secrets remain absent.
    assert result["default_provider"] == "gemini"
    assert result["providers"] == [
        {
            "id": "gemini",
            "protocol": "gemini-native",
            "default_model": "gemini-image",
            "targets": [
                {
                    "id": "primary",
                    "base_url": "https://generativelanguage.example/v1beta",
                    "api_key_env": "GEMINI_API_KEY",
                    "priority": 1,
                    "enabled": True,
                }
            ],
            "enabled": True,
        }
    ]
