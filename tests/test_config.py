from __future__ import annotations

from pathlib import Path

import pytest

from oepnai_image.cli import read_env
from oepnai_image.config import (
    MissingApiKeyError,
    ProviderConfigError,
    load_provider_registry,
    resolve_provider,
)


def write_config(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "providers.toml"
    path.write_text(content, encoding="utf-8")
    return path


def provider_config(
    *,
    provider_id: str = "openai",
    key_env: str = "CONFIG_KEY",
    model: str = "config-model",
    base_url: str = "https://config.example/v1",
    default_provider: str | None = "openai",
) -> str:
    default_line = "" if default_provider is None else f'default_provider = "{default_provider}"\n'
    return "\n".join(
        [
            "schema_version = 1",
            default_line.rstrip("\n"),
            f"[providers.{provider_id}]",
            'protocol = "openai-images"',
            f'default_model = "{model}"',
            f"[providers.{provider_id}.targets.primary]",
            f'base_url = "{base_url}"',
            f'api_key_env = "{key_env}"',
            "priority = 10",
            "enabled = true",
            "",
        ],
    )


def toml_config(*lines: str) -> str:
    return "\n".join([*lines, ""])


def test_read_env_returns_the_legacy_environment_contract(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-value")
    monkeypatch.setenv("OPENAI_BASE_URL", " https://legacy.example/v1 ")
    monkeypatch.setenv("OPENAI_IMAGE_MODEL", " legacy-model ")
    monkeypatch.setenv("OPENAI_IMAGE_TRANSPORT", " HTTP ")
    monkeypatch.setenv("OPENAI_TIMEOUT_SECONDS", " 42 ")

    result = read_env()

    assert result == {
        "api_key": "test-key-value",
        "base_url": "https://legacy.example/v1",
        "image_model": "legacy-model",
        "image_transport": "http",
        "timeout": "42",
    }


def test_load_provider_registry_uses_the_explicit_override_path(tmp_path) -> None:
    path = write_config(tmp_path, provider_config())

    registry = load_provider_registry(path, environment={"OPENAI_IMAGE_CONFIG": "ignored.toml"})

    assert registry.schema_version == 1
    assert [provider.id for provider in registry.providers] == ["openai"]
    assert registry.providers[0].targets[0].base_url == "https://config.example/v1"


def test_load_provider_registry_accepts_openai_chat_images_protocol(tmp_path) -> None:
    # Given: an image provider exposed through an OpenAI-compatible chat endpoint.
    path = write_config(
        tmp_path,
        provider_config().replace('protocol = "openai-images"', 'protocol = "openai-chat-images"'),
    )

    # When: the provider registry is loaded.
    registry = load_provider_registry(path)

    # Then: the chat-image protocol is preserved for routing.
    assert registry.providers[0].protocol == "openai-chat-images"


def test_load_provider_registry_uses_openai_image_config_override(monkeypatch, tmp_path) -> None:
    path = write_config(tmp_path, provider_config(provider_id="override"))
    monkeypatch.setenv("OPENAI_IMAGE_CONFIG", str(path))

    registry = load_provider_registry(home=tmp_path / "unused-home")

    assert [provider.id for provider in registry.providers] == ["override"]


def test_absent_default_config_falls_back_to_legacy_environment(tmp_path) -> None:
    environment = {
        "OPENAI_API_KEY": "legacy-secret-sentinel",
        "OPENAI_BASE_URL": "https://legacy.example/v1",
        "OPENAI_IMAGE_MODEL": "legacy-model",
    }

    resolved = resolve_provider(environment=environment, home=tmp_path)

    assert resolved.provider_id == "openai"
    assert resolved.target_id == "legacy"
    assert resolved.base_url == "https://legacy.example/v1"
    assert resolved.default_model == "legacy-model"
    assert resolved.api_key_env == "OPENAI_API_KEY"
    assert "legacy-secret-sentinel" not in repr(resolved)


def test_malformed_toml_is_rejected_without_echoing_input(tmp_path) -> None:
    path = write_config(tmp_path, 'schema_version = 1\nmalformed = "secret-sentinel\n')

    with pytest.raises(ProviderConfigError) as error:
        load_provider_registry(path)

    assert "secret-sentinel" not in str(error.value)


def test_duplicate_provider_ids_are_rejected(tmp_path) -> None:
    path = write_config(
        tmp_path,
        toml_config(
            "schema_version = 1",
            "[[providers]]",
            'id = "duplicate"',
            'protocol = "openai-images"',
            'default_model = "model"',
            "[[providers.targets]]",
            'id = "primary"',
            'base_url = "https://one.example/v1"',
            'api_key_env = "KEY_ONE"',
            "[[providers]]",
            'id = "duplicate"',
            'protocol = "openai-images"',
            'default_model = "model"',
            "[[providers.targets]]",
            'id = "primary"',
            'base_url = "https://two.example/v1"',
            'api_key_env = "KEY_TWO"',
        ),
    )

    with pytest.raises(ProviderConfigError):
        load_provider_registry(path)


def test_duplicate_target_ids_are_rejected(tmp_path) -> None:
    path = write_config(
        tmp_path,
        toml_config(
            "schema_version = 1",
            "[[providers]]",
            'id = "provider"',
            'protocol = "openai-images"',
            'default_model = "model"',
            "[[providers.targets]]",
            'id = "duplicate"',
            'base_url = "https://one.example/v1"',
            'api_key_env = "KEY_ONE"',
            "[[providers.targets]]",
            'id = "duplicate"',
            'base_url = "https://two.example/v1"',
            'api_key_env = "KEY_TWO"',
        ),
    )

    with pytest.raises(ProviderConfigError):
        load_provider_registry(path)


def test_unknown_fields_are_rejected_without_echoing_secret_values(tmp_path) -> None:
    path = write_config(
        tmp_path,
        provider_config().replace(
            'api_key_env = "CONFIG_KEY"',
            'api_key = "raw-secret-sentinel"',
        ),
    )

    with pytest.raises(ProviderConfigError) as error:
        load_provider_registry(path)

    assert "raw-secret-sentinel" not in str(error.value)


@pytest.mark.parametrize(
    "base_url",
    ["not-a-url", "ftp://config.example/v1", "https://user:pass@config.example/v1"],
)
def test_invalid_urls_are_rejected(tmp_path, base_url: str) -> None:
    path = write_config(tmp_path, provider_config(base_url=base_url))

    with pytest.raises(ProviderConfigError):
        load_provider_registry(path)


def test_unsupported_schema_version_is_rejected(tmp_path) -> None:
    path = write_config(tmp_path, provider_config().replace("schema_version = 1", "schema_version = 2"))

    with pytest.raises(ProviderConfigError):
        load_provider_registry(path)


@pytest.mark.parametrize(
    ("runtime", "job", "batch", "expected"),
    [
        ("runtime", "job", "batch", "runtime"),
        (None, "job", "batch", "job"),
        (None, None, "batch", "batch"),
        (None, None, None, "configured"),
    ],
)
def test_provider_precedence_selects_the_highest_priority_source(
    tmp_path,
    runtime: str | None,
    job: str | None,
    batch: str | None,
    expected: str,
) -> None:
    path = write_config(
        tmp_path,
        toml_config(
            "schema_version = 1",
            'default_provider = "configured"',
            "[[providers]]",
            'id = "runtime"',
            'protocol = "openai-images"',
            'default_model = "runtime-model"',
            "[[providers.targets]]",
            'id = "primary"',
            'base_url = "https://runtime.example/v1"',
            'api_key_env = "RUNTIME_KEY"',
            "[[providers]]",
            'id = "job"',
            'protocol = "openai-images"',
            'default_model = "job-model"',
            "[[providers.targets]]",
            'id = "primary"',
            'base_url = "https://job.example/v1"',
            'api_key_env = "JOB_KEY"',
            "[[providers]]",
            'id = "batch"',
            'protocol = "openai-images"',
            'default_model = "batch-model"',
            "[[providers.targets]]",
            'id = "primary"',
            'base_url = "https://batch.example/v1"',
            'api_key_env = "BATCH_KEY"',
            "[[providers]]",
            'id = "configured"',
            'protocol = "openai-images"',
            'default_model = "configured-model"',
            "[[providers.targets]]",
            'id = "primary"',
            'base_url = "https://configured.example/v1"',
            'api_key_env = "CONFIGURED_KEY"',
        ),
    )
    environment = {
        "RUNTIME_KEY": "runtime-secret-sentinel",
        "JOB_KEY": "job-secret-sentinel",
        "BATCH_KEY": "batch-secret-sentinel",
        "CONFIGURED_KEY": "configured-secret-sentinel",
    }

    resolved = resolve_provider(
        config_path=path,
        runtime_provider=runtime,
        job_provider=job,
        batch_default_provider=batch,
        environment=environment,
    )

    assert resolved.provider_id == expected
    assert all(secret not in repr(resolved) for secret in environment.values())


def test_legacy_environment_is_the_last_precedence_fallback(tmp_path) -> None:
    environment = {
        "OPENAI_API_KEY": "legacy-secret-sentinel",
        "OPENAI_BASE_URL": "https://legacy.example/v1",
        "OPENAI_IMAGE_MODEL": "legacy-model",
    }

    resolved = resolve_provider(
        config_path=write_config(tmp_path, provider_config(default_provider=None)),
        environment=environment,
    )

    assert resolved.provider_id == "openai"
    assert resolved.default_model == "legacy-model"


def test_missing_api_key_is_checked_at_request_time_and_redacted(tmp_path) -> None:
    path = write_config(tmp_path, provider_config(key_env="MISSING_KEY"))
    registry = load_provider_registry(path)

    with pytest.raises(MissingApiKeyError) as error:
        resolve_provider(registry=registry, environment={})

    assert "MISSING_KEY" in str(error.value)
    assert "raw-secret-sentinel" not in str(error.value)
