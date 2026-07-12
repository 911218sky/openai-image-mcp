from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import os
from pathlib import Path
import re
import tomllib
from urllib.parse import urlparse


class ProviderConfigError(RuntimeError):
    pass


class MissingApiKeyError(ProviderConfigError):
    pass


@dataclass(frozen=True, slots=True)
class ProviderTarget:
    id: str
    base_url: str
    api_key_env: str
    priority: int
    enabled: bool


@dataclass(frozen=True, slots=True)
class ProviderConfig:
    id: str
    protocol: str
    default_model: str
    targets: tuple[ProviderTarget, ...]
    enabled: bool


@dataclass(frozen=True, slots=True)
class ProviderRegistry:
    schema_version: int
    providers: tuple[ProviderConfig, ...]
    default_provider: str | None


@dataclass(frozen=True, slots=True)
class ResolvedProvider:
    provider_id: str
    target_id: str
    protocol: str
    base_url: str
    default_model: str
    api_key_env: str
    source: str


def load_provider_registry(
    config_path: Path | None = None,
    *,
    environment: Mapping[str, str] | None = None,
    home: Path | None = None,
) -> ProviderRegistry:
    env = os.environ if environment is None else environment
    selected_path, explicit = _select_config_path(config_path, env, home)
    if not selected_path.is_file():
        if explicit:
            raise ProviderConfigError("Provider config file does not exist.")
        return ProviderRegistry(schema_version=1, providers=(), default_provider=None)

    try:
        raw = tomllib.loads(selected_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as exc:
        raise ProviderConfigError("Provider config could not be read or parsed.") from exc
    return _parse_registry(raw)


def resolve_provider(
    *,
    registry: ProviderRegistry | None = None,
    config_path: Path | None = None,
    runtime_provider: str | None = None,
    job_provider: str | None = None,
    batch_default_provider: str | None = None,
    target_id: str | None = None,
    environment: Mapping[str, str] | None = None,
    legacy_env: Mapping[str, str] | None = None,
    home: Path | None = None,
    require_api_key: bool = True,
) -> ResolvedProvider:
    env = os.environ if environment is None else environment
    registry_value = registry or load_provider_registry(
        config_path,
        environment=env,
        home=home,
    )
    candidate_sources = (
        ("runtime", runtime_provider),
        ("job", job_provider),
        ("batch", batch_default_provider),
        ("configured", registry_value.default_provider),
    )
    selected_source, selected_id = next(
        ((source, provider_id) for source, provider_id in candidate_sources if provider_id),
        (None, None),
    )

    if selected_id is None:
        legacy = env if legacy_env is None else legacy_env
        api_key = legacy.get("OPENAI_API_KEY", "").strip()
        if require_api_key and not api_key:
            raise MissingApiKeyError("Environment variable OPENAI_API_KEY is not set.")
        return ResolvedProvider(
            provider_id="openai",
            target_id="legacy",
            protocol="openai-images",
            base_url=legacy.get("OPENAI_BASE_URL", "https://api.openai.com/v1").strip(),
            default_model=legacy.get("OPENAI_IMAGE_MODEL", "gpt-image-2").strip(),
            api_key_env="OPENAI_API_KEY",
            source="legacy",
        )

    provider = next(
        (item for item in registry_value.providers if item.id == selected_id),
        None,
    )
    if provider is None or not provider.enabled:
        raise ProviderConfigError("Selected provider is not available.")
    targets = sorted(
        (target for target in provider.targets if target.enabled),
        key=lambda target: (target.priority, target.id),
    )
    if not targets:
        raise ProviderConfigError("Selected provider has no enabled targets.")
    target = next((item for item in targets if item.id == target_id), targets[0])
    if target_id is not None and target.id != target_id:
        raise ProviderConfigError("Selected provider target is not available.")
    if require_api_key and not env.get(target.api_key_env, "").strip():
        raise MissingApiKeyError(
            f"Environment variable {target.api_key_env} is not set."
        )
    return ResolvedProvider(
        provider_id=provider.id,
        target_id=target.id,
        protocol=provider.protocol,
        base_url=target.base_url,
        default_model=provider.default_model,
        api_key_env=target.api_key_env,
        source=selected_source or "configured",
    )


_ID_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*$")
_ENV_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ROOT_KEYS = {"schema_version", "default_provider", "providers"}
_PROVIDER_KEYS = {"id", "protocol", "default_model", "targets", "enabled"}
_TARGET_KEYS = {"id", "base_url", "api_key_env", "priority", "enabled"}


def _select_config_path(
    config_path: Path | None,
    environment: Mapping[str, str],
    home: Path | None,
) -> tuple[Path, bool]:
    if config_path is not None:
        return config_path.expanduser(), True
    override = environment.get("OPENAI_IMAGE_CONFIG", "").strip()
    if override:
        return Path(override).expanduser(), True
    root = Path.home() if home is None else home
    return root / ".config" / "openai-image-mcp" / "providers.toml", False


def _parse_registry(raw: dict[str, object]) -> ProviderRegistry:
    _reject_unknown(raw, _ROOT_KEYS, "root")
    schema_version = _required_int(raw, "schema_version", "root")
    if schema_version != 1:
        raise ProviderConfigError("Unsupported provider config schema version.")
    default_provider = _optional_string(raw, "default_provider", "root")
    raw_providers = raw.get("providers", {})
    providers = _parse_providers(raw_providers)
    ids = [provider.id for provider in providers]
    if len(ids) != len(set(ids)):
        raise ProviderConfigError("Duplicate provider id.")
    return ProviderRegistry(schema_version, tuple(providers), default_provider)


def _parse_providers(raw: object) -> list[ProviderConfig]:
    if isinstance(raw, dict):
        entries = [{"id": key, **_as_table(value, "provider")} for key, value in raw.items()]
    elif isinstance(raw, list):
        entries = [_as_table(value, "provider") for value in raw]
    else:
        raise ProviderConfigError("providers must be a table or array of tables.")
    return [_parse_provider(entry) for entry in entries]


def _parse_provider(raw: dict[str, object]) -> ProviderConfig:
    _reject_unknown(raw, _PROVIDER_KEYS, "provider")
    provider_id = _required_id(raw, "id", "provider")
    protocol = _required_string(raw, "protocol", "provider")
    if protocol not in {"openai-images", "gemini-native"}:
        raise ProviderConfigError("Unsupported provider protocol.")
    default_model = _required_string(raw, "default_model", "provider")
    targets = _parse_targets(raw.get("targets", {}))
    target_ids = [target.id for target in targets]
    if len(target_ids) != len(set(target_ids)):
        raise ProviderConfigError("Duplicate target id.")
    return ProviderConfig(
        id=provider_id,
        protocol=protocol,
        default_model=default_model,
        targets=tuple(targets),
        enabled=_optional_bool(raw, "enabled", True, "provider"),
    )


def _parse_targets(raw: object) -> list[ProviderTarget]:
    if isinstance(raw, dict):
        entries = [{"id": key, **_as_table(value, "target")} for key, value in raw.items()]
    elif isinstance(raw, list):
        entries = [_as_table(value, "target") for value in raw]
    else:
        raise ProviderConfigError("targets must be a table or array of tables.")
    return [_parse_target(entry) for entry in entries]


def _parse_target(raw: dict[str, object]) -> ProviderTarget:
    _reject_unknown(raw, _TARGET_KEYS, "target")
    target_id = _required_id(raw, "id", "target")
    base_url = _required_string(raw, "base_url", "target").rstrip("/")
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise ProviderConfigError("Provider target URL must be an absolute HTTP(S) URL without credentials.")
    api_key_env = _required_string(raw, "api_key_env", "target")
    if not _ENV_PATTERN.fullmatch(api_key_env):
        raise ProviderConfigError("api_key_env must be a valid environment variable name.")
    priority = raw.get("priority", 0)
    if not isinstance(priority, int) or isinstance(priority, bool) or priority < 0:
        raise ProviderConfigError("Target priority must be a non-negative integer.")
    return ProviderTarget(
        id=target_id,
        base_url=base_url,
        api_key_env=api_key_env,
        priority=priority,
        enabled=_optional_bool(raw, "enabled", True, "target"),
    )


def _as_table(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ProviderConfigError(f"{label} must be a table.")
    return value


def _reject_unknown(raw: dict[str, object], allowed: set[str], label: str) -> None:
    if set(raw) - allowed:
        raise ProviderConfigError(f"Unknown field in {label} configuration.")


def _required_string(raw: dict[str, object], key: str, label: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ProviderConfigError(f"{label}.{key} must be a non-empty string.")
    return value.strip()


def _optional_string(raw: dict[str, object], key: str, label: str) -> str | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ProviderConfigError(f"{label}.{key} must be a non-empty string.")
    return value.strip()


def _required_int(raw: dict[str, object], key: str, label: str) -> int:
    value = raw.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ProviderConfigError(f"{label}.{key} must be an integer.")
    return value


def _optional_bool(raw: dict[str, object], key: str, default: bool, label: str) -> bool:
    value = raw.get(key, default)
    if not isinstance(value, bool):
        raise ProviderConfigError(f"{label}.{key} must be a boolean.")
    return value


def _required_id(raw: dict[str, object], key: str, label: str) -> str:
    value = _required_string(raw, key, label)
    if not _ID_PATTERN.fullmatch(value):
        raise ProviderConfigError(f"{label}.{key} contains invalid characters.")
    return value
