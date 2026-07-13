from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum, unique
import json
from typing import Protocol, TypeAlias


JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


@unique
class ProviderProtocol(StrEnum):
    OPENAI_IMAGES = "openai-images"
    OPENAI_CHAT_IMAGES = "openai-chat-images"
    GEMINI_NATIVE = "gemini-native"


@unique
class ResponseErrorKind(StrEnum):
    HTTP_ERROR = "http-error"
    NON_JSON = "non-json"
    UNSUPPORTED_RESPONSE = "unsupported-response"


@dataclass(frozen=True, slots=True)
class ProviderIdentity:
    provider_id: str
    target_id: str


@dataclass(frozen=True, slots=True)
class BearerAuth:
    header_name: str = "Authorization"
    scheme: str = "Bearer"


@dataclass(frozen=True, slots=True)
class HeaderAuth:
    header_name: str


type HttpAuth = BearerAuth | HeaderAuth


@dataclass(frozen=True, slots=True)
class ProviderHttpConfig:
    identity: ProviderIdentity
    protocol: ProviderProtocol
    base_url: str
    path: str
    auth: HttpAuth
    timeout_seconds: float


@dataclass(frozen=True, slots=True)
class HttpRequest:
    method: str
    url: str
    headers: Mapping[str, str]
    json_body: Mapping[str, JsonValue]
    timeout_seconds: float


@dataclass(frozen=True, slots=True)
class HttpResponse:
    status_code: int
    body: bytes


class HttpTransport(Protocol):
    def send(self, request: HttpRequest) -> HttpResponse: ...


@dataclass(frozen=True, slots=True)
class ProviderResponseError(RuntimeError):
    identity: ProviderIdentity
    protocol: ProviderProtocol
    status_code: int
    kind: ResponseErrorKind
    detail: str

    def __str__(self) -> str:
        return self.detail


@dataclass(frozen=True, slots=True)
class NormalizedImage:
    b64_json: str | None
    url: str | None
    mime_type: str | None
    revised_prompt: str | None = None


@dataclass(frozen=True, slots=True)
class ResponseMetadata:
    created: int | None = None
    request_id: str | None = None
    model: str | None = None


@dataclass(frozen=True, slots=True)
class NormalizedProviderResponse:
    data: tuple[NormalizedImage, ...]
    status: int
    metadata: ResponseMetadata
    provider_id: str
    target_id: str


class SdkTransport(Protocol):
    def generate(self, payload: Mapping[str, JsonValue]) -> object: ...


class OpenAICompatibleProvider:
    def __init__(
        self,
        config: ProviderHttpConfig,
        transport: HttpTransport,
        credential: str,
    ) -> None:
        self.config = config
        self.transport = transport
        self.credential = credential

    def generate(self, payload: Mapping[str, JsonValue]) -> NormalizedProviderResponse:
        response = self.transport.send(
            HttpRequest(
                method="POST",
                url=f"{self.config.base_url.rstrip('/')}/{self.config.path.lstrip('/')}",
                headers=_auth_headers(self.config.auth, self.credential),
                json_body=payload,
                timeout_seconds=self.config.timeout_seconds,
            )
        )
        body = _decode_json(response, self.config.identity, self.config.protocol)
        if response.status_code >= 400 or "error" in body:
            raise _http_error(response.status_code, self.config, ResponseErrorKind.HTTP_ERROR)
        data = _openai_images(body, self.config)
        return NormalizedProviderResponse(
            data=data,
            status=response.status_code,
            metadata=ResponseMetadata(
                created=_optional_int(body.get("created")),
                model=_optional_text(body.get("model")),
            ),
            provider_id=self.config.identity.provider_id,
            target_id=self.config.identity.target_id,
        )


class GeminiNativeProvider(OpenAICompatibleProvider):
    def generate(self, payload: Mapping[str, JsonValue]) -> NormalizedProviderResponse:
        response = self.transport.send(
            HttpRequest(
                method="POST",
                url=f"{self.config.base_url.rstrip('/')}/{self.config.path.lstrip('/')}",
                headers=_auth_headers(self.config.auth, self.credential),
                json_body=payload,
                timeout_seconds=self.config.timeout_seconds,
            )
        )
        body = _decode_json(response, self.config.identity, self.config.protocol)
        if response.status_code >= 400 or "error" in body:
            raise _http_error(response.status_code, self.config, ResponseErrorKind.HTTP_ERROR)
        images: list[NormalizedImage] = []
        candidates = body.get("candidates")
        if isinstance(candidates, list):
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    continue
                content = candidate.get("content")
                if not isinstance(content, dict):
                    continue
                parts = content.get("parts")
                if not isinstance(parts, list):
                    continue
                for part in parts:
                    if not isinstance(part, dict):
                        continue
                    inline = part.get("inlineData")
                    if not isinstance(inline, dict):
                        continue
                    encoded = inline.get("data")
                    mime_type = inline.get("mimeType")
                    if isinstance(encoded, str):
                        images.append(
                            NormalizedImage(
                                b64_json=encoded,
                                url=None,
                                mime_type=mime_type if isinstance(mime_type, str) else None,
                            )
                        )
        if not images:
            raise _http_error(200, self.config, ResponseErrorKind.UNSUPPORTED_RESPONSE)
        return NormalizedProviderResponse(
            data=tuple(images),
            status=response.status_code,
            metadata=ResponseMetadata(
                request_id=_optional_text(body.get("responseId")),
                model=_optional_text(body.get("modelVersion")),
            ),
            provider_id=self.config.identity.provider_id,
            target_id=self.config.identity.target_id,
        )


class OpenAISdkProvider:
    def __init__(self, identity: ProviderIdentity, transport: SdkTransport) -> None:
        self.identity = identity
        self.transport = transport

    def generate(self, payload: Mapping[str, JsonValue]) -> NormalizedProviderResponse:
        raw = self.transport.generate(payload)
        data_value = _attribute_or_key(raw, "data")
        if not isinstance(data_value, list):
            raise ProviderResponseError(
                self.identity,
                ProviderProtocol.OPENAI_IMAGES,
                200,
                ResponseErrorKind.UNSUPPORTED_RESPONSE,
                "Provider response did not contain image data.",
            )
        images = tuple(_normalize_openai_item(item, self.identity) for item in data_value)
        if not images:
            raise ProviderResponseError(
                self.identity,
                ProviderProtocol.OPENAI_IMAGES,
                200,
                ResponseErrorKind.UNSUPPORTED_RESPONSE,
                "Provider response did not contain image data.",
            )
        return NormalizedProviderResponse(
            data=images,
            status=_optional_int(_attribute_or_key(raw, "status")) or 200,
            metadata=ResponseMetadata(
                created=_optional_int(_attribute_or_key(raw, "created")),
                request_id=_optional_text(_attribute_or_key(raw, "request_id")),
                model=_optional_text(_attribute_or_key(raw, "model")),
            ),
            provider_id=self.identity.provider_id,
            target_id=self.identity.target_id,
        )


def _auth_headers(auth: HttpAuth, credential: str) -> dict[str, str]:
    if isinstance(auth, BearerAuth):
        return {auth.header_name: f"{auth.scheme} {credential}", "Content-Type": "application/json"}
    return {auth.header_name: credential, "Content-Type": "application/json"}


def _decode_json(
    response: HttpResponse,
    identity: ProviderIdentity,
    protocol: ProviderProtocol,
) -> dict[str, JsonValue]:
    try:
        value = json.loads(response.body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProviderResponseError(
            identity,
            protocol,
            response.status_code,
            ResponseErrorKind.NON_JSON,
            "Provider response was not valid JSON.",
        ) from exc
    if not isinstance(value, dict):
        raise ProviderResponseError(
            identity,
            protocol,
            response.status_code,
            ResponseErrorKind.UNSUPPORTED_RESPONSE,
            "Provider response had an unsupported JSON shape.",
        )
    return value


def _http_error(
    status_code: int,
    config: ProviderHttpConfig,
    kind: ResponseErrorKind,
) -> ProviderResponseError:
    return ProviderResponseError(
        config.identity,
        config.protocol,
        status_code,
        kind,
        f"Provider request failed with HTTP status {status_code}.",
    )


def _openai_images(
    body: dict[str, JsonValue],
    config: ProviderHttpConfig,
) -> tuple[NormalizedImage, ...]:
    raw_data = body.get("data")
    if not isinstance(raw_data, list):
        raise _http_error(200, config, ResponseErrorKind.UNSUPPORTED_RESPONSE)
    images = tuple(_normalize_openai_item(item, config.identity) for item in raw_data)
    if not images:
        raise _http_error(200, config, ResponseErrorKind.UNSUPPORTED_RESPONSE)
    return images


def _normalize_openai_item(
    item: JsonValue | object,
    identity: ProviderIdentity,
) -> NormalizedImage:
    b64_json = _attribute_or_key(item, "b64_json")
    url = _attribute_or_key(item, "url")
    mime_type = _attribute_or_key(item, "mime_type")
    revised_prompt = _attribute_or_key(item, "revised_prompt")
    if not isinstance(b64_json, str):
        b64_json = None
    if not isinstance(url, str):
        url = None
    if not isinstance(mime_type, str):
        mime_type = None
    if not isinstance(revised_prompt, str):
        revised_prompt = None
    if b64_json is None and url is None:
        raise ProviderResponseError(
            identity,
            ProviderProtocol.OPENAI_IMAGES,
            200,
            ResponseErrorKind.UNSUPPORTED_RESPONSE,
            "Provider image item had no supported image data.",
        )
    return NormalizedImage(b64_json, url, mime_type, revised_prompt)


def _attribute_or_key(value: object, key: str) -> object:
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _optional_text(value: object) -> str | None:
    return value if isinstance(value, str) else None
