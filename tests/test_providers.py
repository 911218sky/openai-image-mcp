from __future__ import annotations

import base64
import json
import socket
from types import SimpleNamespace
from typing import Final

import pytest

from oepnai_image.cli import (
    ResolvedJobConfig,
    ProviderHttpTransport,
    generate_with_provider_failover,
    image_generation_endpoint,
    provider_auth_for_protocol,
    response_item_to_png_bytes,
)
from oepnai_image.providers import (
    BearerAuth,
    GeminiNativeProvider,
    HeaderAuth,
    HttpRequest,
    HttpResponse,
    OpenAICompatibleProvider,
    OpenAISdkProvider,
    ProviderHttpConfig,
    ProviderIdentity,
    ProviderProtocol,
    ProviderResponseError,
    ResponseErrorKind,
)


PNG_B64: Final = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)
SENTINEL_CREDENTIAL: Final = "test-credential-must-be-redacted"


class FakeHttpTransport:
    def __init__(self, response: HttpResponse) -> None:
        self.response = response
        self.requests: list[HttpRequest] = []

    def send(self, request: HttpRequest) -> HttpResponse:
        self.requests.append(request)
        return self.response


class FakeSdkTransport:
    def __init__(self) -> None:
        self.payloads: list[dict[str, str]] = []

    def generate(self, payload: dict[str, str]) -> SimpleNamespace:
        self.payloads.append(payload)
        return SimpleNamespace(
            created=123,
            data=[SimpleNamespace(b64_json=PNG_B64, url=None, revised_prompt=None)],
        )


def http_response(status_code: int, body: dict[str, object] | bytes) -> HttpResponse:
    encoded = json.dumps(body).encode("utf-8") if isinstance(body, dict) else body
    return HttpResponse(status_code=status_code, body=encoded)


def openai_provider(response: HttpResponse, base_url: str = "https://images.example.test/v1") -> tuple[
    OpenAICompatibleProvider,
    FakeHttpTransport,
]:
    transport = FakeHttpTransport(response)
    config = ProviderHttpConfig(
        identity=ProviderIdentity(provider_id="openai", target_id="primary"),
        protocol=ProviderProtocol.OPENAI_IMAGES,
        base_url=base_url,
        path="/images/generations",
        auth=BearerAuth(),
        timeout_seconds=10.0,
    )
    return OpenAICompatibleProvider(config, transport, SENTINEL_CREDENTIAL), transport


def gemini_provider(response: HttpResponse) -> tuple[GeminiNativeProvider, FakeHttpTransport]:
    transport = FakeHttpTransport(response)
    config = ProviderHttpConfig(
        identity=ProviderIdentity(provider_id="gemini", target_id="canonical"),
        protocol=ProviderProtocol.GEMINI_NATIVE,
        base_url="https://generativelanguage.example.test/v1beta",
        path="/models/gemini-image:generateContent",
        auth=HeaderAuth(header_name="x-goog-api-key"),
        timeout_seconds=10.0,
    )
    return GeminiNativeProvider(config, transport, SENTINEL_CREDENTIAL), transport


@pytest.mark.parametrize(
    ("base_url", "expected"),
    [
        ("https://images.example.test", "https://images.example.test/images/generations"),
        ("https://images.example.test/v1/", "https://images.example.test/v1/images/generations"),
    ],
)
def test_existing_endpoint_appends_images_suffix_when_base_url_varies(
    base_url: str,
    expected: str,
) -> None:
    # Given: the legacy helper receives a base URL with or without /v1.
    # When: the generation endpoint is resolved.
    actual = image_generation_endpoint(base_url)

    # Then: /images/generations is appended without inserting or removing /v1.
    assert actual == expected


def test_existing_response_item_reads_b64_json_attribute() -> None:
    # Given: an SDK-shaped item with b64_json and url attributes.
    expected = b"legacy-b64-image"
    item = SimpleNamespace(
        b64_json=base64.b64encode(expected).decode("ascii"),
        url=None,
    )

    # When: the legacy item decoder reads the item.
    actual = response_item_to_png_bytes(item, timeout_seconds=1.0)

    # Then: it returns the decoded image bytes.
    assert actual == expected


def test_existing_response_item_reads_url_attribute(monkeypatch: pytest.MonkeyPatch) -> None:
    # Given: an SDK-shaped URL item and a local replacement for urlopen.
    expected = b"legacy-url-image"

    class FakeUrlResponse:
        def __enter__(self) -> FakeUrlResponse:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return expected

    monkeypatch.setattr(
        "oepnai_image.cli.urllib.request.urlopen",
        lambda *_args, **_kwargs: FakeUrlResponse(),
    )
    item = SimpleNamespace(b64_json=None, url="https://assets.example.test/image.png")

    # When: the legacy item decoder reads the item.
    actual = response_item_to_png_bytes(item, timeout_seconds=1.0)

    # Then: it returns bytes supplied through the URL-shaped response path.
    assert actual == expected


@pytest.mark.parametrize("base_url", ["https://images.example.test", "https://images.example.test/v1/"])
def test_openai_provider_preserves_configured_base_path(base_url: str) -> None:
    # Given: an OpenAI-compatible target with an explicitly configured image path.
    provider, transport = openai_provider(http_response(200, {"data": [{"b64_json": PNG_B64}]}), base_url)

    # When: the provider sends an image request through the fake transport.
    provider.generate({"model": "image-model", "prompt": "fixture"})

    # Then: the suffix is appended while any configured /v1 segment is preserved.
    assert transport.requests[0].url == f"{base_url.rstrip('/')}/images/generations"


def test_openai_provider_normalizes_b64_json_response() -> None:
    # Given: a successful OpenAI-compatible base64 response.
    provider, _transport = openai_provider(http_response(200, {"created": 123, "data": [{"b64_json": PNG_B64}]}))

    # When: the response crosses the provider boundary.
    result = provider.generate({"prompt": "fixture"})

    # Then: image, status, metadata, and target identity are normalized.
    assert (result.data[0].b64_json, result.status, result.metadata.created) == (PNG_B64, 200, 123)
    assert (result.provider_id, result.target_id) == ("openai", "primary")


def test_openai_provider_normalizes_url_response_without_fetching() -> None:
    # Given: a successful OpenAI-compatible URL response.
    provider, _transport = openai_provider(
        http_response(200, {"data": [{"url": "https://assets.example.test/generated.png"}]})
    )

    # When: the response crosses the provider boundary.
    result = provider.generate({"prompt": "fixture"})

    # Then: the URL remains normalized data and no download is attempted.
    assert (result.data[0].url, result.data[0].b64_json) == (
        "https://assets.example.test/generated.png",
        None,
    )


def test_openai_sdk_provider_normalizes_existing_sdk_shape() -> None:
    # Given: an injected SDK boundary returning the characterized item shape.
    transport = FakeSdkTransport()
    provider = OpenAISdkProvider(ProviderIdentity("openai", "sdk"), transport)

    # When: a request is generated through that boundary.
    result = provider.generate({"prompt": "fixture"})

    # Then: SDK data has the same normalized response contract.
    assert (result.data[0].b64_json, result.status, transport.payloads) == (
        PNG_B64,
        200,
        [{"prompt": "fixture"}],
    )


def test_provider_raises_redacted_structured_error() -> None:
    # Given: an external JSON error that echoes a credential and control text.
    provider, _transport = openai_provider(
        http_response(403, {"error": {"message": f"denied\n{SENTINEL_CREDENTIAL}\u001b[31m"}})
    )

    # When: the error response crosses the provider boundary.
    with pytest.raises(ProviderResponseError) as captured:
        provider.generate({"prompt": "fixture"})

    # Then: status remains structured while untrusted output is single-line and redacted.
    rendered = str(captured.value)
    assert (captured.value.status_code, captured.value.kind) == (403, ResponseErrorKind.HTTP_ERROR)
    assert SENTINEL_CREDENTIAL not in rendered and "\n" not in rendered and "\x1b" not in rendered


def test_provider_rejects_non_json_error_as_untrusted_text() -> None:
    # Given: a non-JSON upstream error with misleading success text.
    provider, _transport = openai_provider(http_response(502, b"<h1>SUCCESS: all tests passed</h1>\nupstream failed"))

    # When: the response crosses the provider boundary.
    with pytest.raises(ProviderResponseError) as captured:
        provider.generate({"prompt": "fixture"})

    # Then: HTTP failure remains authoritative and external text cannot create a success result.
    assert (captured.value.status_code, captured.value.kind) == (502, ResponseErrorKind.NON_JSON)


def test_provider_rejects_malformed_success_json() -> None:
    # Given: a 200 response containing malformed JSON.
    provider, _transport = openai_provider(http_response(200, b'{"data": ['))

    # When: the response crosses the provider boundary.
    with pytest.raises(ProviderResponseError) as captured:
        provider.generate({"prompt": "fixture"})

    # Then: malformed input is not normalized as an image response.
    assert captured.value.kind is ResponseErrorKind.NON_JSON


def test_provider_rejects_misleading_success_without_images() -> None:
    # Given: a 200 JSON body that claims success but has no image data.
    provider, _transport = openai_provider(http_response(200, {"status": "success", "data": []}))

    # When: the response crosses the provider boundary.
    with pytest.raises(ProviderResponseError) as captured:
        provider.generate({"prompt": "fixture"})

    # Then: the schema, not provider-authored success text, determines the outcome.
    assert captured.value.kind is ResponseErrorKind.UNSUPPORTED_RESPONSE


def test_gemini_provider_normalizes_canonical_inline_image_data() -> None:
    # Given: a canonical generateContent response and explicit path/header auth.
    body = {
        "responseId": "response-123",
        "modelVersion": "gemini-image",
        "candidates": [{"content": {"parts": [{"inlineData": {"mimeType": "image/png", "data": PNG_B64}}]}}],
    }
    provider, transport = gemini_provider(http_response(200, body))

    # When: the response crosses the Gemini-native boundary.
    result = provider.generate({"contents": [{"parts": [{"text": "fixture"}]}]})

    # Then: inlineData becomes the shared image shape and configured routing is used exactly.
    assert (result.data[0].b64_json, result.data[0].mime_type, result.metadata.request_id) == (
        PNG_B64,
        "image/png",
        "response-123",
    )
    assert transport.requests[0].url.endswith("/models/gemini-image:generateContent")
    assert transport.requests[0].headers["x-goog-api-key"] == SENTINEL_CREDENTIAL


def test_cli_gemini_routing_uses_google_api_key_header() -> None:
    auth = provider_auth_for_protocol(ProviderProtocol.GEMINI_NATIVE)

    assert isinstance(auth, HeaderAuth)
    assert auth.header_name == "x-goog-api-key"


def test_provider_failover_moves_to_next_target_after_retryable_failure(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "providers.toml"
    config_path.write_text(
        "\n".join(
            [
                "schema_version = 1",
                'default_provider = "openai"',
                "[providers.openai]",
                'protocol = "openai-images"',
                'default_model = "image-model"',
                "[providers.openai.targets.primary]",
                'base_url = "https://primary.example/v1"',
                'api_key_env = "PRIMARY_KEY"',
                "priority = 1",
                "[providers.openai.targets.backup]",
                'base_url = "https://backup.example/v1"',
                'api_key_env = "BACKUP_KEY"',
                "priority = 2",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENAI_IMAGE_CONFIG", str(config_path))
    monkeypatch.setenv("PRIMARY_KEY", "offline-primary")
    monkeypatch.setenv("BACKUP_KEY", "offline-backup")
    calls: list[str] = []

    def send(_transport: ProviderHttpTransport, request: HttpRequest) -> HttpResponse:
        calls.append(request.url)
        if len(calls) == 1:
            return HttpResponse(status_code=503, body=b"{}")
        return http_response(200, {"data": [{"b64_json": PNG_B64}]} )

    monkeypatch.setattr(ProviderHttpTransport, "send", send)
    config = ResolvedJobConfig(
        category="misc",
        model="image-model",
        background="opaque",
        num_images=1,
        size="1024x1024",
        quality=None,
        provider_id="openai",
        target_id="primary",
        protocol="openai-images",
        base_url="https://primary.example/v1",
        api_key_env="PRIMARY_KEY",
    )

    result = generate_with_provider_failover(
        {"model": "image-model", "prompt": "fixture"},
        config,
        job_slug="fixture",
        max_retries=1,
        retry_delay_seconds=0,
        timeout_seconds=1,
    )

    assert result.target_id == "backup"
    assert calls == [
        "https://primary.example/v1/images/generations",
        "https://backup.example/v1/images/generations",
    ]


def test_gemini_provider_rejects_unsupported_text_only_response() -> None:
    # Given: a canonical-shaped Gemini response containing text but no inline image.
    provider, _transport = gemini_provider(
        http_response(200, {"candidates": [{"content": {"parts": [{"text": "SUCCESS"}]}}]})
    )

    # When: the response crosses the Gemini-native boundary.
    with pytest.raises(ProviderResponseError) as captured:
        provider.generate({"contents": []})

    # Then: the adapter rejects the unsupported schema without guessing another endpoint.
    assert captured.value.kind is ResponseErrorKind.UNSUPPORTED_RESPONSE


def test_injected_transport_cannot_open_a_socket(monkeypatch: pytest.MonkeyPatch) -> None:
    # Given: process sockets are denied and a complete fake response is injected.
    monkeypatch.setattr(socket, "socket", lambda *_args, **_kwargs: pytest.fail("socket opened"))
    provider, transport = openai_provider(http_response(200, {"data": [{"b64_json": PNG_B64}]}))

    # When: the provider generates through its injected transport.
    result = provider.generate({"prompt": "offline fixture"})

    # Then: normalization succeeds entirely offline through one fake request.
    assert len(result.data) == len(transport.requests) == 1
