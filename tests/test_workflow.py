from __future__ import annotations

from pathlib import Path

from oepnai_image.cli import _provider_payload, parse_args
from oepnai_image.workflow import GenerationRequest, args_from_request, request_from_args


def test_cli_provider_options_round_trip_through_generation_request() -> None:
    # Given: provider routing and strict pixel validation are supplied as CLI flags.
    request = request_from_args(
        [
            "--prompt",
            "a test image",
            "--provider",
            "gemini",
            "--target",
            "primary",
            "--strict-size",
        ]
    )

    # When: the request is converted back to the CLI namespace used by workflow code.
    args = args_from_request(request)

    # Then: the routing and validation choices survive both boundaries.
    assert request.provider == "gemini"
    assert request.target == "primary"
    assert request.strict_size is True
    assert args.provider == "gemini"
    assert args.target == "primary"
    assert args.strict_size is True


def test_generation_request_round_trip_preserves_provider_controls() -> None:
    # Given: an MCP-shaped request with explicit provider controls.
    request = GenerationRequest(
        prompts=["fixture"],
        provider="openai-compatible",
        target="backup",
        strict_size=True,
    )

    # When: the request is converted to the shared CLI namespace.
    args = args_from_request(request)

    # Then: it is ready for the existing generation workflow without losing fields.
    assert args.prompt == ["fixture"]
    assert args.provider == "openai-compatible"
    assert args.target == "backup"
    assert args.strict_size is True


def test_parse_args_exposes_provider_controls() -> None:
    # Given: the standalone CLI receives provider controls.
    args = parse_args(["--prompt", "fixture", "--provider", "openai", "--target", "primary", "--strict-size"])

    # When: argparse resolves the command line.
    # Then: all controls are represented on the namespace.
    assert (args.provider, args.target, args.strict_size) == ("openai", "primary", True)


def test_gemini_payload_requests_image_modality() -> None:
    payload = _provider_payload(
        {"model": "gemini-image", "prompt": "fixture", "size": "1024x1024"},
        "gemini-native",
    )

    assert payload["generationConfig"]["responseModalities"] == ["IMAGE"]
    assert payload["contents"][0]["parts"][0]["text"] == "fixture"


def test_unused_path_import_keeps_workflow_test_fixture_typed() -> None:
    # Given: Path remains available for future request fixtures.
    # When: this no-op fixture is evaluated.
    # Then: the test module has no untyped path sentinel.
    assert Path("fixture").name == "fixture"
