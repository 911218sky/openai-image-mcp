from __future__ import annotations

import argparse

import pytest

from oepnai_image.cli import (
    build_size_from_ratio,
    parse_resolution,
    resolve_size_argument,
)
from oepnai_image.capabilities import (
    SizeMismatchError,
    build_size_metadata,
    compare_actual_size,
    enforce_size_match,
    resolve_requested_size,
    validate_requested_size,
)


def test_existing_resolution_parser_accepts_divisible_width_height() -> None:
    assert parse_resolution(" 1536 x 1024 ") == (1536, 1024)


def test_existing_resolution_parser_rejects_malformed_and_non_divisible_values() -> None:
    with pytest.raises(RuntimeError, match="WIDTHxHEIGHT"):
        parse_resolution("4k")
    with pytest.raises(RuntimeError, match="divisible by 16"):
        parse_resolution("4096x2335")


def test_existing_ratio_builder_rounds_each_dimension_to_sixteen() -> None:
    assert build_size_from_ratio("16:9", long_edge=3840, short_edge=None) == "3840x2160"


def test_existing_argument_resolution_keeps_conflicting_inputs_invalid() -> None:
    args = argparse.Namespace(
        size="1536x1024",
        resolution=None,
        aspect_ratio="16:9",
        width=None,
        height=None,
        long_edge=None,
        short_edge=None,
    )
    with pytest.raises(RuntimeError, match="either --size/--resolution"):
        resolve_size_argument(args)


def test_named_4k_preset_resolves_only_to_exact_uhd_16_by_9() -> None:
    assert resolve_requested_size("4k") == "3840x2160"
    assert resolve_requested_size("4k", aspect_ratio="16:9") == "3840x2160"
    assert resolve_requested_size("4K", aspect_ratio="16/9") == "3840x2160"


def test_named_4k_preset_rejects_ambiguous_ratio_and_aliases() -> None:
    with pytest.raises(RuntimeError, match="only available for 16:9"):
        resolve_requested_size("4k", aspect_ratio="4:3")
    with pytest.raises(RuntimeError, match="WIDTHxHEIGHT"):
        resolve_requested_size("uhd-4k")


def test_unknown_capabilities_do_not_reject_legacy_valid_size() -> None:
    assert validate_requested_size("1536x1024", supported_sizes=None) == "1536x1024"


def test_actual_size_comparison_records_a_matching_4k_result() -> None:
    result = compare_actual_size("3840x2160", (3840, 2160))
    assert result.actual_size == "3840x2160"
    assert result.size_match is True
    assert result.size_status == "match"


def test_actual_size_comparison_reports_mismatch_without_claiming_4k() -> None:
    result = compare_actual_size("3840x2160", (4096, 2335))
    assert result.actual_size == "4096x2335"
    assert result.size_match is False
    assert result.size_status == "mismatch"


def test_actual_size_comparison_reports_invalid_dimensions_and_stale_state_is_not_reused() -> None:
    invalid = compare_actual_size("3840x2160", (0, 2160))
    fresh = compare_actual_size("3840x2160", (3840, 2160))
    assert invalid.size_status == "invalid_actual"
    assert invalid.size_match is False
    assert fresh.size_match is True


def test_strict_helper_fails_only_after_actual_mismatch_is_inspected() -> None:
    with pytest.raises(SizeMismatchError, match="3840x2160.*4096x2335"):
        enforce_size_match("3840x2160", (4096, 2335))
    assert enforce_size_match("3840x2160", (3840, 2160)).size_match is True


def test_metadata_keeps_legacy_fields_and_adds_actual_fields() -> None:
    metadata = build_size_metadata(
        requested_size="3840x2160",
        actual_size=(3840, 2160),
        original_size="3840x2160",
        final_size="3840x2160",
        resized=False,
    )
    fields = metadata.as_manifest_fields()
    assert fields == {
        "requested_size": "3840x2160",
        "original_size": "3840x2160",
        "final_size": "3840x2160",
        "resized": False,
        "actual_size": "3840x2160",
        "size_match": True,
        "size_status": "match",
    }
