from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final, Literal, TypedDict


Size = tuple[int, int]
SizeStatus = Literal["match", "mismatch", "invalid_requested", "invalid_actual"]
UHD_4K_SIZE: Final[Size] = (3840, 2160)
SIZE_DIVISIBILITY: Final[int] = 16


class ManifestSizeFields(TypedDict):
    requested_size: str
    original_size: str
    final_size: str
    resized: bool
    actual_size: str
    size_match: bool
    size_status: SizeStatus


class SizeValidationError(RuntimeError):
    """Raised when a requested or actual size cannot be parsed safely."""


@dataclass(frozen=True, slots=True)
class ActualSizeComparison:
    """The requested-versus-decoded size result for one image."""

    requested_size: str
    actual_size: str
    size_match: bool
    size_status: SizeStatus


class SizeMismatchError(RuntimeError):
    """Raised by strict mode when decoded pixels do not match the request."""

    def __init__(self, comparison: ActualSizeComparison) -> None:
        self.comparison = comparison
        super().__init__(
            f"Requested size {comparison.requested_size} did not match "
            f"actual size {comparison.actual_size} ({comparison.size_status})."
        )


@dataclass(frozen=True, slots=True)
class ImageSizeMetadata:
    """Legacy image fields plus additive decoded-size inspection fields."""

    requested_size: str
    original_size: str
    final_size: str
    resized: bool
    actual_size: str
    size_match: bool
    size_status: SizeStatus

    def as_manifest_fields(self) -> ManifestSizeFields:
        """Return JSON-compatible fields without writing a manifest."""
        return {
            "requested_size": self.requested_size,
            "original_size": self.original_size,
            "final_size": self.final_size,
            "resized": self.resized,
            "actual_size": self.actual_size,
            "size_match": self.size_match,
            "size_status": self.size_status,
        }


def parse_aspect_ratio(value: str) -> Size:
    """Parse a positive WIDTH:HEIGHT or WIDTH/HEIGHT ratio."""
    match = re.fullmatch(r"\s*(\d+)\s*[:/]\s*(\d+)\s*", value)
    if not match:
        raise SizeValidationError(
            "Aspect ratio must look like WIDTH:HEIGHT, for example 16:9."
        )
    width_ratio = int(match.group(1))
    height_ratio = int(match.group(2))
    if width_ratio < 1 or height_ratio < 1:
        raise SizeValidationError("Aspect ratio values must be positive integers.")
    return width_ratio, height_ratio


def validate_size_dimensions(width: int, height: int) -> None:
    """Require positive dimensions divisible by the legacy size divisor."""
    if width < 1 or height < 1:
        raise SizeValidationError("Resolution width and height must be positive integers.")
    if width % SIZE_DIVISIBILITY != 0 or height % SIZE_DIVISIBILITY != 0:
        raise SizeValidationError(
            f"Invalid size '{width}x{height}'. Width and height must both be divisible by "
            f"{SIZE_DIVISIBILITY}."
        )


def parse_resolution(value: str) -> Size:
    """Parse a legacy WIDTHxHEIGHT resolution and preserve divisibility rules."""
    match = re.fullmatch(r"\s*(\d+)\s*x\s*(\d+)\s*", value, flags=re.IGNORECASE)
    if not match:
        raise SizeValidationError(
            "Resolution must look like WIDTHxHEIGHT, for example 1536x1024."
        )
    width = int(match.group(1))
    height = int(match.group(2))
    validate_size_dimensions(width, height)
    return width, height


def _round_dimension(value: float) -> int:
    return max(1, int(round(value)))


def _round_to_multiple(value: int) -> int:
    return max(SIZE_DIVISIBILITY, int(round(value / SIZE_DIVISIBILITY)) * SIZE_DIVISIBILITY)


def _build_size_from_ratio(
    aspect_ratio: str,
    *,
    long_edge: int | None,
    short_edge: int | None,
) -> Size:
    width_ratio, height_ratio = parse_aspect_ratio(aspect_ratio)
    if long_edge is not None and short_edge is not None:
        raise SizeValidationError(
            "Use either --long-edge or --short-edge with --aspect-ratio, not both."
        )
    if long_edge is None and short_edge is None:
        long_edge = 1536
    if width_ratio >= height_ratio:
        if long_edge is not None:
            width = long_edge
            height = _round_dimension(long_edge * height_ratio / width_ratio)
        else:
            height = short_edge or 1024
            width = _round_dimension(height * width_ratio / height_ratio)
    elif long_edge is not None:
        height = long_edge
        width = _round_dimension(long_edge * width_ratio / height_ratio)
    else:
        width = short_edge or 1024
        height = _round_dimension(width * height_ratio / width_ratio)
    result = _round_to_multiple(width), _round_to_multiple(height)
    validate_size_dimensions(*result)
    return result


def resolve_requested_size(
    size: str | None = None,
    *,
    resolution: str | None = None,
    aspect_ratio: str | None = None,
    width: int | None = None,
    height: int | None = None,
    long_edge: int | None = None,
    short_edge: int | None = None,
) -> str | None:
    """Resolve legacy size inputs and the single named 16:9 UHD 4K preset."""
    explicit_sizes = [value for value in (size, resolution) if value]
    if len(explicit_sizes) > 1 and size != resolution:
        raise SizeValidationError("--size and --resolution both set different values.")
    selected = explicit_sizes[0] if explicit_sizes else None
    is_4k = selected is not None and selected.strip().casefold() == "4k"
    if is_4k:
        if any(value is not None for value in (width, height, long_edge, short_edge)):
            raise SizeValidationError("The 4k preset cannot be combined with explicit dimensions.")
        if aspect_ratio is not None and parse_aspect_ratio(aspect_ratio) != (16, 9):
            raise SizeValidationError("The 4k preset is only available for 16:9.")
        return f"{UHD_4K_SIZE[0]}x{UHD_4K_SIZE[1]}"
    if selected is not None and aspect_ratio:
        raise SizeValidationError("Use either --size/--resolution or --aspect-ratio, not both.")
    if width is not None or height is not None:
        if selected or aspect_ratio:
            raise SizeValidationError(
                "--width/--height cannot be combined with --size, --resolution, or --aspect-ratio."
            )
        if width is None or height is None:
            raise SizeValidationError("Use --width and --height together.")
        validate_size_dimensions(width, height)
        return f"{width}x{height}"
    if selected is not None:
        parsed = parse_resolution(selected)
        return f"{parsed[0]}x{parsed[1]}"
    if aspect_ratio:
        if long_edge is not None and long_edge < 1:
            raise SizeValidationError("--long-edge must be at least 1.")
        if short_edge is not None and short_edge < 1:
            raise SizeValidationError("--short-edge must be at least 1.")
        parsed = _build_size_from_ratio(
            aspect_ratio,
            long_edge=long_edge,
            short_edge=short_edge,
        )
        return f"{parsed[0]}x{parsed[1]}"
    if long_edge is not None or short_edge is not None:
        raise SizeValidationError("--long-edge or --short-edge requires --aspect-ratio.")
    return None


def validate_requested_size(
    requested_size: str,
    *,
    supported_sizes: frozenset[str] | None = None,
) -> str:
    """Validate a size, treating unknown provider capabilities as advisory."""
    parsed = parse_resolution(requested_size)
    normalized = f"{parsed[0]}x{parsed[1]}"
    if supported_sizes is not None and normalized not in supported_sizes:
        raise SizeValidationError(f"Provider does not support requested size {normalized}.")
    return normalized


def _parse_actual_size(value: str | Size) -> Size:
    if isinstance(value, tuple):
        width, height = value
        if width < 1 or height < 1:
            raise SizeValidationError("Actual image dimensions must be positive integers.")
        return width, height
    match = re.fullmatch(r"\s*(\d+)\s*x\s*(\d+)\s*", value, flags=re.IGNORECASE)
    if not match:
        raise SizeValidationError("Actual image size must look like WIDTHxHEIGHT.")
    width = int(match.group(1))
    height = int(match.group(2))
    if width < 1 or height < 1:
        raise SizeValidationError("Actual image dimensions must be positive integers.")
    return width, height


def compare_actual_size(requested_size: str, actual_size: str | Size) -> ActualSizeComparison:
    """Compare requested dimensions with decoded dimensions without side effects."""
    try:
        requested = parse_resolution(requested_size)
    except SizeValidationError:
        requested_text = requested_size.strip()
        actual_text = _format_actual_size(actual_size)
        return ActualSizeComparison(requested_text, actual_text, False, "invalid_requested")
    try:
        actual = _parse_actual_size(actual_size)
    except SizeValidationError:
        return ActualSizeComparison(
            f"{requested[0]}x{requested[1]}",
            _format_actual_size(actual_size),
            False,
            "invalid_actual",
        )
    requested_text = f"{requested[0]}x{requested[1]}"
    actual_text = f"{actual[0]}x{actual[1]}"
    matches = requested == actual
    return ActualSizeComparison(
        requested_text,
        actual_text,
        matches,
        "match" if matches else "mismatch",
    )


def _format_actual_size(value: str | Size) -> str:
    if isinstance(value, tuple):
        return f"{value[0]}x{value[1]}"
    return value.strip()


def enforce_size_match(requested_size: str, actual_size: str | Size) -> ActualSizeComparison:
    """Return inspected dimensions or fail in opt-in strict mode."""
    comparison = compare_actual_size(requested_size, actual_size)
    if not comparison.size_match:
        raise SizeMismatchError(comparison)
    return comparison


def build_size_metadata(
    *,
    requested_size: str,
    actual_size: str | Size,
    original_size: str,
    final_size: str,
    resized: bool,
) -> ImageSizeMetadata:
    """Build additive metadata from actual pixels while preserving legacy fields."""
    comparison = compare_actual_size(requested_size, actual_size)
    return ImageSizeMetadata(
        requested_size=requested_size,
        original_size=original_size,
        final_size=final_size,
        resized=resized,
        actual_size=comparison.actual_size,
        size_match=comparison.size_match,
        size_status=comparison.size_status,
    )
