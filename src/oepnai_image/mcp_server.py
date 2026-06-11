from __future__ import annotations

import base64
import os
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.types import CallToolResult, ImageContent, TextContent

from .cli import format_style_listing, load_styles
from .workflow import GenerationRequest, run_generation_request


mcp = FastMCP("openai-image-mcp", json_response=True)
OUTPUT_DIR_ENV = "OPENAI_IMAGE_OUTPUT_DIR"
PUBLIC_IMAGES_ROOT_ENV = "OPENAI_IMAGE_PUBLIC_IMAGES_ROOT"
PUBLIC_URL_PREFIX_ENV = "OPENAI_IMAGE_PUBLIC_URL_PREFIX"
DEFAULT_PUBLIC_IMAGES_ROOT = Path("/app/client/public/images")
DEFAULT_PUBLIC_OUTPUT_SUBDIR = "openai-image-mcp"
DEFAULT_PUBLIC_URL_PREFIX = "/images"
EMBED_MAX_BYTES_ENV = "OPENAI_IMAGE_EMBED_MAX_BYTES"
DEFAULT_EMBED_MAX_BYTES = 8 * 1024 * 1024


def _path(value: str | None) -> Path | None:
    return Path(value).expanduser() if value else None


def _positive_int(value: int | None, *, name: str) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{name} must be a positive integer.") from exc
    if parsed < 1:
        raise RuntimeError(f"{name} must be at least 1.")
    return parsed


def _resolve_num_images(
    num_images: int = 1,
    *,
    n: int | None = None,
    count: int | None = None,
    image_count: int | None = None,
) -> int:
    """Resolve common image-count aliases used by LLMs and image APIs."""
    for name, value in (("n", n), ("count", count), ("image_count", image_count)):
        parsed = _positive_int(value, name=name)
        if parsed is not None:
            return parsed
    return _positive_int(num_images, name="num_images") or 1


def _public_images_root() -> Path:
    return Path(os.getenv(PUBLIC_IMAGES_ROOT_ENV, str(DEFAULT_PUBLIC_IMAGES_ROOT))).expanduser()


def _public_url_prefix() -> str:
    prefix = os.getenv(PUBLIC_URL_PREFIX_ENV, DEFAULT_PUBLIC_URL_PREFIX).strip() or DEFAULT_PUBLIC_URL_PREFIX
    if not prefix.startswith("/"):
        prefix = f"/{prefix}"
    return prefix.rstrip("/")


def _default_output_dir(output_dir: str | None) -> Path | None:
    explicit = _path(output_dir) or _path(os.getenv(OUTPUT_DIR_ENV))
    if explicit:
        return explicit

    public_root = _public_images_root()
    if public_root.exists():
        return public_root / DEFAULT_PUBLIC_OUTPUT_SUBDIR
    return None


def _public_url_for_file(file_path: str) -> str | None:
    path = Path(file_path)
    if not path.is_absolute():
        path = Path.cwd() / path

    try:
        relative = path.resolve().relative_to(_public_images_root().resolve())
    except ValueError:
        return None

    return f"{_public_url_prefix()}/{relative.as_posix()}"


def _embed_max_bytes() -> int:
    raw = os.getenv(EMBED_MAX_BYTES_ENV)
    if not raw:
        return DEFAULT_EMBED_MAX_BYTES
    try:
        parsed = int(raw)
    except ValueError:
        return DEFAULT_EMBED_MAX_BYTES
    return parsed if parsed > 0 else DEFAULT_EMBED_MAX_BYTES


def _title_from_image(image: dict[str, Any], job: dict[str, Any]) -> str:
    file_stem = Path(str(image.get("file") or "")).stem
    if file_stem and not file_stem.startswith("prompt-"):
        title = re.sub(r"[-_]+", " ", file_stem).strip()
    else:
        title = str(job.get("base_prompt") or job.get("slug") or "Generated Image").strip()

    title = re.sub(r"\s+", " ", title)
    if re.fullmatch(r"[a-z0-9][a-z0-9 -]*", title):
        title = title.title()
    return title[:80].rstrip() or "Generated Image"


def _enrich_for_librechat(manifest: dict[str, Any]) -> dict[str, Any]:
    markdown_blocks: list[str] = []

    for job in manifest.get("jobs", []):
        if not isinstance(job, dict):
            continue
        for image in job.get("images", []):
            if not isinstance(image, dict):
                continue

            title = _title_from_image(image, job)
            image["title"] = title
            url = _public_url_for_file(str(image.get("file") or ""))
            if not url:
                continue

            image["url"] = url
            markdown_blocks.append(f"### {title}\n\n![{title}]({url})")

    if markdown_blocks:
        markdown = "\n\n".join(markdown_blocks)
        manifest["text"] = markdown
        manifest["display_markdown"] = markdown
    return manifest


def _image_content_for_file(file_path: str) -> ImageContent | None:
    path = Path(file_path)
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.is_file():
        return None
    if path.stat().st_size > _embed_max_bytes():
        return None

    return ImageContent(
        type="image",
        data=base64.b64encode(path.read_bytes()).decode("ascii"),
        mimeType="image/png",
    )


def _to_tool_result(manifest: dict[str, Any]) -> CallToolResult:
    content: list[TextContent | ImageContent] = []
    text = str(manifest.get("display_markdown") or manifest.get("text") or "").strip()
    if text:
        content.append(TextContent(type="text", text=text))

    for job in manifest.get("jobs", []):
        if not isinstance(job, dict):
            continue
        for image in job.get("images", []):
            if not isinstance(image, dict):
                continue
            image_content = _image_content_for_file(str(image.get("file") or ""))
            if image_content:
                content.append(image_content)

    if not content:
        content.append(TextContent(type="text", text="(No images generated)"))

    return CallToolResult(content=content, structuredContent=manifest)


def _run_request(request: GenerationRequest) -> dict[str, Any]:
    # MCP stdio reserves stdout for JSON-RPC frames. The CLI workflow prints
    # human progress lines, so route them to stderr when called as a tool.
    with redirect_stdout(sys.stderr):
        return _enrich_for_librechat(run_generation_request(request))


@mcp.tool()
def list_prompt_styles(style_dir: str | None = None) -> dict[str, Any]:
    """List bundled and optional custom prompt styles."""
    styles = load_styles(_path(style_dir))
    return {
        "styles": [
            {
                "slug": slug,
                "name": style.name,
                "description": style.description,
                "defaults": style.defaults,
            }
            for slug, style in sorted(styles.items())
        ],
        "text": format_style_listing(styles),
    }


@mcp.tool()
def plan_image_generation(
    prompt: str,
    style: str | None = None,
    size: str | None = None,
    aspect_ratio: str | None = None,
    long_edge: int | None = None,
    short_edge: int | None = None,
    model: str | None = None,
    quality: str | None = None,
    background: str | None = None,
    category: str = "misc",
    filename_prefix: str | None = None,
    style_dir: str | None = None,
) -> CallToolResult:
    """Resolve the payload and manifest shape for one prompt without calling the image API."""
    return _to_tool_result(
        _run_request(
            GenerationRequest(
                prompts=[prompt],
                category=category,
                filename_prefix=filename_prefix,
                model=model,
                style=style,
                style_dir=_path(style_dir),
                dry_run=True,
                size=size,
                aspect_ratio=aspect_ratio,
                long_edge=long_edge,
                short_edge=short_edge,
                quality=quality,
                background=background,
            )
        )
    )


@mcp.tool()
def generate_image(
    prompt: str,
    output_dir: str | None = None,
    style: str | None = None,
    size: str | None = None,
    aspect_ratio: str | None = None,
    long_edge: int | None = None,
    short_edge: int | None = None,
    model: str | None = None,
    quality: str | None = None,
    background: str | None = None,
    num_images: int = 1,
    n: int | None = None,
    count: int | None = None,
    image_count: int | None = None,
    category: str = "misc",
    filename_prefix: str | None = None,
    flat_output: bool = False,
    timeout: float | None = None,
    max_retries: int = 5,
    retry_delay: float = 1.0,
    style_dir: str | None = None,
) -> CallToolResult:
    """Generate one or more image files from a single prompt.

    Use num_images to generate multiple variations in one tool call from the
    same prompt. The aliases n, count, and image_count are also accepted and
    override num_images when provided.
    """
    return _to_tool_result(
        _run_request(
            GenerationRequest(
                prompts=[prompt],
                output_dir=_default_output_dir(output_dir),
                category=category,
                filename_prefix=filename_prefix,
                model=model,
                style=style,
                style_dir=_path(style_dir),
                num_images=_resolve_num_images(
                    num_images,
                    n=n,
                    count=count,
                    image_count=image_count,
                ),
                max_retries=max_retries,
                retry_delay=retry_delay,
                timeout=timeout,
                size=size,
                aspect_ratio=aspect_ratio,
                long_edge=long_edge,
                short_edge=short_edge,
                quality=quality,
                background=background,
                flat_output=flat_output,
            )
        )
    )


@mcp.tool()
def generate_images_batch(
    batch_path: str,
    output_dir: str | None = None,
    only: list[str] | None = None,
    limit: int | None = None,
    style: str | None = None,
    size: str | None = None,
    model: str | None = None,
    quality: str | None = None,
    background: str | None = None,
    workers: int = 1,
    flat_output: bool = False,
    timeout: float | None = None,
    max_retries: int = 5,
    retry_delay: float = 1.0,
    style_dir: str | None = None,
    dry_run: bool = False,
) -> CallToolResult:
    """Generate image files from a JSON batch definition."""
    return _to_tool_result(
        _run_request(
            GenerationRequest(
                batch=_path(batch_path),
                output_dir=_default_output_dir(output_dir),
                model=model,
                style=style,
                style_dir=_path(style_dir),
                limit=limit,
                only=only,
                dry_run=dry_run,
                workers=workers,
                max_retries=max_retries,
                retry_delay=retry_delay,
                timeout=timeout,
                size=size,
                quality=quality,
                background=background,
                flat_output=flat_output,
            )
        )
    )


def run_mcp() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    run_mcp()
