from __future__ import annotations

import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from .cli import format_style_listing, load_styles
from .workflow import GenerationRequest, run_generation_request


mcp = FastMCP("openai-image", json_response=True)


def _path(value: str | None) -> Path | None:
    return Path(value).expanduser() if value else None


def _run_request(request: GenerationRequest) -> dict[str, Any]:
    # MCP stdio reserves stdout for JSON-RPC frames. The CLI workflow prints
    # human progress lines, so route them to stderr when called as a tool.
    with redirect_stdout(sys.stderr):
        return run_generation_request(request)


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
) -> dict[str, Any]:
    """Resolve the payload and manifest shape for one prompt without calling the image API."""
    return _run_request(
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
    category: str = "misc",
    filename_prefix: str | None = None,
    flat_output: bool = False,
    timeout: float | None = None,
    max_retries: int = 5,
    retry_delay: float = 1.0,
    style_dir: str | None = None,
) -> dict[str, Any]:
    """Generate image files from one prompt and return the manifest with saved file paths."""
    return _run_request(
        GenerationRequest(
            prompts=[prompt],
            output_dir=_path(output_dir),
            category=category,
            filename_prefix=filename_prefix,
            model=model,
            style=style,
            style_dir=_path(style_dir),
            num_images=num_images,
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
) -> dict[str, Any]:
    """Generate image files from a JSON batch definition."""
    return _run_request(
        GenerationRequest(
            batch=_path(batch_path),
            output_dir=_path(output_dir),
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


def run_mcp() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    run_mcp()
