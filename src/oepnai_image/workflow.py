from __future__ import annotations

import json
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .cli import (
    DEFAULT_MAX_RETRIES,
    build_client,
    collect_jobs,
    default_output_root,
    ensure_dir,
    filter_jobs,
    format_path_for_manifest,
    generate_job,
    load_styles,
    parse_args,
    read_env,
    resolve_size_argument,
    resolve_style,
    resolve_timeout_seconds,
    validate_main_args,
    GenerationOptions,
    ImageJob,
)


@dataclass(frozen=True)
class GenerationRequest:
    prompts: list[str] | None = None
    prompts_file: Path | None = None
    batch: Path | None = None
    output_dir: Path | None = None
    category: str = "misc"
    filename_prefix: str | None = None
    model: str | None = None
    style: str | None = None
    style_dir: Path | None = None
    num_images: int = 1
    limit: int | None = None
    only: list[str] | None = None
    dry_run: bool = False
    workers: int = 1
    max_retries: int = DEFAULT_MAX_RETRIES
    retry_delay: float = 1.0
    timeout: float | None = None
    size: str | None = None
    resolution: str | None = None
    width: int | None = None
    height: int | None = None
    aspect_ratio: str | None = None
    long_edge: int | None = None
    short_edge: int | None = None
    quality: str | None = None
    background: str | None = None
    flat_output: bool = False


def request_from_args(argv: list[str] | None = None) -> GenerationRequest:
    args = parse_args(argv)
    validate_main_args(args)
    return GenerationRequest(
        prompts=list(args.prompt or []),
        prompts_file=args.prompts_file,
        batch=args.batch,
        output_dir=args.output_dir,
        category=args.category,
        filename_prefix=args.filename_prefix,
        model=args.model,
        style=args.style,
        style_dir=args.style_dir,
        num_images=args.num_images,
        limit=args.limit,
        only=args.only,
        dry_run=args.dry_run,
        workers=args.workers,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
        timeout=args.timeout,
        size=args.size,
        resolution=args.resolution,
        width=args.width,
        height=args.height,
        aspect_ratio=args.aspect_ratio,
        long_edge=args.long_edge,
        short_edge=args.short_edge,
        quality=args.quality,
        background=args.background,
        flat_output=args.flat_output,
    )


def args_from_request(request: GenerationRequest) -> Any:
    args = argparse.Namespace(
        prompt=list(request.prompts or []),
        prompts_file=request.prompts_file,
        batch=request.batch,
        output_dir=request.output_dir,
        category=request.category,
        filename_prefix=request.filename_prefix,
        model=request.model,
        style=request.style,
        style_dir=request.style_dir,
        list_styles=False,
        num_images=request.num_images,
        limit=request.limit,
        only=request.only,
        dry_run=request.dry_run,
        workers=request.workers,
        max_retries=request.max_retries,
        retry_delay=request.retry_delay,
        timeout=request.timeout,
        size=request.size,
        resolution=request.resolution,
        width=request.width,
        height=request.height,
        aspect_ratio=request.aspect_ratio,
        long_edge=request.long_edge,
        short_edge=request.short_edge,
        quality=request.quality,
        background=request.background,
        flat_output=request.flat_output,
    )
    validate_main_args(args)
    return args


def run_generation_request(request: GenerationRequest) -> dict[str, Any]:
    return run_generation(args_from_request(request))


def run_generation_request_with_env(
    request: GenerationRequest,
    *,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    return run_generation(args_from_request(request), env=env)


def run_generation(args: Any, *, env: dict[str, str] | None = None) -> dict[str, Any]:
    resolved_env = env or read_env()
    timeout_seconds = resolve_timeout_seconds(
        str(args.timeout) if args.timeout is not None else resolved_env.get("timeout")
    )
    resolved_size = resolve_size_argument(args)
    jobs, source = collect_jobs(args)
    jobs = filter_jobs(jobs, only=args.only, limit=args.limit)

    if not jobs:
        raise RuntimeError("No jobs matched the current filters.")

    return run_jobs(
        jobs=jobs,
        source=source,
        output_dir=args.output_dir,
        env=resolved_env,
        style_name=args.style,
        style_dir=args.style_dir,
        options=GenerationOptions(
            size_override=resolved_size,
            quality_override=args.quality,
            model_override=args.model,
            background_override=args.background,
            flat_output=args.flat_output,
            max_retries=args.max_retries,
            retry_delay_seconds=args.retry_delay,
            timeout_seconds=timeout_seconds,
            dry_run=args.dry_run,
        ),
        workers=args.workers,
    )


def run_jobs(
    *,
    jobs: list[ImageJob],
    source: str,
    output_dir: Path | None,
    env: dict[str, str],
    style_name: str | None,
    style_dir: Path | None,
    options: GenerationOptions,
    workers: int,
) -> dict[str, Any]:
    styles = load_styles(style_dir) if style_name or any(job.style for job in jobs) else {}
    resolved_output_dir = output_dir or (default_output_root() / datetime.now().strftime("%Y%m%d-%H%M%S"))
    if not options.dry_run:
        ensure_dir(resolved_output_dir)

    client = build_client(env, timeout_seconds=options.timeout_seconds) if not options.dry_run and workers == 1 else None
    manifest: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": source,
        "output_dir": format_path_for_manifest(resolved_output_dir),
        "jobs": [],
    }

    if workers == 1 or options.dry_run:
        for job in jobs:
            manifest["jobs"].append(
                generate_job(
                    client=client,
                    job=job,
                    output_dir=resolved_output_dir,
                    env=env,
                    style=resolve_style(style_name or job.style, styles),
                    options=options,
                )
            )
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_map = {
                executor.submit(
                    generate_job,
                    None,
                    job,
                    resolved_output_dir,
                    env,
                    resolve_style(style_name or job.style, styles),
                    options,
                ): job.slug
                for job in jobs
            }
            for future in as_completed(future_map):
                manifest["jobs"].append(future.result())
        manifest["jobs"].sort(key=lambda item: item["slug"])

    if options.dry_run:
        manifest["dry_run"] = True
    else:
        manifest_path = resolved_output_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        manifest["manifest_path"] = format_path_for_manifest(manifest_path)
    return manifest
