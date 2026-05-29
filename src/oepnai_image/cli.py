from __future__ import annotations

import argparse
import base64
import json
import os
import re
import subprocess
import time
import sys
import urllib.request
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from importlib import resources
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from dotenv import find_dotenv, load_dotenv
from openai import OpenAI
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
FALLBACK_STYLE_DIR = REPO_ROOT / "prompt_styles"
DEFAULT_OUTPUT_DIRNAME = "generated_images"
DEFAULT_MAX_RETRIES = 5
DEFAULT_TIMEOUT_SECONDS = 1200.0
NON_RETRYABLE_STATUS_CODES = {400, 401, 403, 404}
SIZE_DIVISIBILITY = 16
BUILTIN_STYLE_RESOURCE_DIR = "prompt_styles"


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = re.sub(r"-{2,}", "-", value)
    return value.strip("-") or "image"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def default_output_root() -> Path:
    return Path.cwd() / DEFAULT_OUTPUT_DIRNAME


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def format_path_for_manifest(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(resolved)


def load_prompts_file(path: Path) -> list[str]:
    prompts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [prompt for prompt in prompts if prompt]


def parse_resolution(value: str) -> tuple[int, int]:
    match = re.fullmatch(r"\s*(\d+)\s*x\s*(\d+)\s*", value, flags=re.IGNORECASE)
    if not match:
        raise RuntimeError("Resolution must look like WIDTHxHEIGHT, for example 1536x1024.")
    width = int(match.group(1))
    height = int(match.group(2))
    if width < 1 or height < 1:
        raise RuntimeError("Resolution width and height must be positive integers.")
    validate_size_dimensions(width, height)
    return width, height


def parse_aspect_ratio(value: str) -> tuple[int, int]:
    match = re.fullmatch(r"\s*(\d+)\s*[:/]\s*(\d+)\s*", value)
    if not match:
        raise RuntimeError("Aspect ratio must look like WIDTH:HEIGHT, for example 16:9.")
    width_ratio = int(match.group(1))
    height_ratio = int(match.group(2))
    if width_ratio < 1 or height_ratio < 1:
        raise RuntimeError("Aspect ratio values must be positive integers.")
    return width_ratio, height_ratio


def round_dimension(value: float) -> int:
    return max(1, int(round(value)))


def round_to_multiple(value: int, divisor: int) -> int:
    return max(divisor, int(round(value / divisor)) * divisor)


def validate_size_dimensions(width: int, height: int) -> None:
    if width % SIZE_DIVISIBILITY != 0 or height % SIZE_DIVISIBILITY != 0:
        raise RuntimeError(
            f"Invalid size '{width}x{height}'. Width and height must both be divisible by {SIZE_DIVISIBILITY}."
        )


def build_size_from_ratio(
    aspect_ratio: str,
    *,
    long_edge: int | None,
    short_edge: int | None,
) -> str:
    width_ratio, height_ratio = parse_aspect_ratio(aspect_ratio)
    if long_edge is not None and short_edge is not None:
        raise RuntimeError("Use either --long-edge or --short-edge with --aspect-ratio, not both.")
    if long_edge is None and short_edge is None:
        long_edge = 1536

    if width_ratio >= height_ratio:
        if long_edge is not None:
            width = long_edge
            height = round_dimension(long_edge * height_ratio / width_ratio)
        else:
            height = short_edge or 1024
            width = round_dimension(height * width_ratio / height_ratio)
    else:
        if long_edge is not None:
            height = long_edge
            width = round_dimension(long_edge * width_ratio / height_ratio)
        else:
            width = short_edge or 1024
            height = round_dimension(width * height_ratio / width_ratio)

    width = round_to_multiple(width, SIZE_DIVISIBILITY)
    height = round_to_multiple(height, SIZE_DIVISIBILITY)
    validate_size_dimensions(width, height)
    return f"{width}x{height}"


def resolve_size_argument(args: argparse.Namespace) -> str | None:
    explicit_sizes = [value for value in [args.size, args.resolution] if value]
    if explicit_sizes and args.aspect_ratio:
        raise RuntimeError("Use either --size/--resolution or --aspect-ratio, not both.")
    if len(explicit_sizes) > 1 and args.size != args.resolution:
        raise RuntimeError("--size and --resolution both set different values.")
    if args.width is not None or args.height is not None:
        if explicit_sizes or args.aspect_ratio:
            raise RuntimeError("--width/--height cannot be combined with --size, --resolution, or --aspect-ratio.")
        if args.width is None or args.height is None:
            raise RuntimeError("Use --width and --height together.")
        if args.width < 1 or args.height < 1:
            raise RuntimeError("--width and --height must be at least 1.")
        validate_size_dimensions(args.width, args.height)
        return f"{args.width}x{args.height}"
    if explicit_sizes:
        width, height = parse_resolution(explicit_sizes[0])
        return f"{width}x{height}"
    if args.aspect_ratio:
        if args.long_edge is not None and args.long_edge < 1:
            raise RuntimeError("--long-edge must be at least 1.")
        if args.short_edge is not None and args.short_edge < 1:
            raise RuntimeError("--short-edge must be at least 1.")
        return build_size_from_ratio(
            args.aspect_ratio,
            long_edge=args.long_edge,
            short_edge=args.short_edge,
        )
    if args.long_edge is not None or args.short_edge is not None:
        raise RuntimeError("--long-edge or --short-edge requires --aspect-ratio.")
    return None


@dataclass
class ImageJob:
    slug: str
    prompt: str
    category: str = "misc"
    filename_prefix: str | None = None
    model: str | None = None
    size: str | None = None
    quality: str | None = None
    background: str | None = None
    n: int | None = None
    style: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any], defaults: dict[str, Any]) -> "ImageJob":
        merged = {**defaults, **data}
        missing = [key for key in ["slug", "prompt"] if key not in merged or not str(merged[key]).strip()]
        if missing:
            raise RuntimeError(f"Batch job is missing required field(s): {', '.join(missing)}.")
        slug = slugify(str(merged["slug"]))
        return cls(
            slug=slug,
            prompt=str(merged["prompt"]).strip(),
            category=slugify(str(merged.get("category", "misc"))),
            filename_prefix=slugify(str(merged.get("filename_prefix") or merged["slug"])),
            model=merged.get("model"),
            size=merged.get("size"),
            quality=merged.get("quality"),
            background=merged.get("background"),
            n=validate_num_images(merged.get("n", 1), source=f"Batch job '{slug}' field 'n'"),
            style=merged.get("style"),
        )


@dataclass(frozen=True)
class PromptStyle:
    slug: str
    name: str
    description: str
    template: str
    defaults: dict[str, Any]


@dataclass(frozen=True)
class GenerationOptions:
    size_override: str | None
    quality_override: str | None
    model_override: str | None
    background_override: str | None
    flat_output: bool
    max_retries: int
    retry_delay_seconds: float
    timeout_seconds: float
    dry_run: bool


@dataclass(frozen=True)
class ResolvedJobConfig:
    category: str
    model: str
    background: str
    num_images: int
    size: str | None
    quality: str | None


def make_runtime_error(message: str, *, status_code: int | None = None) -> RuntimeError:
    error = RuntimeError(message)
    if status_code is not None:
        error.status_code = status_code  # type: ignore[attr-defined]
    return error


def validate_num_images(value: Any, *, source: str) -> int:
    try:
        count = int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{source} must be an integer greater than or equal to 1.") from exc
    if count < 1:
        raise RuntimeError(f"{source} must be greater than or equal to 1.")
    return count


def load_style_definition(data: dict[str, Any], *, source: str) -> PromptStyle:
    slug = slugify(str(data.get("slug") or Path(source).stem))
    template = str(data.get("template") or "").strip()
    if not template:
        raise RuntimeError(f"Style template is missing or empty: {source}")
    return PromptStyle(
        slug=slug,
        name=str(data.get("name") or slug),
        description=str(data.get("description") or "").strip(),
        template=template,
        defaults=dict(data.get("defaults") or {}),
    )


def load_style_file(path: Path) -> PromptStyle:
    return load_style_definition(load_json(path), source=str(path))


def load_builtin_styles() -> dict[str, PromptStyle]:
    styles: dict[str, PromptStyle] = {}
    try:
        resource_root = resources.files("oepnai_image").joinpath(BUILTIN_STYLE_RESOURCE_DIR)
        if resource_root.is_dir():
            for resource in sorted(resource_root.iterdir(), key=lambda item: item.name):
                if resource.name.endswith(".json"):
                    style = load_style_definition(
                        json.loads(resource.read_text(encoding="utf-8")),
                        source=f"{BUILTIN_STYLE_RESOURCE_DIR}/{resource.name}",
                    )
                    styles[style.slug] = style
            return styles
    except (FileNotFoundError, ModuleNotFoundError):
        pass

    if FALLBACK_STYLE_DIR.exists():
        for path in sorted(FALLBACK_STYLE_DIR.glob("*.json")):
            style = load_style_file(path)
            styles[style.slug] = style
    return styles


def load_styles(style_dir: Path | None = None) -> dict[str, PromptStyle]:
    styles = load_builtin_styles()
    if style_dir is None or not style_dir.exists():
        return styles
    for path in sorted(style_dir.glob("*.json")):
        style = load_style_file(path)
        styles[style.slug] = style
    return styles


def format_style_listing(styles: dict[str, PromptStyle]) -> str:
    if not styles:
        return "No styles found."
    lines = ["Available styles:"]
    for slug, style in sorted(styles.items()):
        description = style.description or "No description."
        lines.append(f"- {slug}: {description}")
    return "\n".join(lines)


def validate_main_args(args: argparse.Namespace) -> None:
    if args.num_images < 1:
        raise RuntimeError("--num-images must be at least 1.")
    if args.limit is not None and args.limit < 1:
        raise RuntimeError("--limit must be at least 1.")
    if args.workers < 1:
        raise RuntimeError("--workers must be at least 1.")
    if args.max_retries < 1:
        raise RuntimeError("--max-retries must be at least 1.")
    if args.max_retries > DEFAULT_MAX_RETRIES:
        raise RuntimeError(f"--max-retries cannot be greater than {DEFAULT_MAX_RETRIES}.")
    if args.retry_delay < 0:
        raise RuntimeError("--retry-delay must be 0 or greater.")


def resolve_style(style_name: str | None, styles: dict[str, PromptStyle]) -> PromptStyle | None:
    if not style_name:
        return None
    slug = slugify(style_name)
    style = styles.get(slug)
    if style is None:
        available = ", ".join(sorted(styles)) or "none"
        raise RuntimeError(f"Unknown style '{style_name}'. Available styles: {available}.")
    return style


def compose_prompt(base_prompt: str, style: PromptStyle | None) -> str:
    prompt = base_prompt.strip()
    if not style:
        return prompt
    if "{prompt}" in style.template:
        return style.template.format(prompt=prompt)
    return f"{prompt}\n\n{style.template}"


def read_env() -> dict[str, str]:
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path, override=False)
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()
    image_model = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-2").strip()
    image_transport = os.getenv("OPENAI_IMAGE_TRANSPORT", "auto").strip().lower()
    timeout = os.getenv("OPENAI_TIMEOUT_SECONDS", str(DEFAULT_TIMEOUT_SECONDS)).strip()
    return {
        "api_key": api_key,
        "base_url": base_url,
        "image_model": image_model,
        "image_transport": image_transport,
        "timeout": timeout,
    }


def resolve_timeout_seconds(value: str | None) -> float:
    if value is None or not str(value).strip():
        return DEFAULT_TIMEOUT_SECONDS
    try:
        timeout = float(value)
    except ValueError as exc:
        raise RuntimeError("Timeout must be a number of seconds.") from exc
    if timeout <= 0:
        raise RuntimeError("Timeout must be greater than 0 seconds.")
    return timeout


def build_client(env: dict[str, str], timeout_seconds: float | None = None) -> OpenAI:
    if not env["api_key"]:
        raise RuntimeError("OPENAI_API_KEY is missing. Set it in the environment or in a local .env file.")
    timeout = timeout_seconds if timeout_seconds is not None else resolve_timeout_seconds(env.get("timeout"))
    return OpenAI(
        api_key=env["api_key"],
        base_url=env["base_url"],
        timeout=timeout,
        max_retries=0,
    )


def should_use_curl_transport(env: dict[str, str]) -> bool:
    transport = env.get("image_transport", "auto")
    if transport not in {"auto", "sdk", "curl"}:
        raise RuntimeError("OPENAI_IMAGE_TRANSPORT must be one of: auto, sdk, curl.")
    if transport == "curl":
        return True
    if transport == "sdk":
        return False
    base_url = env.get("base_url", "https://api.openai.com/v1").lower()
    return "api.openai.com" not in base_url


def should_retry_exception(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    return status_code not in NON_RETRYABLE_STATUS_CODES


def format_terminal_api_error(exc: Exception) -> str:
    status_code = getattr(exc, "status_code", None)
    message = str(exc)

    if status_code == 403 and "Image generation is not enabled for this group" in message:
        return (
            "Image generation is not enabled for this provider group/key. "
            "The base URL is reaching the API, but this account does not have image-generation permission."
        )

    return message


def response_item_to_png_bytes(item: Any, timeout_seconds: float) -> bytes:
    b64_json = getattr(item, "b64_json", None)
    if b64_json:
        return base64.b64decode(b64_json)

    url = getattr(item, "url", None)
    if url:
        with urllib.request.urlopen(url, timeout=timeout_seconds) as response:
            return response.read()

    raise RuntimeError("Image response did not contain b64_json or url data.")


def image_generation_endpoint(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/images/generations"


def generate_image_with_curl(
    env: dict[str, str],
    payload: dict[str, Any],
    *,
    timeout_seconds: float,
) -> Any:
    if not env["api_key"]:
        raise RuntimeError("OPENAI_API_KEY is missing. Set it in the environment or in a local .env file.")

    command = [
        "curl",
        "-sS",
        "--max-time",
        f"{timeout_seconds:g}",
        image_generation_endpoint(env["base_url"]),
        "-H",
        "Content-Type: application/json",
        "-H",
        f"Authorization: Bearer {env['api_key']}",
        "-d",
        json.dumps(payload),
        "--write-out",
        "\n__HTTP_STATUS__:%{http_code}",
    ]
    result = subprocess.run(command, check=False, capture_output=True)
    stdout_with_status = result.stdout.decode("utf-8", errors="replace")
    stderr = result.stderr.decode("utf-8", errors="replace").strip()
    stdout, _, status_line = stdout_with_status.rpartition("\n__HTTP_STATUS__:")
    http_status = int(status_line) if status_line.isdigit() else None

    if result.returncode != 0:
        detail = stderr or stdout or stdout_with_status or f"curl exited with status {result.returncode}"
        raise RuntimeError(detail)

    try:
        response = json.loads(stdout)
    except json.JSONDecodeError as exc:
        snippet = stdout[:1000] if stdout else stderr
        raise RuntimeError(f"Image generation returned non-JSON response: {snippet}") from exc

    if "error" in response:
        error = response["error"]
        if isinstance(error, dict):
            message = error.get("message") or json.dumps(error, ensure_ascii=False)
        else:
            message = str(error)
        raise make_runtime_error(message, status_code=http_status)

    return json.loads(json.dumps(response), object_hook=lambda data: SimpleNamespace(**data))


def inspect_png_bytes(image_bytes: bytes) -> dict[str, Any]:
    with Image.open(BytesIO(image_bytes)) as image:
        original_size = image.size
        return {
            "original_size": f"{original_size[0]}x{original_size[1]}",
            "final_size": f"{original_size[0]}x{original_size[1]}",
            "resized": False,
        }


def generate_with_retries(
    client: OpenAI,
    payload: dict[str, Any],
    *,
    job_slug: str,
    max_retries: int,
    retry_delay_seconds: float,
) -> Any:
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            return client.images.generate(**payload)
        except Exception as exc:
            last_error = exc
            if not should_retry_exception(exc):
                raise RuntimeError(format_terminal_api_error(exc)) from exc
            if attempt >= max_retries:
                break
            print(
                f"[retry] {job_slug} attempt {attempt}/{max_retries} failed: {exc}",
                file=sys.stderr,
            )
            time.sleep(retry_delay_seconds)
    assert last_error is not None
    raise RuntimeError(
        f"Image generation failed for {job_slug} after {max_retries} attempts: {last_error}"
    ) from last_error


def make_prompt_jobs(
    prompts: list[str],
    *,
    model: str | None,
    size: str | None,
    quality: str | None,
    background: str | None,
    num_images: int,
    category: str,
    filename_prefix: str | None,
    style: str | None,
) -> list[ImageJob]:
    jobs: list[ImageJob] = []
    normalized_prefix = slugify(filename_prefix) if filename_prefix else None
    for index, prompt in enumerate(prompts, start=1):
        if not prompt.strip():
            continue
        prompt_slug = slugify(prompt[:80])
        slug = f"prompt-{index:02d}-{prompt_slug}"
        prefix = normalized_prefix or slug
        if normalized_prefix and len(prompts) > 1:
            prefix = f"{normalized_prefix}-{index:02d}"
        jobs.append(
            ImageJob(
                slug=slug,
                prompt=prompt.strip(),
                category=slugify(category),
                filename_prefix=prefix,
                model=model,
                size=size,
                quality=quality,
                background=background,
                n=num_images,
                style=style,
            )
        )
    return jobs


def load_batch_jobs(batch_path: Path) -> list[ImageJob]:
    batch_data = load_json(batch_path)
    defaults = batch_data.get("defaults", {})
    jobs = batch_data.get("jobs", [])
    if not jobs:
        raise RuntimeError(f"No jobs found in batch file: {batch_path}")
    return [ImageJob.from_dict(job, defaults) for job in jobs]


def collect_jobs(args: argparse.Namespace) -> tuple[list[ImageJob], str]:
    jobs: list[ImageJob] = []
    sources: list[str] = []

    if args.batch:
        jobs.extend(load_batch_jobs(args.batch))
        sources.append(f"batch:{args.batch}")

    prompts: list[str] = list(args.prompt or [])
    if args.prompts_file:
        prompts.extend(load_prompts_file(args.prompts_file))
        sources.append(f"prompts-file:{args.prompts_file}")
    elif prompts:
        sources.append("cli-prompts")

    if prompts:
        jobs.extend(
            make_prompt_jobs(
                prompts,
                model=args.model,
                size=args.size,
                quality=args.quality,
                background=args.background,
                num_images=args.num_images,
                category=args.category,
                filename_prefix=args.filename_prefix,
                style=args.style,
            )
        )

    if not jobs:
        raise RuntimeError("No jobs found. Use --prompt, --prompts-file, or --batch.")

    return jobs, ", ".join(sources) or "unknown"


def filter_jobs(
    jobs: list[ImageJob],
    *,
    only: list[str] | None,
    limit: int | None,
) -> list[ImageJob]:
    selected_slugs = {slugify(item) for item in only or []}
    if selected_slugs:
        jobs = [job for job in jobs if job.slug in selected_slugs]
    if limit is not None:
        jobs = jobs[:limit]
    return jobs


def resolve_job_config(
    job: ImageJob,
    env: dict[str, str],
    style: PromptStyle | None,
    options: GenerationOptions,
) -> ResolvedJobConfig:
    style_defaults = style.defaults if style else {}
    category = (
        slugify(str(style_defaults["category"]))
        if job.category == "misc" and style_defaults.get("category")
        else job.category
    )
    background = (
        options.background_override
        if options.background_override is not None
        else job.background or style_defaults.get("background") or "opaque"
    )
    size = (
        options.size_override
        if options.size_override is not None
        else job.size or style_defaults.get("size")
    )
    quality = (
        options.quality_override
        if options.quality_override is not None
        else job.quality or style_defaults.get("quality")
    )
    if size:
        parse_resolution(str(size))
    return ResolvedJobConfig(
        category=category,
        model=options.model_override or job.model or style_defaults.get("model") or env["image_model"],
        background=background,
        num_images=job.n if job.n is not None else 1,
        size=size,
        quality=quality,
    )


def build_generation_payload(
    job: ImageJob,
    style: PromptStyle | None,
    config: ResolvedJobConfig,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": config.model,
        "prompt": compose_prompt(job.prompt, style),
    }
    if config.background != "opaque":
        payload["background"] = config.background
    if config.num_images != 1:
        payload["n"] = config.num_images
    if config.size:
        payload["size"] = config.size
    if config.quality:
        payload["quality"] = config.quality
    return payload


def generate_job(
    client: OpenAI | None,
    job: ImageJob,
    output_dir: Path,
    env: dict[str, str],
    style: PromptStyle | None,
    options: GenerationOptions,
) -> dict[str, Any]:
    started_at = time.monotonic()
    config = resolve_job_config(job, env, style, options)
    target_dir = output_dir if options.flat_output else (output_dir / config.category)
    if not options.dry_run:
        ensure_dir(target_dir)
    payload = build_generation_payload(job, style, config)

    print(
        f"[generate] {job.slug} -> {config.category} "
        f"({payload['model']}, size={payload.get('size', 'default')}, "
        f"quality={payload.get('quality', 'default')}, n={config.num_images})",
        flush=True,
    )

    if options.dry_run:
        return {
            "slug": job.slug,
            "category": config.category,
            "style": style.slug if style else None,
            "base_prompt": job.prompt,
            "payload": payload,
            "files": [],
            "dry_run": True,
        }

    if should_use_curl_transport(env):
        response = generate_with_retries(
            SimpleNamespace(
                images=SimpleNamespace(
                    generate=lambda **request_payload: generate_image_with_curl(
                        env,
                        request_payload,
                        timeout_seconds=options.timeout_seconds,
                    )
                )
            ),
            payload,
            job_slug=job.slug,
            max_retries=options.max_retries,
            retry_delay_seconds=options.retry_delay_seconds,
        )
    else:
        if client is None:
            client = build_client(env, timeout_seconds=options.timeout_seconds)

        response = generate_with_retries(
            client,
            payload,
            job_slug=job.slug,
            max_retries=options.max_retries,
            retry_delay_seconds=options.retry_delay_seconds,
        )
    print(f"[received] {job.slug} API response in {time.monotonic() - started_at:.1f}s", flush=True)
    if not getattr(response, "data", None):
        raise RuntimeError(f"Image generation returned no image data for {job.slug}.")
    files: list[str] = []
    images: list[dict[str, Any]] = []
    for index, item in enumerate(response.data, start=1):
        suffix = f"-{index:02d}" if len(response.data) > 1 else ""
        filename = f"{job.filename_prefix}{suffix}.png"
        file_path = target_dir / filename
        print(f"[save] {job.slug} image {index}/{len(response.data)} -> {file_path}", flush=True)
        image_bytes = response_item_to_png_bytes(item, options.timeout_seconds)
        image_metadata = inspect_png_bytes(image_bytes)
        file_path.write_bytes(image_bytes)
        print(
            f"[saved] {job.slug} image {index}/{len(response.data)} "
            f"({image_metadata['original_size']} -> {image_metadata['final_size']})",
            flush=True,
        )
        files.append(format_path_for_manifest(file_path))
        images.append(
            {
                "file": format_path_for_manifest(file_path),
                "requested_size": config.size,
                **image_metadata,
            }
        )

    return {
        "slug": job.slug,
        "category": config.category,
        "style": style.slug if style else None,
        "base_prompt": job.prompt,
        "payload": payload,
        "files": files,
        "images": images,
        "duration_seconds": round(time.monotonic() - started_at, 3),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate OpenAI images from direct prompts, prompt files, or batch JSON."
    )
    parser.add_argument(
        "--prompt",
        action="append",
        default=[],
        help="Prompt text. Repeat this flag to generate from multiple prompts.",
    )
    parser.add_argument(
        "--prompts-file",
        type=Path,
        default=None,
        help="Text file where each non-empty line is a prompt.",
    )
    parser.add_argument(
        "--batch",
        type=Path,
        default=None,
        help="Optional JSON batch definition file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to generated_images/<timestamp>.",
    )
    parser.add_argument(
        "--category",
        default="misc",
        help="Category folder used for prompts passed from CLI.",
    )
    parser.add_argument(
        "--filename-prefix",
        default=None,
        help="Filename prefix for CLI prompts. With multiple prompts, an index is appended.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional model override, for example gpt-image-2.",
    )
    parser.add_argument(
        "--style",
        default=None,
        help="Prompt style slug to apply, for example paper-figure.",
    )
    parser.add_argument(
        "--style-dir",
        type=Path,
        default=None,
        help="Optional directory containing extra JSON prompt styles.",
    )
    parser.add_argument(
        "--list-styles",
        action="store_true",
        help="List available styles and exit.",
    )
    parser.add_argument(
        "--num-images",
        "-n",
        type=int,
        default=1,
        help="Number of images to generate per prompt.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only run the first N resolved jobs.",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Only run jobs with these slugs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved jobs without calling the API.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of jobs to run in parallel.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help="Maximum retries for a failed API request.",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=1.0,
        help="Delay in seconds between retry attempts.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help=f"Per API request and image download timeout in seconds. Default: {DEFAULT_TIMEOUT_SECONDS:g}.",
    )
    parser.add_argument(
        "--size",
        default=None,
        help="Optional size override. Example: 1024x1024.",
    )
    parser.add_argument(
        "--resolution",
        default=None,
        help="Alias for --size. Example: 1536x1024.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Width in pixels. Use together with --height.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Height in pixels. Use together with --width.",
    )
    parser.add_argument(
        "--aspect-ratio",
        default=None,
        help="Aspect ratio like 1:1, 4:3, 3:4, 16:9, or 9:16.",
    )
    parser.add_argument(
        "--long-edge",
        type=int,
        default=None,
        help="Long edge in pixels when using --aspect-ratio. Default: 1536.",
    )
    parser.add_argument(
        "--short-edge",
        type=int,
        default=None,
        help="Short edge in pixels when using --aspect-ratio.",
    )
    parser.add_argument(
        "--quality",
        default=None,
        help="Optional quality override. Example: low, medium, high.",
    )
    parser.add_argument(
        "--background",
        default=None,
        help="Optional background override. Example: transparent or opaque.",
    )
    parser.add_argument(
        "--flat-output",
        action="store_true",
        help="Write images directly into the output directory without category subfolders.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_main_args(args)

    if args.list_styles:
        styles = load_styles(args.style_dir)
        print(format_style_listing(styles))
        return 0

    env = read_env()
    timeout_seconds = resolve_timeout_seconds(
        str(args.timeout) if args.timeout is not None else env.get("timeout")
    )
    resolved_size = resolve_size_argument(args)
    jobs, source = collect_jobs(args)
    jobs = filter_jobs(jobs, only=args.only, limit=args.limit)

    if not jobs:
        raise RuntimeError("No jobs matched the current filters.")

    styles = load_styles(args.style_dir) if args.style or any(job.style for job in jobs) else {}
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = args.output_dir or (default_output_root() / timestamp)
    if not args.dry_run:
        ensure_dir(output_dir)

    client = build_client(env, timeout_seconds=timeout_seconds) if not args.dry_run and args.workers == 1 else None
    options = GenerationOptions(
        size_override=resolved_size,
        quality_override=args.quality,
        model_override=args.model,
        background_override=args.background,
        flat_output=args.flat_output,
        max_retries=args.max_retries,
        retry_delay_seconds=args.retry_delay,
        timeout_seconds=timeout_seconds,
        dry_run=args.dry_run,
    )
    manifest: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": source,
        "output_dir": format_path_for_manifest(output_dir),
        "jobs": [],
    }

    if args.workers == 1 or args.dry_run:
        for job in jobs:
            result = generate_job(
                client=client,
                job=job,
                output_dir=output_dir,
                env=env,
                style=resolve_style(args.style or job.style, styles),
                options=options,
            )
            manifest["jobs"].append(result)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_map = {
                executor.submit(
                    generate_job,
                    None,
                    job,
                    output_dir,
                    env,
                    resolve_style(args.style or job.style, styles),
                    options,
                ): job.slug
                for job in jobs
            }
            for future in as_completed(future_map):
                manifest["jobs"].append(future.result())
        manifest["jobs"].sort(key=lambda item: item["slug"])

    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        print("[done] dry run only; no files written")
    else:
        manifest_path = output_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[done] wrote manifest: {manifest_path}")
    return 0


def run_cli(argv: list[str] | None = None) -> int:
    try:
        return main(argv)
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(run_cli())
