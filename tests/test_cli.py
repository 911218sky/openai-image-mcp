from __future__ import annotations

import base64
import json
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from oepnai_image.cli import (
    build_size_from_ratio,
    build_client,
    collect_jobs,
    compose_prompt,
    default_output_root,
    format_style_listing,
    format_terminal_api_error,
    generate_image_with_http,
    GenerationOptions,
    generate_job,
    generate_with_retries,
    ImageJob,
    inspect_png_bytes,
    load_batch_jobs,
    load_styles,
    main,
    make_prompt_jobs,
    parse_args,
    response_item_to_png_bytes,
    resolve_timeout_seconds,
    resolve_style,
    run_cli,
    resolve_size_argument,
    should_use_http_transport,
    should_retry_exception,
    slugify,
)
from oepnai_image.workflow import GenerationRequest, run_generation_request


def test_slugify_falls_back_to_image() -> None:
    assert slugify("  !!!  ") == "image"


def test_make_prompt_jobs_creates_distinct_jobs() -> None:
    jobs = make_prompt_jobs(
        ["A red car", "A blue car"],
        model="gpt-image-2",
        size="1024x1024",
        quality=None,
        background="opaque",
        num_images=2,
        category="concept art",
        filename_prefix=None,
        style="paper-figure",
    )

    assert [job.slug for job in jobs] == [
        "prompt-01-a-red-car",
        "prompt-02-a-blue-car",
    ]
    assert all(job.n == 2 for job in jobs)
    assert all(job.category == "concept-art" for job in jobs)
    assert all(job.style == "paper-figure" for job in jobs)


def test_make_prompt_jobs_uses_filename_prefix_with_index() -> None:
    jobs = make_prompt_jobs(
        ["A red car", "A blue car"],
        model="gpt-image-2",
        size=None,
        quality=None,
        background=None,
        num_images=1,
        category="misc",
        filename_prefix="hero shot",
        style=None,
    )

    assert [job.filename_prefix for job in jobs] == [
        "hero-shot-01",
        "hero-shot-02",
    ]


def test_collect_jobs_merges_cli_prompts_and_file_prompts(tmp_path: Path) -> None:
    prompts_file = tmp_path / "prompts.txt"
    prompts_file.write_text("line one\n\nline two\n", encoding="utf-8")

    args = parse_args(
        [
            "--prompt",
            "inline prompt",
            "--prompts-file",
            str(prompts_file),
            "--filename-prefix",
            "campaign",
            "--style",
            "paper-figure",
        ]
    )
    jobs, source = collect_jobs(args)

    assert len(jobs) == 3
    assert source == f"prompts-file:{prompts_file}"
    assert jobs[0].prompt == "inline prompt"
    assert jobs[1].prompt == "line one"
    assert jobs[2].prompt == "line two"
    assert [job.filename_prefix for job in jobs] == [
        "campaign-01",
        "campaign-02",
        "campaign-03",
    ]
    assert all(job.style == "paper-figure" for job in jobs)


def test_collect_jobs_loads_batch(tmp_path: Path) -> None:
    batch_path = tmp_path / "jobs.json"
    batch_path.write_text(
        json.dumps(
            {
                "defaults": {"category": "drafts", "n": 1},
                "jobs": [
                    {"slug": "alpha", "prompt": "first"},
                    {"slug": "beta", "prompt": "second", "n": 3},
                ],
            }
        ),
        encoding="utf-8",
    )

    args = parse_args(["--batch", str(batch_path)])
    jobs, source = collect_jobs(args)

    assert source == f"batch:{batch_path}"
    assert [job.slug for job in jobs] == ["alpha", "beta"]
    assert jobs[0].category == "drafts"
    assert jobs[1].n == 3


def test_load_batch_jobs_requires_slug_and_prompt(tmp_path: Path) -> None:
    batch_path = tmp_path / "jobs.json"
    batch_path.write_text(
        json.dumps(
            {
                "jobs": [
                    {"slug": "missing-prompt"},
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="prompt"):
        load_batch_jobs(batch_path)


def test_load_batch_jobs_rejects_non_positive_n(tmp_path: Path) -> None:
    batch_path = tmp_path / "jobs.json"
    batch_path.write_text(
        json.dumps(
            {
                "jobs": [
                    {"slug": "bad", "prompt": "oops", "n": 0},
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="greater than or equal to 1"):
        load_batch_jobs(batch_path)


def test_resolve_size_argument_accepts_resolution_alias() -> None:
    args = parse_args(["--prompt", "test", "--resolution", "1536x1024"])
    assert resolve_size_argument(args) == "1536x1024"


def test_build_size_from_ratio_with_default_long_edge() -> None:
    assert build_size_from_ratio("16:9", long_edge=None, short_edge=None) == "1536x864"


def test_resolve_size_argument_from_width_and_height() -> None:
    args = parse_args(["--prompt", "test", "--width", "1200", "--height", "800"])
    assert resolve_size_argument(args) == "1200x800"


def test_resolve_size_argument_rejects_conflicting_inputs() -> None:
    args = parse_args(["--prompt", "test", "--size", "1024x1024", "--aspect-ratio", "16:9"])
    with pytest.raises(RuntimeError):
        resolve_size_argument(args)


def test_generate_with_retries_succeeds_before_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"count": 0}

    def fake_generate(**_: object) -> dict[str, str]:
        calls["count"] += 1
        if calls["count"] < 3:
            raise RuntimeError("temporary failure")
        return {"ok": "yes"}

    client = SimpleNamespace(images=SimpleNamespace(generate=fake_generate))
    monkeypatch.setattr("oepnai_image.cli.time.sleep", lambda _: None)

    result = generate_with_retries(
        client,
        {"prompt": "test"},
        job_slug="job-1",
        max_retries=5,
        retry_delay_seconds=0,
    )

    assert result == {"ok": "yes"}
    assert calls["count"] == 3


def test_generate_with_retries_raises_after_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"count": 0}

    def fake_generate(**_: object) -> dict[str, str]:
        calls["count"] += 1
        raise RuntimeError("still failing")

    client = SimpleNamespace(images=SimpleNamespace(generate=fake_generate))
    monkeypatch.setattr("oepnai_image.cli.time.sleep", lambda _: None)

    with pytest.raises(RuntimeError, match="after 5 attempts"):
        generate_with_retries(
            client,
            {"prompt": "test"},
            job_slug="job-2",
            max_retries=5,
            retry_delay_seconds=0,
        )

    assert calls["count"] == 5


def test_should_retry_exception_false_for_403() -> None:
    exc = RuntimeError("permission denied")
    exc.status_code = 403  # type: ignore[attr-defined]

    assert should_retry_exception(exc) is False


def test_should_use_http_transport_for_non_openai_base_url() -> None:
    env = {
        "api_key": "test-key",
        "base_url": "https://xidaoapi.com/v1",
        "image_model": "gpt-image-2",
        "image_transport": "auto",
        "timeout": "180",
    }

    assert should_use_http_transport(env) is True


def test_generate_image_with_http_returns_namespace(monkeypatch: pytest.MonkeyPatch) -> None:
    response_body = {
        "data": [
            {
                "b64_json": base64.b64encode(b"png-bytes").decode("ascii"),
            }
        ]
    }

    class FakeResponse:
        status_code = 200
        text = json.dumps(response_body)

        def json(self) -> dict[str, object]:
            return response_body

    captured: dict[str, object] = {}

    def fake_post(url: str, **kwargs: object) -> FakeResponse:
        captured["url"] = url
        captured["headers"] = kwargs["headers"]
        captured["json"] = kwargs["json"]
        captured["timeout"] = kwargs["timeout"]
        return FakeResponse()

    monkeypatch.setattr("oepnai_image.cli.httpx.post", fake_post)

    result = generate_image_with_http(
        {
            "api_key": "test-key",
            "base_url": "https://xidaoapi.com/v1",
            "image_model": "gpt-image-2",
            "image_transport": "http",
            "timeout": "180",
        },
        {"model": "gpt-image-2", "prompt": "test"},
        timeout_seconds=3,
    )

    assert result.data[0].b64_json == response_body["data"][0]["b64_json"]
    assert captured["url"] == "https://xidaoapi.com/v1/images/generations"
    assert captured["json"] == {"model": "gpt-image-2", "prompt": "test"}
    assert captured["timeout"] == 3
    assert captured["headers"] == {
        "Authorization": "Bearer test-key",
        "Content-Type": "application/json",
    }


def test_generate_image_with_http_preserves_http_status(monkeypatch: pytest.MonkeyPatch) -> None:
    response_body = {
        "error": {
            "message": "Image generation is not enabled for this group",
        }
    }

    class FakeResponse:
        status_code = 403
        text = json.dumps(response_body)

        def json(self) -> dict[str, object]:
            return response_body

    monkeypatch.setattr("oepnai_image.cli.httpx.post", lambda *args, **kwargs: FakeResponse())

    with pytest.raises(RuntimeError, match="Image generation is not enabled for this group") as excinfo:
        generate_image_with_http(
            {
                "api_key": "test-key",
                "base_url": "https://xidaoapi.com/v1",
                "image_model": "gpt-image-2",
                "image_transport": "http",
                "timeout": "180",
            },
            {"model": "gpt-image-2", "prompt": "test"},
            timeout_seconds=3,
        )

    assert getattr(excinfo.value, "status_code", None) == 403


def test_generate_with_retries_does_not_retry_non_retryable_status() -> None:
    calls = {"count": 0}

    def fake_generate(**_: object) -> dict[str, str]:
        calls["count"] += 1
        exc = RuntimeError("permission denied")
        exc.status_code = 403  # type: ignore[attr-defined]
        raise exc

    client = SimpleNamespace(images=SimpleNamespace(generate=fake_generate))

    with pytest.raises(RuntimeError, match="permission denied"):
        generate_with_retries(
            client,
            {"prompt": "test"},
            job_slug="job-403",
            max_retries=5,
            retry_delay_seconds=0,
        )

    assert calls["count"] == 1


def test_format_terminal_api_error_for_group_permission() -> None:
    exc = RuntimeError(
        "Error code: 403 - {'error': {'message': "
        "'Image generation is not enabled for this group', 'type': 'permission_error'}}"
    )
    exc.status_code = 403  # type: ignore[attr-defined]

    assert "provider group/key" in format_terminal_api_error(exc)


def test_response_item_to_png_bytes_decodes_b64_without_urlopen(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"count": 0}

    def fake_urlopen(*_: object, **__: object) -> None:
        calls["count"] += 1
        raise AssertionError("urlopen should not be used for b64 responses")

    monkeypatch.setattr("oepnai_image.cli.urllib.request.urlopen", fake_urlopen)
    item = SimpleNamespace(b64_json=base64.b64encode(b"png-bytes").decode("ascii"))

    assert response_item_to_png_bytes(item, timeout_seconds=3) == b"png-bytes"
    assert calls["count"] == 0


def test_response_item_to_png_bytes_passes_timeout_to_urlopen(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def read(self) -> bytes:
            return b"downloaded"

    def fake_urlopen(url: str, *, timeout: float) -> FakeResponse:
        captured["url"] = url
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("oepnai_image.cli.urllib.request.urlopen", fake_urlopen)
    item = SimpleNamespace(url="https://example.test/image.png")

    assert response_item_to_png_bytes(item, timeout_seconds=7) == b"downloaded"
    assert captured == {"url": "https://example.test/image.png", "timeout": 7}


def test_resolve_timeout_seconds_uses_default_for_empty_value() -> None:
    assert resolve_timeout_seconds("") == 1200.0


def test_resolve_timeout_seconds_rejects_non_positive_value() -> None:
    with pytest.raises(RuntimeError, match="greater than 0"):
        resolve_timeout_seconds("0")


def test_resolve_timeout_seconds_rejects_invalid_number() -> None:
    with pytest.raises(RuntimeError, match="number of seconds"):
        resolve_timeout_seconds("abc")


def test_build_client_requires_api_key_with_local_env_hint() -> None:
    with pytest.raises(RuntimeError, match="local \\.env file"):
        build_client(
            {
                "api_key": "",
                "base_url": "https://api.openai.com/v1",
                "image_model": "gpt-image-2",
                "image_transport": "sdk",
                "timeout": "180",
            }
        )


def test_load_styles_reads_json_definitions(tmp_path: Path) -> None:
    style_path = tmp_path / "paper-figure.json"
    style_path.write_text(
        json.dumps(
            {
                "slug": "paper-figure",
                "name": "Paper Figure",
                "description": "Academic figure style.",
                "template": "Draw this: {prompt}",
                "defaults": {"quality": "high"},
            }
        ),
        encoding="utf-8",
    )

    styles = load_styles(tmp_path)

    assert list(styles) == ["paper-figure"]
    assert styles["paper-figure"].name == "Paper Figure"
    assert styles["paper-figure"].defaults["quality"] == "high"


def test_resolve_style_returns_style_by_slug(tmp_path: Path) -> None:
    style_path = tmp_path / "paper-figure.json"
    style_path.write_text(
        json.dumps(
            {
                "slug": "paper-figure",
                "template": "Draw this: {prompt}",
            }
        ),
        encoding="utf-8",
    )

    style = resolve_style("paper figure", load_styles(tmp_path))

    assert style is not None
    assert style.slug == "paper-figure"


def test_compose_prompt_applies_template_placeholder(tmp_path: Path) -> None:
    style_path = tmp_path / "paper-figure.json"
    style_path.write_text(
        json.dumps(
            {
                "slug": "paper-figure",
                "template": "Academic figure about: {prompt}",
            }
        ),
        encoding="utf-8",
    )

    style = resolve_style("paper-figure", load_styles(tmp_path))

    assert style is not None
    assert compose_prompt("microphone array beamforming", style) == (
        "Academic figure about: microphone array beamforming"
    )


def test_format_style_listing_includes_description(tmp_path: Path) -> None:
    style_path = tmp_path / "paper-figure.json"
    style_path.write_text(
        json.dumps(
            {
                "slug": "paper-figure",
                "description": "Academic figure style.",
                "template": "Draw this: {prompt}",
            }
        ),
        encoding="utf-8",
    )

    output = format_style_listing(load_styles(tmp_path))

    assert "paper-figure" in output
    assert "Academic figure style." in output


def test_generate_job_uses_style_model_size_and_quality_defaults(tmp_path: Path) -> None:
    style_path = tmp_path / "paper-figure.json"
    style_path.write_text(
        json.dumps(
            {
                "slug": "paper-figure",
                "template": "Academic figure about: {prompt}",
                "defaults": {
                    "category": "paper-figures",
                    "model": "gpt-image-2",
                    "size": "1536x864",
                    "quality": "medium",
                    "background": "opaque",
                },
            }
        ),
        encoding="utf-8",
    )
    style = resolve_style("paper-figure", load_styles(tmp_path))
    assert style is not None

    job = make_prompt_jobs(
        ["AudioScan model diagram"],
        model=None,
        size=None,
        quality=None,
        background=None,
        num_images=1,
        category="misc",
        filename_prefix=None,
        style="paper-figure",
    )[0]

    options = GenerationOptions(
        size_override=None,
        quality_override=None,
        model_override=None,
        background_override=None,
        flat_output=False,
        max_retries=1,
        retry_delay_seconds=0,
        timeout_seconds=180,
        dry_run=True,
    )
    result = generate_job(
        client=None,
        job=job,
        output_dir=tmp_path / "out",
        env={"image_model": "gpt-image-2"},
        style=style,
        options=options,
    )

    assert result["category"] == "paper-figures"
    assert result["payload"]["model"] == "gpt-image-2"
    assert result["payload"]["size"] == "1536x864"
    assert result["payload"]["quality"] == "medium"
    assert not (tmp_path / "out").exists()


def test_generate_job_writes_api_png_without_reencoding(tmp_path: Path) -> None:
    source = BytesIO()
    Image.new("RGB", (16, 9), "white").save(source, format="PNG")
    original_bytes = source.getvalue()
    response = SimpleNamespace(
        data=[
            SimpleNamespace(
                b64_json=base64.b64encode(original_bytes).decode("ascii"),
            )
        ]
    )
    client = SimpleNamespace(images=SimpleNamespace(generate=lambda **_: response))
    options = GenerationOptions(
        size_override="160x96",
        quality_override=None,
        model_override=None,
        background_override=None,
        flat_output=True,
        max_retries=1,
        retry_delay_seconds=0,
        timeout_seconds=180,
        dry_run=False,
    )

    result = generate_job(
        client=client,
        job=ImageJob(slug="sample", prompt="sample prompt", filename_prefix="sample"),
        output_dir=tmp_path,
        env={"image_model": "gpt-image-2"},
        style=None,
        options=options,
    )

    output_path = tmp_path / "sample.png"
    assert output_path.read_bytes() == original_bytes
    assert result["images"][0]["requested_size"] == "160x96"
    assert result["images"][0]["original_size"] == "16x9"
    assert result["images"][0]["final_size"] == "16x9"
    assert result["images"][0]["resized"] is False


def test_generate_job_raises_on_empty_response_data(tmp_path: Path) -> None:
    response = SimpleNamespace(data=[])
    client = SimpleNamespace(images=SimpleNamespace(generate=lambda **_: response))
    options = GenerationOptions(
        size_override=None,
        quality_override=None,
        model_override=None,
        background_override=None,
        flat_output=True,
        max_retries=1,
        retry_delay_seconds=0,
        timeout_seconds=180,
        dry_run=False,
    )

    with pytest.raises(RuntimeError, match="returned no image data"):
        generate_job(
            client=client,
            job=ImageJob(slug="sample", prompt="sample prompt", filename_prefix="sample"),
            output_dir=tmp_path,
            env={"image_model": "gpt-image-2"},
            style=None,
            options=options,
        )


def test_inspect_png_bytes_does_not_resize_requested_resolution() -> None:
    source = BytesIO()
    Image.new("RGB", (16, 9), "white").save(source, format="PNG")

    metadata = inspect_png_bytes(source.getvalue())

    assert metadata == {
        "original_size": "16x9",
        "final_size": "16x9",
        "resized": False,
    }


def test_default_output_root_uses_current_working_directory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    assert default_output_root() == tmp_path / "generated_images"


def test_run_cli_returns_clean_error_for_unknown_style(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = run_cli(["--prompt", "test", "--style", "does-not-exist", "--dry-run"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "[error] Unknown style" in captured.err
    assert "Traceback" not in captured.err


def test_main_list_styles_does_not_require_env(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_TIMEOUT_SECONDS", "not-a-number")

    exit_code = main(["--list-styles"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "paper-figure" in captured.out


def test_validate_main_args_rejects_negative_limit() -> None:
    args = parse_args(["--prompt", "test", "--limit", "-1"])

    with pytest.raises(RuntimeError, match="--limit must be at least 1"):
        from oepnai_image.cli import validate_main_args

        validate_main_args(args)


def test_generate_with_retries_does_not_retry_http_permission_error(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"count": 0}

    def fake_generate(**_: object) -> dict[str, str]:
        calls["count"] += 1
        exc = RuntimeError("Image generation is not enabled for this group")
        exc.status_code = 403  # type: ignore[attr-defined]
        raise exc

    client = SimpleNamespace(images=SimpleNamespace(generate=fake_generate))
    monkeypatch.setattr("oepnai_image.cli.time.sleep", lambda _: None)

    with pytest.raises(RuntimeError, match="provider group/key"):
        generate_with_retries(
            client,
            {"prompt": "test"},
            job_slug="job-http-403",
            max_retries=5,
            retry_delay_seconds=0,
        )

    assert calls["count"] == 1


def test_workflow_request_dry_run_returns_manifest_without_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    manifest = run_generation_request(
        GenerationRequest(
            prompts=["a clean diagram of an MCP image server"],
            style="paper-figure",
            dry_run=True,
            aspect_ratio="16:9",
        )
    )

    assert manifest["dry_run"] is True
    assert manifest["jobs"][0]["payload"]["size"] == "1536x864"
    assert manifest["jobs"][0]["style"] == "paper-figure"
    assert not (tmp_path / "generated_images").exists()


def test_mcp_server_tools_are_importable() -> None:
    from oepnai_image import mcp_server

    assert mcp_server.mcp.name == "openai-image-mcp"
    assert callable(mcp_server.generate_image)
    assert callable(mcp_server.generate_images_batch)
    assert callable(mcp_server.plan_image_generation)
    assert callable(mcp_server.list_prompt_styles)


def test_mcp_server_resolves_image_count_aliases() -> None:
    from oepnai_image import mcp_server

    assert mcp_server._resolve_num_images() == 1
    assert mcp_server._resolve_num_images(3) == 3
    assert mcp_server._resolve_num_images(1, n=5) == 5
    assert mcp_server._resolve_num_images(1, count=4) == 4
    assert mcp_server._resolve_num_images(1, image_count=2) == 2


def test_mcp_server_rejects_invalid_image_count_aliases() -> None:
    from oepnai_image import mcp_server

    with pytest.raises(RuntimeError, match="n must be at least 1"):
        mcp_server._resolve_num_images(n=0)


def test_mcp_server_enriches_public_images_for_librechat(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from oepnai_image import mcp_server

    public_root = tmp_path / "public" / "images"
    image_path = public_root / "openai-image-mcp" / "travel" / "china-travel-map.png"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"png")

    monkeypatch.setenv("OPENAI_IMAGE_PUBLIC_IMAGES_ROOT", str(public_root))
    monkeypatch.setenv("OPENAI_IMAGE_PUBLIC_URL_PREFIX", "/images")

    manifest = mcp_server._enrich_for_librechat(
        {
            "jobs": [
                {
                    "slug": "china-travel-map",
                    "base_prompt": "China travel map",
                    "images": [{"file": str(image_path)}],
                }
            ]
        }
    )

    image = manifest["jobs"][0]["images"][0]
    assert image["title"] == "China Travel Map"
    assert image["url"] == "/images/openai-image-mcp/travel/china-travel-map.png"
    assert "![China Travel Map](/images/openai-image-mcp/travel/china-travel-map.png)" in manifest["text"]

    tool_result = mcp_server._to_tool_result(manifest)
    assert tool_result.structuredContent == manifest
    assert tool_result.content[0].type == "text"
    assert "![China Travel Map](/images/openai-image-mcp/travel/china-travel-map.png)" in tool_result.content[0].text
    assert tool_result.content[1].type == "image"
    assert tool_result.content[1].data == base64.b64encode(b"png").decode("ascii")
    assert tool_result.content[1].mimeType == "image/png"


def test_mcp_server_skips_inline_image_when_file_is_too_large(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from oepnai_image import mcp_server

    public_root = tmp_path / "public" / "images"
    image_path = public_root / "openai-image-mcp" / "travel" / "large-map.png"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"png-data")

    monkeypatch.setenv("OPENAI_IMAGE_PUBLIC_IMAGES_ROOT", str(public_root))
    monkeypatch.setenv("OPENAI_IMAGE_PUBLIC_URL_PREFIX", "/images")
    monkeypatch.setenv("OPENAI_IMAGE_EMBED_MAX_BYTES", "1")

    manifest = mcp_server._enrich_for_librechat(
        {
            "jobs": [
                {
                    "slug": "large-map",
                    "base_prompt": "Large map",
                    "images": [{"file": str(image_path)}],
                }
            ]
        }
    )
    tool_result = mcp_server._to_tool_result(manifest)

    assert len(tool_result.content) == 1
    assert tool_result.content[0].type == "text"
    assert "![Large Map](/images/openai-image-mcp/travel/large-map.png)" in tool_result.content[0].text
