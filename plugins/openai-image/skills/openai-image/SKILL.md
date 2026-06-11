---
name: openai-image
description: Use when the user wants to generate, plan, batch, or inspect AI images through the OpenAI Image MCP plugin or local openai-image CLI, especially from Codex, Claude Code, or IDE MCP clients.
---

# OpenAI Image Workflow

Use this skill when image generation should run through this plugin instead of hand-written API calls.

## Default Path

1. For non-trivial prompts, call `plan_image_generation` first with `dry_run` behavior through the MCP tool. Check model, size, style defaults, and output path shape.
2. Call `generate_image` for one prompt or `generate_images_batch` for batch JSON files.
3. Return the saved image paths and `manifest_path` from the tool result.
4. Do not expose `OPENAI_API_KEY` or local `.env` values.

## MCP Tools

- `list_prompt_styles`: list bundled and custom styles.
- `plan_image_generation`: resolve payload and manifest shape without calling the API.
- `generate_image`: generate one prompt and return a manifest with saved PNG paths.
- `generate_images_batch`: run a batch JSON file, optionally with `only`, `limit`, `workers`, or `dry_run`.

## Practical Defaults

- Use `style: "paper-figure"` for clean academic diagrams and paper figures.
- Use `aspect_ratio` with `long_edge` for flexible layouts, or exact `size` when the user gives dimensions.
- Keep dimensions divisible by 16.
- Use `output_dir` when the user needs files in a project-specific folder.
- Use `dry_run: true` for batch validation before spending API calls.

## CLI Fallback

If MCP tools are unavailable, run the CLI from the plugin root:

```bash
uv run openai-image --prompt "a concise image prompt" --dry-run
uv run openai-image --prompt "a concise image prompt" --output-dir ./generated_images/manual
uv run openai-image --batch batches/jobs.json --workers 4
```

## Failure Handling

- Missing key: ask the user to set `OPENAI_API_KEY` in the environment or `.env`.
- Provider permission errors such as image generation not enabled are account/key issues, not prompt issues.
- MCP stdio clients require server logs on stderr only; do not add stdout progress output to MCP server code.
