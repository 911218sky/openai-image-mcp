# OpenAI Image MCP

OpenAI Image MCP is a local image-generation plugin that works as:

- a Codex plugin with a bundled skill
- a stdio MCP server for Codex, Claude Code, and IDE MCP clients
- a local `openai-image` CLI

It generates PNG files through OpenAI-compatible image APIs, writes a structured `manifest.json`, supports batch jobs, prompt styles, dry-run planning, custom dimensions, retries, and non-OpenAI compatible base URLs.

## Requirements

- Python 3.12+
- `uv`
- `OPENAI_API_KEY`

## Quick Start

```bash
git clone https://github.com/911218sky/openai-image-mcp.git
cd openai-image-mcp
uv sync --extra dev
cp .env.example .env
uv run openai-image --list-styles
```

Set `.env`:

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_IMAGE_MODEL=gpt-image-2
OPENAI_IMAGE_TRANSPORT=auto
OPENAI_TIMEOUT_SECONDS=1200
```

This project is managed with `uv`. Image requests use the OpenAI SDK for the official API and a Python HTTP transport for OpenAI-compatible providers; no `curl` binary is required.

## Codex Plugin Install

This repository includes a marketplace file, so Codex can install the plugin directly from the cloned repo:

```bash
codex plugin marketplace add .
codex plugin add openai-image@openai-image-mcp
```

Start a new Codex thread after installing so the bundled skill and MCP tools are loaded.

## MCP Tools

| Tool | Use |
| --- | --- |
| `list_prompt_styles` | List bundled and custom prompt styles. |
| `plan_image_generation` | Dry-run one prompt and inspect payload/manifest shape. |
| `generate_image` | Generate one prompt and return saved PNG paths. |
| `generate_images_batch` | Run a JSON batch with optional filters and workers. |

Generic MCP config:

```json
{
  "mcpServers": {
    "openai-image-mcp": {
      "command": "uv",
      "args": ["run", "openai-image-mcp"],
      "cwd": "/absolute/path/to/openai-image-mcp",
      "env": {
        "OPENAI_API_KEY": "your_api_key_here"
      }
    }
  }
}
```

For detailed install and IDE notes, see [Install](docs/install.md) and [IDE Clients](docs/ide-clients.md).

## CLI Examples

Single image:

```bash
uv run openai-image --prompt "a cinematic photo of a red sports car"
```

Plan without calling the API:

```bash
uv run openai-image \
  --prompt "a flat illustration of a robot barista" \
  --dry-run
```

Use a style:

```bash
uv run openai-image \
  --style paper-figure \
  --prompt "a diagram explaining microphone-array beamforming"
```

Run a batch:

```bash
uv run openai-image --batch jobs.json --workers 4
```

## Dimensions

Direct size:

```bash
uv run openai-image --prompt "city skyline" --size 1536x1024
```

Aspect ratio:

```bash
uv run openai-image --prompt "city skyline" --aspect-ratio 16:9 --long-edge 1920
```

Width and height must both be divisible by `16`.

## Batch Format

```json
{
  "defaults": {
    "category": "misc",
    "model": "gpt-image-2",
    "size": "1024x1024",
    "background": "opaque",
    "n": 1
  },
  "jobs": [
    {
      "slug": "poster-concept",
      "prompt": "a bold movie poster for a sci-fi thriller"
    }
  ]
}
```

## Output

Default output:

```text
generated_images/<timestamp>/
  manifest.json
  misc/
    prompt-01-example.png
```

Use `--flat-output` to skip category subfolders.

## Prompt Styles

Bundled style:

- `paper-figure`: clean academic schematic figures, default `1536x864`, `quality=medium`

Style files are JSON:

```json
{
  "slug": "paper-figure",
  "name": "Paper Figure",
  "description": "Clean academic schematic figures.",
  "template": "Create a clean academic figure about: {prompt}",
  "defaults": {
    "category": "paper-figures",
    "model": "gpt-image-2",
    "size": "1536x864",
    "background": "opaque",
    "quality": "medium"
  }
}
```

## Development

```bash
uv sync --extra dev
uv run pytest -q
uv run python -m compileall -q src
python3 path/to/plugin-creator/scripts/validate_plugin.py .
```

See [Development](docs/development.md).

## Security

- Do not commit `.env`.
- Do not print or commit `OPENAI_API_KEY`.
- Generated images, batches, local caches, and build outputs are ignored by git.
- Provider permission errors such as `Image generation is not enabled for this group` are account/key issues, not prompt issues.

## License

AGPL-3.0-only. See [LICENSE](LICENSE).
