# OpenAI Image MCP

A Codex plugin + MCP server + CLI for generating images through OpenAI-compatible APIs.

## Features

- **9 bundled prompt styles** (BioRender-inspired anatomical, pathway, cellular, infographic + academic paper figures, signal flow, data viz, concept maps, engineering schematics)
- **Batch generation** with parallel workers
- **Dry-run planning** before spending API calls
- **Structured manifests** with saved PNG paths
- **OpenAI-compatible base URLs** for alternative providers
- **Works as**: Codex plugin, stdio MCP server, or standalone CLI

## Quick Start

```bash
git clone https://github.com/911218sky/openai-image-mcp.git
cd openai-image-mcp
uv sync --extra dev
cp .env.example .env
# Edit .env with your OPENAI_API_KEY
uv run openai-image --list-styles
```

## Project Structure

```
plugins/openai-image/       # The actual plugin
├── .codex-plugin/          # Plugin manifest
├── .mcp.json               # MCP server config
├── skills/openai-image/    # Codex skill (SKILL.md)
├── src/oepnai_image/       # Python source (CLI + MCP server)
├── prompt_styles/          # 9 bundled prompt style templates
└── pyproject.toml          # Package config
```

## Prompt Styles

| Style | Description | Default Size |
|-------|-------------|-------------|
| `paper-figure` | Clean academic schematic figures | 1536x864 |
| `biorender-anatomy` | Medical/anatomical cross-sections (BioRender style) | 1536x1024 |
| `biorender-pathway` | Molecular/biological pathways (BioRender style) | 1536x1024 |
| `biorender-cellular` | Cell biology, organelles, receptors (BioRender style) | 1024x1024 |
| `biorender-infographic` | Scientific infographics, graphical abstracts | 1536x1024 |
| `signal-flow` | Signal processing block diagrams (IEEE style) | 1536x1024 |
| `data-viz` | Charts, plots, data visualization (Nature style) | 1536x1024 |
| `concept-map` | Concept/relationship diagrams | 1024x1024 |
| `engineering-schematic` | Technical engineering schematics (ASME/IEEE) | 1536x1024 |

Usage:

```bash
uv run openai-image --style biorender-pathway --prompt "MAPK signaling cascade"
uv run openai-image --style paper-figure --prompt "ANC system block diagram"
```

## Codex Plugin Install

```bash
codex plugin marketplace add .
codex plugin add openai-image@openai-image-mcp
```

Start a new Codex thread after installing.

## MCP Server

Generic stdio config for any MCP client:

```json
{
  "mcpServers": {
    "openai-image-mcp": {
      "command": "uv",
      "args": ["run", "openai-image-mcp"],
      "cwd": "/absolute/path/to/openai-image-mcp",
      "env": {
        "OPENAI_API_KEY": "your_key"
      }
    }
  }
}
```

## CLI Examples

```bash
# Single image
uv run openai-image --prompt "a cinematic photo" --style paper-figure

# Plan without API call
uv run openai-image --prompt "beamforming diagram" --dry-run

# Batch with parallel workers
uv run openai-image --batch jobs.json --workers 4

# Custom dimensions
uv run openai-image --prompt "city skyline" --aspect-ratio 16:9 --long-edge 1920
```

## Batch Format

```json
{
  "defaults": {
    "model": "gpt-image-2",
    "size": "1024x1024",
    "background": "opaque"
  },
  "jobs": [
    {"slug": "fig-01", "prompt": "system overview", "style": "paper-figure"},
    {"slug": "fig-02", "prompt": "signal flow", "style": "signal-flow"}
  ]
}
```

## Configuration

| Env Var | Default | Description |
|---------|---------|-------------|
| `OPENAI_API_KEY` | (required) | API key |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | API endpoint |
| `OPENAI_IMAGE_MODEL` | `gpt-image-2` | Model name |
| `OPENAI_IMAGE_TRANSPORT` | `auto` | `auto`, `sdk`, or `http` |
| `OPENAI_TIMEOUT_SECONDS` | `1200` | Request timeout |

### Provider Registry

For multiple providers or targets, copy the checked-in example and point the
CLI/MCP process at the local copy:

```bash
mkdir -p ~/.config/openai-image-mcp
cp providers.toml.example ~/.config/openai-image-mcp/providers.toml
export OPENAI_IMAGE_CONFIG="$HOME/.config/openai-image-mcp/providers.toml"
export GEMINI_API_KEY="your_gemini_key"
```

The example includes a Gemini-native target for
`gemini-3.1-flash-image` and the official OpenAI image target. Select a target
explicitly when generating:

```bash
uv run openai-image \
  --prompt "a clean academic ear canal cross-section" \
  --provider gemini \
  --target inroi \
  --model gemini-3.1-flash-image
```

For gateways that expose image models through an OpenAI-compatible
`/chat/completions` endpoint, use `protocol = "openai-chat-images"`. The
gateway response must provide image bytes in `data[].b64_json`.

Use `.env.example` as the environment variable reference. Never commit a
local `.env`, provider registry containing secrets, or generated images.

## Development

```bash
uv sync --extra dev
uv run pytest -q
uv run python -m compileall -q plugins/openai-image/src
uv run openai-image --list-styles
```

## Security

- Never commit `.env` or log `OPENAI_API_KEY`.
- Generated images and batch outputs are gitignored.

## License

AGPL-3.0-only. See [LICENSE](LICENSE).
