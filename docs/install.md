# Install OpenAI Image MCP

## Requirements

- Python 3.12+
- `uv`
- Codex, Claude Code, or another MCP client
- `OPENAI_API_KEY` for an OpenAI-compatible image API

## Local Setup

```bash
git clone https://github.com/911218sky/openai-image-mcp.git
cd openai-image-mcp
uv sync --extra dev
cp .env.example .env
```

Edit `.env`:

```dotenv
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_IMAGE_MODEL=gpt-image-2
OPENAI_IMAGE_TRANSPORT=auto
OPENAI_TIMEOUT_SECONDS=1200
```

Validate:

```bash
uv run pytest -q
uv run python -m compileall -q src
uv run openai-image --list-styles
```

## Codex Plugin

This repository includes `.agents/plugins/marketplace.json`, so the simplest local install is:

```bash
codex plugin marketplace add .
codex plugin add openai-image@openai-image-mcp
```

Start a new Codex thread after installing or refreshing the plugin so the bundled skill and MCP server reload.

The marketplace entry in this repo is:

```json
{
  "name": "openai-image-mcp",
  "interface": {
    "displayName": "OpenAI Image MCP"
  },
  "plugins": [
    {
      "name": "openai-image",
      "source": {
        "source": "local",
        "path": "./plugins/openai-image"
      },
      "policy": {
        "installation": "AVAILABLE",
        "authentication": "ON_INSTALL"
      },
      "category": "Productivity"
    }
  ]
}
```

## Generic MCP Client

Use stdio:

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

Prefer setting `OPENAI_API_KEY` in the client environment instead of writing it into shared config.
