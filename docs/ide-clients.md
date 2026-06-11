# IDE And MCP Clients

This plugin exposes a stdio MCP server, so it can be used by Codex, Claude Code, and IDEs that support MCP server configuration.

## Tools

| Tool | Purpose |
| --- | --- |
| `list_prompt_styles` | Inspect bundled or custom prompt styles. |
| `plan_image_generation` | Dry-run one prompt and inspect the final payload. |
| `generate_image` | Generate one prompt and return saved file paths. |
| `generate_images_batch` | Run batch JSON jobs with optional filters. |

## Claude Code

Add an MCP server that runs:

```bash
uv run openai-image-mcp
```

Set the working directory to the cloned repository and pass `OPENAI_API_KEY` through the environment.

## VS Code Or Other IDEs

Use the same stdio command:

```json
{
  "command": "uv",
  "args": ["run", "openai-image-mcp"],
  "cwd": "/absolute/path/to/openai-image"
}
```

The MCP server returns JSON manifests with relative or absolute saved file paths. The generated PNG files remain on disk and are not streamed through the MCP response.

## Operational Notes

- Use `plan_image_generation` before expensive or ambiguous prompts.
- Keep generated files out of git unless the project explicitly wants assets committed.
- Use `OPENAI_IMAGE_TRANSPORT=curl` for OpenAI-compatible providers whose SDK behavior differs from the official OpenAI API.
