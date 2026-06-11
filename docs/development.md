# Development

## Layout

```text
.codex-plugin/plugin.json      Codex plugin manifest
.mcp.json                      MCP server configuration
skills/openai-image/SKILL.md   Codex workflow guidance
src/oepnai_image/cli.py        Existing CLI and core generation primitives
src/oepnai_image/workflow.py   Reusable workflow used by CLI and MCP
src/oepnai_image/mcp_server.py MCP stdio server
tests/                         Unit and MCP smoke tests
```

The import package path remains `oepnai_image` for backward compatibility with the existing CLI code. The distribution name and user-facing commands are `openai-image`.

## Validation

```bash
uv sync --extra dev
uv run pytest -q
uv run python -m compileall -q src
python3 path/to/plugin-creator/scripts/validate_plugin.py .
```

## Release Checklist

- Do not commit `.env`, generated images, batch outputs, or cache directories.
- Run the validation commands above.
- Verify `uv run openai-image --prompt "test" --dry-run` prints a manifest.
- Verify `uv run python -m oepnai_image.mcp_server` imports and starts under an MCP smoke test.
- Reinstall the Codex plugin and open a new thread.
