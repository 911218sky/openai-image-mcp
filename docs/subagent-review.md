# Review Notes

This file records review areas used before publishing or refreshing the plugin.

## Review Areas

- Architecture: CLI and MCP share one workflow without duplicating generation behavior.
- Security: secrets stay in environment or `.env`, and generated artifacts remain ignored.
- MCP compatibility: stdio output is not polluted by human progress logs.
- Usability: Codex, Claude Code, and generic IDE clients have install guidance.

Detailed findings should be appended during each release review.
