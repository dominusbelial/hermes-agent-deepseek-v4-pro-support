"""Pi (Qwen3-Coder) sub-agent delegation plugin — bundled, auto-loaded.

Registers 3 tools (pi_delegate, pi_spawn, pi_join) into the ``delegation``
toolset. Each delegates coding work to a local Qwen3-Coder instance reached
via the llama.cpp proxy at 127.0.0.1:8001, using the ``pi`` coding agent
(@earendil-works/pi-coding-agent) in RPC mode over stdin/stdout.

Migrated from ``tools/pi_delegate_tool.py`` to a plugin per the Hermes
footprint ladder (AGENTS.md): local-only delegation capability belongs in
``plugins/``, not in core ``tools/``. The tool implementations live in
``plugins/pi_delegate/tools.py``; this module only wires registration.
"""

from __future__ import annotations

from .tools import (
    _DELEGATE_SCHEMA,
    _JOIN_SCHEMA,
    _SPAWN_SCHEMA,
    check_pi_requirements,
    pi_delegate,
    pi_join,
    pi_spawn,
)

_TOOLS = (
    ("pi_delegate", _DELEGATE_SCHEMA, pi_delegate, "🤖"),
    ("pi_spawn", _SPAWN_SCHEMA, pi_spawn, "🤖"),
    ("pi_join", _JOIN_SCHEMA, pi_join, "🤖"),
)


def register(ctx) -> None:
    """Register Pi delegation tools. Called once by the plugin loader."""
    for name, schema, handler, emoji in _TOOLS:
        ctx.register_tool(
            name=name,
            toolset="delegation",
            schema=schema,
            handler=handler,
            check_fn=check_pi_requirements,
            emoji=emoji,
        )
