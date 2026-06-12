"""Wiring for the official Alpaca MCP server via the Claude Agent SDK.

The SDK namespaces MCP tools as ``mcp__<server>__<tool>``. We expose only the
subset of Alpaca tools the strategy needs (read data + place/close orders) via
``allowed_tools`` so the model cannot reach for anything else.

Risk enforcement is attached here as a ``PreToolUse`` hook (see ``risk.hook``):
every order tool call passes through an in-process gate that can deny it before
it reaches Alpaca.
"""

from __future__ import annotations

from claude_agent_sdk import ClaudeAgentOptions, HookMatcher

from .config import AppConfig

MCP_SERVER_NAME = "alpaca"

# Bare Alpaca tool names we rely on (without the SDK's mcp__<server>__ prefix).
DATA_TOOLS = [
    "get_account_info",
    "get_all_positions",
    "get_open_position",
    "get_stock_bars",
]
ORDER_TOOLS = [
    "place_stock_order",
    "place_crypto_order",
    "close_position",
    "close_all_positions",
    "cancel_all_orders",
]


def tool_name(bare: str) -> str:
    """Return the SDK-namespaced tool name for an Alpaca tool."""
    return f"mcp__{MCP_SERVER_NAME}__{bare}"


# Fully-qualified names used for allow-listing and hook matching.
ALLOWED_TOOLS = [tool_name(t) for t in DATA_TOOLS + ORDER_TOOLS]
ORDER_TOOL_NAMES = {tool_name(t) for t in ORDER_TOOLS}


def _mcp_server_config(cfg: AppConfig) -> dict:
    """Build the SDK mcp_servers entry for the chosen transport."""
    s = cfg.settings
    if s.alpaca_mcp_transport == "http":
        return {"type": "http", "url": s.alpaca_mcp_url}

    # stdio: SDK spawns the server as a subprocess and pipes Alpaca creds in.
    return {
        "type": "stdio",
        "command": s.alpaca_mcp_command,
        "args": [a for a in s.alpaca_mcp_args.split(",") if a],
        "env": {
            "ALPACA_API_KEY": s.alpaca_api_key,
            "ALPACA_SECRET_KEY": s.alpaca_secret_key,
            "ALPACA_PAPER_TRADE": "true" if s.alpaca_paper_trade else "false",
        },
    }


def build_options(
    cfg: AppConfig,
    system_prompt: str,
    risk_hook,
) -> ClaudeAgentOptions:
    """Assemble ClaudeAgentOptions for one ReAct tick.

    ``risk_hook`` is the async PreToolUse callback from ``risk.hook.make_risk_hook``.
    """
    return ClaudeAgentOptions(
        model=cfg.settings.trader_model,
        system_prompt=system_prompt,
        mcp_servers={MCP_SERVER_NAME: _mcp_server_config(cfg)},
        allowed_tools=ALLOWED_TOOLS,
        # The risk hook matches order tools by name (matcher is a regex on tool name).
        hooks={
            "PreToolUse": [
                HookMatcher(
                    matcher=f"mcp__{MCP_SERVER_NAME}__place_.*",
                    hooks=[risk_hook],
                ),
                HookMatcher(
                    matcher=f"mcp__{MCP_SERVER_NAME}__close_.*",
                    hooks=[risk_hook],
                ),
            ],
        },
    )
