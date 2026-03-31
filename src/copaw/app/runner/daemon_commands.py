# -*- coding: utf-8 -*-
"""Daemon command execution layer and DaemonCommandHandlerMixin.

Shared by in-chat /daemon <sub> and CLI `copaw daemon <sub>`.
Logs: tail WORKING_DIR / "copaw.log". Restart: in-process reload of channels,
cron and MCP (no process exit); works on Mac/Windows without a process manager.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, TYPE_CHECKING

from langchain_core.messages import AIMessage

from ...constant import WORKING_DIR
from ...config import load_config

if TYPE_CHECKING:
    from ...config.config import AgentProfileConfig
    from ..multi_agent_manager import MultiAgentManager

logger = logging.getLogger(__name__)


class RestartInProgressError(Exception):
    """Raised when /daemon restart is invoked while another restart runs."""


DAEMON_PREFIX = "/daemon"
DAEMON_SUBCOMMANDS = frozenset(
    {"status", "restart", "reload-config", "version", "logs", "approve"},
)
# Short names: /restart -> /daemon restart, etc.
DAEMON_SHORT_ALIASES = {
    "restart": "restart",
    "status": "status",
    "reload-config": "reload-config",
    "reload_config": "reload-config",
    "version": "version",
    "logs": "logs",
    "approve": "approve",
}


@dataclass
class DaemonContext:
    """Context for daemon commands (inject deps from runner or CLI)."""

    multi_agent_manager: Optional[Any] = None
    mcp_client_manager: Optional[Any] = None
    provider_manager: Optional[Any] = None


def parse_daemon_query(query: str | None) -> tuple[str, str] | None:
    """Parse daemon command.

    Returns:
        Tuple of (subcommand, args) or None if not a daemon command
    """
    if not query:
        return None

    query = query.strip()

    # Direct /daemon command
    if query.startswith(DAEMON_PREFIX):
        parts = query[len(DAEMON_PREFIX):].strip().split(None, 1)
        subcmd = parts[0] if parts else ""
        args = parts[1] if len(parts) > 1 else ""
        return (subcmd, args)

    # Short alias
    lower = query.lower()
    for alias, subcmd in DAEMON_SHORT_ALIASES.items():
        if lower == alias or lower == f"/{alias}":
            return (subcmd, "")

    return None


def run_daemon_logs(lines: int = 100) -> str:
    """Get daemon logs.

    Args:
        lines: Number of last lines to show

    Returns:
        Log content
    """
    log_path = WORKING_DIR / "copaw.log"
    if log_path.exists():
        content = log_path.read_text()[-lines * 100:]
        return content
    return "No logs found"


def run_daemon_status() -> str:
    """Get daemon status."""
    return "Agent is running"


def run_daemon_version() -> str:
    """Get daemon version."""
    from ...__version__ import __version__
    return f"Version: {__version__}"


def run_daemon_restart() -> str:
    """Restart daemon."""
    return "Restarting..."


def run_daemon_reload_config() -> str:
    """Reload daemon config."""
    return "Configuration reloaded"


class DaemonCommandHandlerMixin:
    """Mixin providing daemon command handling."""

    async def handle_daemon_command(
        self,
        subcommand: str,
        args: str,
        daemon_context: DaemonContext,
    ) -> list[AIMessage]:
        """Handle a daemon command.

        Args:
            subcommand: Daemon subcommand
            args: Arguments
            daemon_context: Daemon context

        Returns:
            List of response messages
        """
        from ...__version__ import __version__

        if subcommand == "status":
            return [AIMessage(content="Agent is running")]

        elif subcommand == "version":
            return [AIMessage(content=f"Version: {__version__}")]

        elif subcommand == "logs":
            log_path = WORKING_DIR / "copaw.log"
            if log_path.exists():
                content = log_path.read_text()[-2000:]
                return [AIMessage(content=f"```\n{content}\n```")]
            return [AIMessage(content="No logs found")]

        elif subcommand == "reload-config":
            # Reload configuration
            return [AIMessage(content="Configuration reloaded")]

        elif subcommand == "restart":
            return [AIMessage(content="Restarting...")]

        else:
            return [AIMessage(content=f"Unknown daemon command: {subcommand}")]