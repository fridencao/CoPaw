# -*- coding: utf-8 -*-
"""Command dispatch: run command path without creating CoPawAgent.

Yields messages compatible with LangGraph query_handler stream.
"""
from __future__ import annotations

import logging
from typing import AsyncIterator, Any, List

from langchain_core.messages import AIMessage, HumanMessage

from . import control_commands
from .daemon_commands import (
    DaemonContext,
    DaemonCommandHandlerMixin,
    parse_daemon_query,
)
from ...agents.command_handler import CommandHandler
from ...config.config import load_agent_config

logger = logging.getLogger(__name__)


def _get_last_user_text(msgs: List[Any]) -> str | None:
    """Extract last user message text from msgs (LangGraph message list)."""
    if not msgs or len(msgs) == 0:
        return None
    last = msgs[-1]

    # LangGraph messages
    if hasattr(last, "content"):
        return last.content

    # Dict format
    if isinstance(last, dict):
        content = last.get("content") or last.get("text")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    return block.get("text")
    return None


def _is_conversation_command(query: str | None) -> bool:
    """True if query is a conversation command (/compact, /new, etc.)."""
    if not query or not query.startswith("/"):
        return False

    conversation_commands = {
        "/compact",
        "/new",
        "/clear",
        "/forget",
    }
    return query.strip().lower() in conversation_commands


async def run_command_path(
    runner: Any,
    query: str,
    daemon_context: DaemonContext | None = None,
) -> AsyncIterator[tuple[AIMessage, bool]]:
    """Run a command path and yield messages.

    Args:
        runner: Agent runner
        query: Command query
        daemon_context: Optional daemon context

    Yields:
        Tuples of (message, is_last)
    """
    from ...agents.command_handler import CommandHandler

    # Load config and create command handler
    config = load_agent_config(runner.agent_id)
    command_handler = CommandHandler(config)

    # Process command
    async for msg in command_handler.handle_command(
        query=query,
        runner=runner,
        daemon_context=daemon_context,
    ):
        yield (AIMessage(content=str(msg)), False)

    yield (AIMessage(content=""), True)


def _is_command(query: str | None) -> bool:
    """True if query is a command (starts with / or is daemon query)."""
    if not query:
        return False
    # Regular command
    if query.startswith("/"):
        return True
    # Daemon query
    return parse_daemon_query(query) is not None


# Import control_commands after defining helpers
from . import control_commands