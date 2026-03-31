# -*- coding: utf-8 -*-
"""Utility functions for runner.

This module provides utility functions compatible with LangGraph.
"""
import json
import logging
import platform
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from ...config import load_config

logger = logging.getLogger(__name__)


def build_env_context(
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    channel: Optional[str] = None,
    working_dir: Optional[str] = None,
    add_hint: bool = True,
) -> str:
    """
    Build environment context with current request context prepended.

    Args:
        session_id: Current session ID
        user_id: Current user ID
        channel: Current channel name
        working_dir: Working directory path
        add_hint: Whether to add hint context
    Returns:
        Formatted environment context string
    """
    parts = []
    user_tz = load_config().user_timezone or "UTC"
    try:
        now = datetime.now(ZoneInfo(user_tz))
    except (ZoneInfoNotFoundError, KeyError):
        logger.warning("Invalid timezone %r, falling back to UTC", user_tz)
        now = datetime.now(timezone.utc)
        user_tz = "UTC"

    if session_id is not None:
        parts.append(f"- Session ID: {session_id}")
    if user_id is not None:
        parts.append(f"- User ID: {user_id}")
    if channel is not None:
        parts.append(f"- Channel: {channel}")

    parts.append(
        f"- OS: {platform.system()} {platform.release()} "
        f"({platform.machine()})",
    )

    if working_dir is not None:
        parts.append(f"- Working directory: {working_dir}")
    parts.append(
        f"- Current date: {now.strftime('%Y-%m-%d')} "
        f"{user_tz} ({now.strftime('%A')})",
    )

    if add_hint:
        parts.append(
            "- Important:\n"
            "  1. Prefer using skills when completing tasks "
            "(e.g. use the cron skill for scheduled tasks). "
            "Consult the relevant skill documentation if unsure.\n"
            "  2. When using write_file, if you want to avoid overwriting "
            "existing content, use read_file first to inspect the file, "
            "then use edit_file for partial updates or appending.\n"
            "  3. Use tool calls to perform actions. A response without a "
            "tool call indicates the task is complete. To continue a task, "
            "you must generate a tool call or provide useful feedback if "
            "you are blocked.\n",
        )

    return (
        "====================\n" + "\n".join(parts) + "\n===================="
    )


# LangGraph compatible message types
Message = Union[HumanMessage, AIMessage, BaseMessage]


def _is_local_file_url(url: str) -> bool:
    """True if url is a local file reference (file:// or absolute path)."""
    if not url or not isinstance(url, str):
        return False
    s = url.strip()
    if not s:
        return False
    lower = s.lower()

    # Check for remote URLs
    if lower.startswith(("http://", "https://", "data:")):
        return False

    # Check for local file patterns: file://, Unix paths, or Windows drives
    return (
        lower.startswith("file:")
        or (s.startswith("/") and not s.startswith("//"))
        or (len(s) >= 2 and s[1] == ":" and s[0].isalpha())
    )


def _abspath_from_url(url: str) -> str:
    """Extract absolute path from file:// URL."""
    s = url.strip()
    if s.lower().startswith("file:"):
        s = s[5:]
    s = "/" + s.lstrip("/")
    return s


def _resolve_content_url(url: str) -> str:
    """If url is local, return filename only; frontend builds URL."""
    if not isinstance(url, str):
        return url
    if not _is_local_file_url(url):
        return url
    return _abspath_from_url(url)


# Compatibility function - converts LangGraph messages
def langgraph_msg_to_message(
    messages: Union[BaseMessage, List[BaseMessage]],
) -> List[BaseMessage]:
    """Convert LangGraph messages.

    Args:
        messages: LangGraph message(s)

    Returns:
        List of messages
    """
    if isinstance(messages, BaseMessage):
        return [messages]
    elif isinstance(messages, list):
        return messages
    else:
        raise TypeError(f"Expected BaseMessage or list[BaseMessage], got {type(messages)}")


# Alias for backward compatibility
agentscope_msg_to_message = langgraph_msg_to_message