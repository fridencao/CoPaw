# -*- coding: utf-8 -*-
"""Runner factory for CoPaw.

This module provides a factory for creating the appropriate Runner
based on configuration. It supports switching between the original
AgentScope-based runner and the new LangGraph-based runner.
"""

import os
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Configuration key for switching runners
LANGGRAPH_ENABLED = os.getenv("COPAW_USE_LANGGRAPH", "false").lower() == "true"


def get_runner_class(use_langgraph: bool = LANGGRAPH_ENABLED):
    """Get the appropriate Runner class based on configuration.

    Args:
        use_langgraph: Whether to use LangGraph runner (default from env)

    Returns:
        Runner class (either AgentRunner or LangGraphRunnerAdapter)
    """
    if use_langgraph:
        logger.info("Using LangGraph-based runner")
        from ..langgraph.adapter import LangGraphRunnerAdapter

        return LangGraphRunnerAdapter
    else:
        logger.info("Using AgentScope-based runner")
        from ...app.runner.runner import AgentRunner

        return AgentRunner


def create_runner(
    agent_id: str = "default",
    workspace_dir: Path | None = None,
    task_tracker: Any | None = None,
    use_langgraph: Optional[bool] = None,
):
    """Factory function to create a Runner instance.

    Args:
        agent_id: Agent configuration ID
        workspace_dir: Workspace directory
        task_tracker: Optional task tracker
        use_langgraph: Override for LangGraph usage (default from env)

    Returns:
        Runner instance (AgentRunner or LangGraphRunnerAdapter)
    """
    if use_langgraph is None:
        use_langgraph = LANGGRAPH_ENABLED

    runner_class = get_runner_class(use_langgraph)

    return runner_class(
        agent_id=agent_id,
        workspace_dir=workspace_dir,
        task_tracker=task_tracker,
    )


def is_langgraph_enabled() -> bool:
    """Check if LangGraph runner is enabled.

    Returns:
        True if LangGraph runner is enabled
    """
    return LANGGRAPH_ENABLED


def enable_langgraph() -> None:
    """Enable LangGraph runner for subsequent runner creations."""
    global LANGGRAPH_ENABLED
    LANGGRAPH_ENABLED = True
    logger.info("LangGraph runner enabled")


def disable_langgraph() -> None:
    """Disable LangGraph runner (use AgentScope instead)."""
    global LANGGRAPH_ENABLED
    LANGGRAPH_ENABLED = False
    logger.info("LangGraph runner disabled, using AgentScope")


__all__ = [
    "get_runner_class",
    "create_runner",
    "is_langgraph_enabled",
    "enable_langgraph",
    "disable_langgraph",
    "LANGGRAPH_ENABLED",
]