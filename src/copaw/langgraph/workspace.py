# -*- coding: utf-8 -*-
"""LangGraph workspace for CoPaw.

This module provides a workspace wrapper that uses LangGraph runner
instead of the AgentScope-based workspace.
"""

import logging
from pathlib import Path
from typing import Any, Optional

from .runner import LangGraphRunner
from .adapter import LangGraphRunnerAdapter
from .tools import tool_registry

logger = logging.getLogger(__name__)


class LangGraphWorkspace:
    """Workspace wrapper for LangGraph runner.

    This class provides a compatibility layer between the existing
    workspace interface and the new LangGraph runner.
    """

    def __init__(
        self,
        agent_id: str,
        workspace_dir: Path,
        mcp_client_manager: Optional[Any] = None,
        memory_manager: Optional[Any] = None,
    ):
        """Initialize the LangGraph workspace.

        Args:
            agent_id: Agent configuration ID
            workspace_dir: Workspace directory
            mcp_client_manager: Optional MCP client manager
            memory_manager: Optional memory manager
        """
        self.agent_id = agent_id
        self.workspace_dir = workspace_dir
        self._mcp_client_manager = mcp_client_manager
        self._memory_manager = memory_manager

        # Create runner adapter
        self._runner_adapter = LangGraphRunnerAdapter(
            agent_id=agent_id,
            workspace_dir=workspace_dir,
            mcp_client_manager=mcp_client_manager,
            memory_manager=memory_manager,
        )

        logger.info(
            "LangGraphWorkspace initialized for agent '%s'",
            agent_id,
        )

    @property
    def runner(self) -> LangGraphRunnerAdapter:
        """Get the runner adapter."""
        return self._runner_adapter

    @property
    def framework_type(self) -> str:
        """Get the framework type."""
        return "langgraph"

    async def start(self) -> None:
        """Start the workspace."""
        logger.info(f"Starting LangGraph workspace for agent: {self.agent_id}")

    async def stop(self) -> None:
        """Stop the workspace."""
        logger.info(f"Stopping LangGraph workspace for agent: {self.agent_id}")

    async def close(self) -> None:
        """Close the workspace and cleanup resources."""
        await self.stop()


def create_langgraph_workspace(
    agent_id: str,
    workspace_dir: Path,
    mcp_client_manager: Optional[Any] = None,
    memory_manager: Optional[Any] = None,
) -> LangGraphWorkspace:
    """Factory function to create a LangGraph workspace.

    Args:
        agent_id: Agent configuration ID
        workspace_dir: Workspace directory
        mcp_client_manager: Optional MCP client manager
        memory_manager: Optional memory manager

    Returns:
        LangGraphWorkspace instance
    """
    return LangGraphWorkspace(
        agent_id=agent_id,
        workspace_dir=workspace_dir,
        mcp_client_manager=mcp_client_manager,
        memory_manager=memory_manager,
    )


__all__ = [
    "LangGraphWorkspace",
    "create_langgraph_workspace",
]