# -*- coding: utf-8 -*-
"""LangGraph Runner Factory for CoPaw.

This module provides a factory to create and manage LangGraph runners
for each agent, enabling integration with the existing FastAPI app.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from ..config.config import load_agent_config
from ..langgraph.adapter import LangGraphRunnerAdapter
from ..langgraph.adapters import (
    MCPClientManagerAdapter,
    LangGraphCheckpointAdapter,
    CoPawCheckpointer,
)
from ..langgraph.tools import tool_registry, register_builtin_tools

logger = logging.getLogger(__name__)


class LangGraphRunnerFactory:
    """Factory for creating and managing LangGraph runners.

    This factory:
    - Creates LangGraphRunnerAdapter for each agent
    - Manages MCP client integration
    - Handles memory/checkpoint integration
    - Provides hot-reload support for configuration changes
    """

    # Singleton instance
    _instance: Optional["LangGraphRunnerFactory"] = None
    _lock = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, workspace_dir: Optional[Path] = None):
        """Initialize the factory.

        Args:
            workspace_dir: Base workspace directory
        """
        if self._initialized:
            return

        self.workspace_dir = workspace_dir
        self._runners: Dict[str, LangGraphRunnerAdapter] = {}
        self._mcp_adapter: Optional[MCPClientManagerAdapter] = None
        self._checkpoint_adapter: Optional[LangGraphCheckpointAdapter] = None
        self._initialized = True

        logger.info("LangGraphRunnerFactory initialized")

    def get_runner(
        self,
        agent_id: str,
        workspace_dir: Optional[Path] = None,
        mcp_client_manager: Optional[Any] = None,
        memory_manager: Optional[Any] = None,
    ) -> LangGraphRunnerAdapter:
        """Get or create a LangGraph runner for the given agent.

        Args:
            agent_id: Agent configuration ID
            workspace_dir: Workspace directory for this agent
            mcp_client_manager: Optional MCP client manager
            memory_manager: Optional memory manager

        Returns:
            LangGraphRunnerAdapter instance
        """
        # Return cached runner if available
        if agent_id in self._runners:
            return self._runners[agent_id]

        # Determine workspace directory
        if workspace_dir is None:
            workspace_dir = self.workspace_dir
        if workspace_dir is None:
            from ..constant import WORKING_DIR
            workspace_dir = WORKING_DIR / "agents" / agent_id

        # Create runner
        logger.info(f"Creating LangGraph runner for agent: {agent_id}")

        runner = LangGraphRunnerAdapter(
            agent_id=agent_id,
            workspace_dir=workspace_dir,
            mcp_client_manager=mcp_client_manager,
            memory_manager=memory_manager,
        )

        self._runners[agent_id] = runner
        return runner

    def set_mcp_manager(self, mcp_client_manager: Any) -> None:
        """Set MCP client manager for tool integration.

        Args:
            mcp_client_manager: CoPaw's MCPClientManager instance
        """
        self._mcp_adapter = MCPClientManagerAdapter(mcp_client_manager)
        logger.info("MCP client manager set for LangGraph factory")

    def set_memory_manager(self, memory_manager: Any) -> None:
        """Set memory manager for checkpoint integration.

        Args:
            memory_manager: CoPaw's memory manager instance
        """
        self._checkpoint_adapter = LangGraphCheckpointAdapter(
            memory_manager=memory_manager,
        )
        logger.info("Memory manager set for LangGraph factory")

    async def refresh_runner(self, agent_id: str) -> LangGraphRunnerAdapter:
        """Refresh a runner (for hot-reload).

        Args:
            agent_id: Agent configuration ID

        Returns:
            New runner instance
        """
        # Remove cached runner
        if agent_id in self._runners:
            old_runner = self._runners[agent_id]
            # Cleanup if needed
            self._runners.pop(agent_id)
            logger.info(f"Refreshed LangGraph runner for agent: {agent_id}")

        # Create new runner
        return self.get_runner(agent_id)

    def remove_runner(self, agent_id: str) -> bool:
        """Remove a runner.

        Args:
            agent_id: Agent configuration ID

        Returns:
            True if removed
        """
        if agent_id in self._runners:
            self._runners.pop(agent_id)
            logger.info(f"Removed LangGraph runner for agent: {agent_id}")
            return True
        return False

    def clear_all(self) -> None:
        """Clear all cached runners."""
        self._runners.clear()
        logger.info("Cleared all LangGraph runners")

    def get_runner_count(self) -> int:
        """Get number of active runners."""
        return len(self._runners)


# Global factory instance
_factory: Optional[LangGraphRunnerFactory] = None


def get_langgraph_runner_factory(
    workspace_dir: Optional[Path] = None,
) -> LangGraphRunnerFactory:
    """Get the global LangGraph runner factory instance.

    Args:
        workspace_dir: Optional workspace directory

    Returns:
        LangGraphRunnerFactory instance
    """
    global _factory
    if _factory is None:
        _factory = LangGraphRunnerFactory(workspace_dir)
    return _factory