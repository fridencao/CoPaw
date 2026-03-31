# -*- coding: utf-8 -*-
"""LangGraph Runner for CoPaw.

This module provides the LangGraph-based runner that replaces
the AgentScope-based AgentRunner.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncIterator, Optional

from langchain_core.messages import HumanMessage, AIMessage

if TYPE_CHECKING:
    from agentscope_runtime.engine.schemas.agent_schemas import AgentRequest

logger = logging.getLogger(__name__)


class LangGraphRunner:
    """LangGraph-based runner for CoPaw.

    This runner uses LangGraph to execute agent conversations,
    replacing the AgentScope-based AgentRunner.
    """

    def __init__(
        self,
        agent_id: str = "default",
        workspace_dir: Path | None = None,
        task_tracker: Any | None = None,
    ):
        """Initialize the runner.

        Args:
            agent_id: Agent configuration ID
            workspace_dir: Workspace directory
            task_tracker: Optional task tracker
        """
        self.framework_type = "langgraph"
        self.agent_id = agent_id
        self.workspace_dir = workspace_dir
        self._task_tracker = task_tracker

        self._chat_manager = None
        self._mcp_manager = None
        self._workspace = None
        self.memory_manager = None
        self._langgraph_runner = None

    def set_chat_manager(self, chat_manager):
        """Set chat manager for auto-registration."""
        self._chat_manager = chat_manager

    def set_mcp_manager(self, mcp_manager):
        """Set MCP client manager for hot-reload."""
        self._mcp_manager = mcp_manager

    def set_workspace(self, workspace):
        """Set workspace for control commands."""
        self._workspace = workspace

    def _get_langgraph_runner(self):
        """Get or create the LangGraph runner."""
        if self._langgraph_runner is not None:
            return self._langgraph_runner

        from copaw.langgraph.adapter import LangGraphRunnerAdapter

        self._langgraph_runner = LangGraphRunnerAdapter(
            agent_id=self.agent_id,
            workspace_dir=self.workspace_dir,
            task_tracker=self._task_tracker,
            mcp_client_manager=self._mcp_manager,
            memory_manager=self.memory_manager,
        )

        return self._langgraph_runner

    async def query_handler(
        self,
        msgs: Any,
        request: "AgentRequest" = None,
        **kwargs,
    ) -> AsyncIterator[tuple[Any, bool]]:
        """Handle agent query and yield results.

        Args:
            msgs: List of messages
            request: AgentRequest
            **kwargs: Additional arguments

        Yields:
            Tuples of (message, is_last)
        """
        runner = self._get_langgraph_runner()
        async for result in runner.query_handler(msgs, request, **kwargs):
            yield result

    async def stream_query(
        self,
        msgs: Any,
        request: "AgentRequest" = None,
        **kwargs,
    ) -> AsyncIterator[tuple[Any, bool]]:
        """Stream query results.

        Args:
            msgs: List of messages
            request: AgentRequest
            **kwargs: Additional arguments

        Yields:
            Tuples of (message, is_last)
        """
        runner = self._get_langgraph_runner()
        async for result in runner.stream_query(msgs, request, **kwargs):
            yield result


# Alias for backward compatibility
AgentRunner = LangGraphRunner
HybridRunner = LangGraphRunner