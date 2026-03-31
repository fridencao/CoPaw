# -*- coding: utf-8 -*-
"""LangGraph Runner Adapter.

This module provides an adapter that makes LangGraphRunner compatible
with the existing Runner interface used by AgentApp.

This allows for easier migration - the adapter can be used while
testing, and eventually replace the original runner.
"""

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncIterator, Optional

from langchain_core.messages import HumanMessage, AIMessage

from .runner import LangGraphRunner
from .tools import tool_registry, register_builtin_tools

if TYPE_CHECKING:
    from agentscope_runtime.engine.schemas.agent_schemas import AgentRequest

logger = logging.getLogger(__name__)


class LangGraphRunnerAdapter:
    """Adapter to make LangGraphRunner compatible with AgentApp.

    This adapter wraps LangGraphRunner and provides the same interface
    as the original AgentScope-based AgentRunner, allowing for
    incremental migration.
    """

    def __init__(
        self,
        agent_id: str = "default",
        workspace_dir: Path | None = None,
        task_tracker: Any | None = None,
        mcp_client_manager: Any | None = None,
        memory_manager: Any | None = None,
    ):
        """Initialize the adapter.

        Args:
            agent_id: Agent configuration ID
            workspace_dir: Workspace directory for prompts
            task_tracker: Optional task tracker for cancellation support
            mcp_client_manager: Optional MCP client manager
            memory_manager: Optional memory manager
        """
        self.framework_type = "langgraph"
        self.agent_id = agent_id
        self.workspace_dir = workspace_dir
        self._task_tracker = task_tracker
        self._mcp_client_manager = mcp_client_manager
        self._memory_manager = memory_manager

        # Store configuration for runner creation
        self._agent_config = None
        self._runner: Optional[LangGraphRunner] = None

        # Register built-in tools on init
        register_builtin_tools()

        # MCP adapter
        self._mcp_adapter = None
        if mcp_client_manager is not None:
            from .adapters import MCPClientManagerAdapter
            self._mcp_adapter = MCPClientManagerAdapter(mcp_client_manager)

        logger.info(
            "LangGraphRunnerAdapter initialized for agent '%s'",
            agent_id,
        )

    def _get_or_create_runner(self, request: "AgentRequest") -> LangGraphRunner:
        """Get or create the LangGraphRunner instance."""
        if self._runner is not None:
            return self._runner

        # Load agent config
        from ...config.config import load_agent_config

        try:
            self._agent_config = load_agent_config(self.agent_id)
        except Exception as e:
            logger.error(f"Failed to load agent config: {e}")
            raise

        # Get additional tools from MCP if available
        mcp_tools = []
        if self._mcp_adapter is not None:
            try:
                import asyncio
                try:
                    # Try to get tools synchronously first
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # We're in async context, schedule in background
                        # Tools will be retrieved on first call
                        logger.debug("In async context, deferring MCP tool retrieval")
                    else:
                        mcp_tools = loop.run_until_complete(
                            self._mcp_adapter.get_tools()
                        )
                except RuntimeError:
                    # No event loop, will be created automatically
                    mcp_tools = asyncio.run(self._mcp_adapter.get_tools())
            except Exception as e:
                logger.warning(f"Failed to get MCP tools: {e}")

        # Create runner
        self._runner = LangGraphRunner(
            agent_config=self._agent_config,
            tool_registry=tool_registry,
            enable_builtin_tools=False,  # Already registered
            mcp_tools=mcp_tools,
            memory_manager=self._memory_manager,
            workspace_dir=self.workspace_dir,
        )

        return self._runner

    async def query_handler(
        self,
        msgs: Any,
        request: "AgentRequest" = None,
        **kwargs,
    ) -> AsyncIterator[tuple[Any, bool]]:
        """Handle agent query (compatible interface).

        This method provides the same interface as AgentRunner.query_handler,
        but uses LangGraph internally.

        Args:
            msgs: List of messages
            request: AgentRequest containing session/user info
            **kwargs: Additional arguments

        Yields:
            Tuples of (message, is_last)
        """
        if request is None:
            logger.warning("query_handler called without request")
            yield (AIMessage(content="Error: No request provided"), True)
            return

        # Extract user input from messages
        user_input = self._extract_user_input(msgs)
        if not user_input:
            yield (AIMessage(content="Error: No user input found"), True)
            return

        session_id = request.session_id or "default"
        user_id = request.user_id or "default"
        channel = getattr(request, "channel", "console")

        logger.info(
            "LangGraphRunnerAdapter.query_handler: "
            "session_id=%s, user_id=%s, input=%s",
            session_id,
            user_id,
            user_input[:50],
        )

        try:
            # Get or create runner
            runner = self._get_or_create_runner(request)

            # Execute and yield events
            last = False
            async for event in runner.execute(
                user_input=user_input,
                session_id=session_id,
                user_id=user_id,
                channel=channel,
            ):
                # Convert event to message format
                event_type = event.get("type")

                if event_type == "message":
                    content = event.get("content", "")
                    yield (AIMessage(content=content), False)
                    last = True

                elif event_type == "tool_call":
                    # Could yield tool call events
                    pass

                elif event_type == "tool_result":
                    # Tool results are typically not shown to user
                    pass

                elif event_type == "error":
                    error_content = event.get("content", "Unknown error")
                    yield (AIMessage(content=f"Error: {error_content}"), True)
                    return

                elif event_type == "usage":
                    # Token usage - could be logged or returned
                    logger.debug(
                        "Token usage: %s",
                        event.get("content", {}),
                    )

            if not last:
                yield (AIMessage(content=""), True)

        except Exception as e:
            logger.error(
                "Error in LangGraphRunnerAdapter.query_handler: %s",
                e,
                exc_info=True,
            )
            yield (AIMessage(content=f"Error: {str(e)}"), True)

    def _extract_user_input(self, msgs: Any) -> str:
        """Extract user input from messages.

        Args:
            msgs: List of message objects

        Returns:
            User input string
        """
        if not msgs:
            return ""

        # Handle different message formats
        last_msg = msgs[-1] if isinstance(msgs, (list, tuple)) else msgs

        # Try to extract content based on message type
        if hasattr(last_msg, "get_text_content"):
            # AgentScope Msg
            return last_msg.get_text_content() or ""
        elif hasattr(last_msg, "content"):
            # LangChain message or similar
            content = last_msg.content
            if isinstance(content, str):
                return content
            elif isinstance(content, list):
                # Handle content blocks
                text_parts = []
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                return "".join(text_parts)
        elif isinstance(last_msg, str):
            return last_msg

        return str(last_msg) if last_msg else ""

    async def stream_query(
        self,
        msgs: Any,
        request: "AgentRequest" = None,
        **kwargs,
    ) -> AsyncIterator[tuple[Any, bool]]:
        """Stream query (alias for query_handler).

        Args:
            msgs: List of messages
            request: AgentRequest
            **kwargs: Additional arguments

        Yields:
            Tuples of (message, is_last)
        """
        async for result in self.query_handler(msgs, request, **kwargs):
            yield result

    # Compatibility methods

    def set_chat_manager(self, chat_manager: Any) -> None:
        """Set chat manager (for compatibility, not used in LangGraph)."""
        pass

    def set_mcp_manager(self, mcp_manager: Any) -> None:
        """Set MCP client manager (for compatibility)."""
        pass

    def set_workspace(self, workspace: Any) -> None:
        """Set workspace (for compatibility).

        This extracts the task_tracker from the workspace for cancellation support.
        """
        if workspace is not None:
            # Try to get task_tracker from workspace
            self._task_tracker = getattr(workspace, "task_tracker", None)
            if self._task_tracker:
                logger.info(
                    "LangGraphRunnerAdapter: TaskTracker integrated from workspace"
                )

    # TaskTracker integration for cancellation support

    async def get_task_status(self, run_key: str) -> str:
        """Get task status from TaskTracker.

        Args:
            run_key: The run key (usually chat_id)

        Returns:
            'idle' or 'running'
        """
        if self._task_tracker is not None:
            return await self._task_tracker.get_status(run_key)
        return "idle"

    async def request_task_stop(self, run_key: str) -> bool:
        """Request task cancellation via TaskTracker.

        Args:
            run_key: The run key (usually chat_id)

        Returns:
            True if task was cancelled, False otherwise
        """
        if self._task_tracker is not None:
            return await self._task_tracker.request_stop(run_key)
        return False

    async def has_active_tasks(self) -> bool:
        """Check if there are active tasks.

        Returns:
            True if any tasks are running
        """
        if self._task_tracker is not None:
            return await self._task_tracker.has_active_tasks()
        return False

    async def list_active_tasks(self) -> list[str]:
        """List all active task keys.

        Returns:
            List of active run_keys
        """
        if self._task_tracker is not None:
            return await self._task_tracker.list_active_tasks()
        return []


def create_langgraph_runner_adapter(
    agent_id: str = "default",
    workspace_dir: Path | None = None,
    task_tracker: Any | None = None,
) -> LangGraphRunnerAdapter:
    """Factory function to create a LangGraph runner adapter.

    Args:
        agent_id: Agent configuration ID
        workspace_dir: Workspace directory
        task_tracker: Optional task tracker

    Returns:
        LangGraphRunnerAdapter instance
    """
    return LangGraphRunnerAdapter(
        agent_id=agent_id,
        workspace_dir=workspace_dir,
        task_tracker=task_tracker,
    )


__all__ = [
    "LangGraphRunnerAdapter",
    "create_langgraph_runner_adapter",
]