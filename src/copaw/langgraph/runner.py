# -*- coding: utf-8 -*-
"""LangGraph Runner for CoPaw.

This module provides the Runner implementation that uses LangGraph
to execute agent conversations, replacing the AgentScope-based Runner.
"""

import logging
from typing import Any, AsyncIterator, Optional

from langchain_core.messages import HumanMessage

from .graph import create_react_graph
from .state import AgentState, get_initial_state
from .tools import ToolRegistry, tool_registry, register_builtin_tools
from ..langchain.factory import create_langchain_model_by_agent_id
from ..langchain.callbacks import TokenUsageCallbackHandler

logger = logging.getLogger(__name__)


class LangGraphRunner:
    """Runner implementation using LangGraph.

    This class provides the core agent execution logic using LangGraph's
    StateGraph with ReAct pattern. It handles:
    - Agent initialization with model and tools
    - Conversation execution with streaming support
    - Session checkpointing for persistence
    - Token usage tracking
    """

    def __init__(
        self,
        agent_config: Any,
        tool_registry: Optional[ToolRegistry] = None,
        memory_manager: Any = None,
        enable_builtin_tools: bool = True,
        mcp_tools: Optional[list] = None,
        workspace_dir: Any = None,
        **kwargs: Any,
    ):
        """Initialize the LangGraph Runner.

        Args:
            agent_config: Agent configuration (AgentProfileConfig)
            tool_registry: Optional tool registry (uses global if not provided)
            memory_manager: Optional memory manager for long-term memory
            enable_builtin_tools: Whether to register built-in tools
            mcp_tools: Optional list of MCP tools to add
            workspace_dir: Workspace directory for reading config files
            **kwargs: Additional configuration options
        """
        self.agent_config = agent_config
        self.memory_manager = memory_manager
        self.mcp_tools = mcp_tools or []
        self.workspace_dir = workspace_dir

        # Setup checkpointer with memory manager integration
        from .adapters import CoPawCheckpointer
        self._checkpointer = CoPawCheckpointer(memory_manager=memory_manager)

        # Setup tool registry
        if tool_registry is not None:
            self.tool_registry = tool_registry
        else:
            self.tool_registry = tool_registry  # Uses global tool_registry
            if enable_builtin_tools:
                register_builtin_tools()

        # Register MCP tools if provided
        if self.mcp_tools:
            for tool in self.mcp_tools:
                if tool is not None:
                    self.tool_registry.register_tool(tool)
            logger.info(f"Registered {len(self.mcp_tools)} MCP tools")

        # Get configuration
        running_config = agent_config.running
        self._max_iterations = running_config.max_iters
        self._system_prompt = self._build_system_prompt()

        # Create the ReAct graph
        self.graph = create_react_graph(
            max_iterations=self._max_iterations,
            checkpointer=self._checkpointer,
        )

        # Initialize chat model (will be set on first execute)
        self._chat_model = None

        logger.info(
            "LangGraphRunner initialized for agent '%s' "
            "(max_iterations=%d)",
            agent_config.id,
            self._max_iterations,
        )

    def _build_system_prompt(self) -> str:
        """Build system prompt from agent configuration.

        Returns:
            System prompt string
        """
        from .prompts import build_system_prompt

        return build_system_prompt(
            agent_config=self.agent_config,
            workspace_dir=self.workspace_dir if hasattr(self, 'workspace_dir') else None,
        )

    def _get_chat_model(self, callbacks: Optional[list] = None):
        """Get or create the chat model.

        Args:
            callbacks: Optional list of callback handlers

        Returns:
            LangChain chat model instance
        """
        if self._chat_model is not None:
            return self._chat_model

        # Create model using the LangChain factory
        model, _ = create_langchain_model_by_agent_id(
            agent_id=self.agent_config.id,
            callbacks=callbacks,
        )

        self._chat_model = model
        return model

    def _prepare_initial_state(
        self,
        session_id: str,
        user_id: str = "default",
        channel: str = "console",
    ) -> AgentState:
        """Prepare initial state for a new conversation.

        Args:
            session_id: Session identifier
            user_id: User identifier
            channel: Channel name

        Returns:
            Initial AgentState
        """
        return get_initial_state(
            session_id=session_id,
            user_id=user_id,
            channel=channel,
            agent_id=self.agent_config.id,
            max_iterations=self._max_iterations,
            system_prompt=self._system_prompt,
        )

    async def execute(
        self,
        user_input: str,
        session_id: str,
        user_id: str = "default",
        channel: str = "console",
    ) -> AsyncIterator[dict[str, Any]]:
        """Execute a user input and yield events.

        This is the main entry point for running agent conversations.
        It supports streaming responses through the async iterator.

        Args:
            user_input: User's input message
            session_id: Session identifier for checkpointing
            user_id: User identifier
            channel: Channel name

        Yields:
            Event dictionaries containing:
            - type: Event type (message, tool_call, tool_result, etc.)
            - content: Event content
            - metadata: Optional metadata
        """
        # Get chat model with token tracking
        token_callback = TokenUsageCallbackHandler(
            provider_id=self.agent_config.active_model.provider_id
            if self.agent_config.active_model
            else "unknown",
            model_id=self.agent_config.active_model.model
            if self.agent_config.active_model
            else "unknown",
            session_id=session_id,
            user_id=user_id,
        )

        chat_model = self._get_chat_model(callbacks=[token_callback])

        # Prepare initial state
        state = self._prepare_initial_state(
            session_id=session_id,
            user_id=user_id,
            channel=channel,
        )

        # Set up runtime components (not serialized)
        state["chat_model"] = chat_model
        state["tool_registry"] = self.tool_registry

        # Add user message
        user_message = HumanMessage(content=user_input)
        state["messages"] = [user_message]

        # Create checkpoint config
        config = {
            "configurable": {
                "thread_id": session_id,
            }
        }

        logger.info(
            "LangGraphRunner.execute: session_id=%s, input_length=%d",
            session_id,
            len(user_input),
        )

        # Execute the graph
        try:
            async for event in self.graph.astream(state, config):
                # Parse event and yield formatted output
                for node_name, node_output in event.items():
                    if node_name == "reasoning":
                        # Yield model response
                        if hasattr(node_output, "content"):
                            yield {
                                "type": "message",
                                "content": node_output.content,
                                "metadata": {
                                    "node": node_name,
                                    "has_tool_calls": bool(
                                        getattr(node_output, "tool_calls", None)
                                    ),
                                },
                            }
                        # Yield tool calls if present
                        tool_calls = getattr(node_output, "tool_calls", None)
                        if tool_calls:
                            yield {
                                "type": "tool_call",
                                "tool_calls": tool_calls,
                                "metadata": {"node": node_name},
                            }

                    elif node_name == "acting":
                        # Yield tool results
                        tool_results = node_output.get("tool_results", [])
                        for result in tool_results:
                            yield {
                                "type": "tool_result",
                                "tool_name": result.get("tool_name", ""),
                                "content": result.get("content", ""),
                                "is_error": result.get("is_error", False),
                                "metadata": {"node": node_name},
                            }

                    elif node_name == "check_tools":
                        # Yield decision
                        yield {
                            "type": "decision",
                            "next_step": node_output,
                            "metadata": {"node": node_name},
                        }

            # Yield token usage at the end
            usage_dict = token_callback.get_usage_dict()
            yield {
                "type": "usage",
                "content": usage_dict,
                "metadata": {},
            }

        except Exception as e:
            logger.error(
                "Error in LangGraphRunner.execute: %s",
                e,
                exc_info=True,
            )
            yield {
                "type": "error",
                "content": str(e),
                "metadata": {"session_id": session_id},
            }

    async def execute_simple(
        self,
        user_input: str,
        session_id: str,
        user_id: str = "default",
        channel: str = "console",
    ) -> str:
        """Execute a user input and return the final response.

        This is a simplified interface that returns only the final
        assistant message.

        Args:
            user_input: User's input message
            session_id: Session identifier
            user_id: User identifier
            channel: Channel name

        Returns:
            Assistant's response text
        """
        response_text = ""
        tool_results = []

        async for event in self.execute(
            user_input=user_input,
            session_id=session_id,
            user_id=user_id,
            channel=channel,
        ):
            if event["type"] == "message":
                response_text = event["content"]
            elif event["type"] == "tool_result":
                tool_results.append(event)

        return response_text

    async def get_session_history(
        self,
        session_id: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Get conversation history for a session.

        Args:
            session_id: Session identifier
            limit: Maximum number of messages to return

        Returns:
            List of message dictionaries
        """
        try:
            config = {"configurable": {"thread_id": session_id}}
            checkpoint = self._checkpointer.get(config)

            if checkpoint and "messages" in checkpoint:
                messages = checkpoint["messages"]
                # Return last 'limit' messages
                return messages[-limit:] if len(messages) > limit else messages

        except Exception as e:
            logger.warning(f"Failed to get session history: {e}")

        return []

    async def clear_session(self, session_id: str) -> None:
        """Clear a session's history.

        Args:
            session_id: Session identifier
        """
        try:
            config = {"configurable": {"thread_id": session_id}}
            self._checkpointer.delete(config)
            logger.info(f"Cleared session history: {session_id}")
        except Exception as e:
            logger.warning(f"Failed to clear session: {e}")


__all__ = ["LangGraphRunner"]