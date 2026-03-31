# -*- coding: utf-8 -*-
"""Agent state definition for LangGraph.

This module defines the state schema used throughout the ReAct graph.
"""

from typing import TypedDict, Annotated, Any, Sequence, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages


class AgentState(TypedDict, total=False):
    """Core state for CoPaw agent in LangGraph.

    This TypedDict defines all the state variables that flow through
    the ReAct graph during agent execution.
    """

    # Messages in the conversation
    # Annotated with add_messages to automatically append new messages
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # Pending tool calls to execute
    tool_calls: list[dict[str, Any]]

    # Results from tool executions
    tool_results: list[dict[str, Any]]

    # Current iteration count (for max_iters limit)
    iteration_count: int

    # Maximum iterations allowed
    max_iterations: int

    # Session identifier for checkpointing
    session_id: str

    # User identifier
    user_id: str

    # Channel name (e.g., 'console', 'dingtalk', 'discord')
    channel: str

    # Agent configuration ID
    agent_id: str

    # Memory content for long-term memory
    memory_content: list[dict[str, Any]]

    # Metadata for additional context
    metadata: dict[str, Any]

    # The LangChain chat model (not serialized, set at runtime)
    chat_model: Any  # BaseChatModel

    # Tool registry reference (not serialized, set at runtime)
    tool_registry: Any  # ToolRegistry

    # System prompt
    system_prompt: str

    # Whether the agent should terminate
    should_terminate: bool

    # Termination reason (if should_terminate is True)
    termination_reason: Optional[str]


def get_initial_state(
    session_id: str,
    user_id: str = "default",
    channel: str = "console",
    agent_id: str = "default",
    max_iterations: int = 100,
    system_prompt: str = "",
) -> AgentState:
    """Get initial state for a new agent conversation.

    Args:
        session_id: Session identifier
        user_id: User identifier
        channel: Channel name
        agent_id: Agent configuration ID
        max_iterations: Maximum iterations allowed
        system_prompt: System prompt for the agent

    Returns:
        Initial AgentState dictionary
    """
    return {
        "messages": [],
        "tool_calls": [],
        "tool_results": [],
        "iteration_count": 0,
        "max_iterations": max_iterations,
        "session_id": session_id,
        "user_id": user_id,
        "channel": channel,
        "agent_id": agent_id,
        "memory_content": [],
        "metadata": {},
        "chat_model": None,
        "tool_registry": None,
        "system_prompt": system_prompt,
        "should_terminate": False,
        "termination_reason": None,
    }


__all__ = ["AgentState", "get_initial_state"]