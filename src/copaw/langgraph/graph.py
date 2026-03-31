# -*- coding: utf-8 -*-
"""ReAct graph definition for LangGraph.

This module assembles the ReAct agent as a StateGraph with
reasoning, tool checking, acting, and observation nodes.
"""

import logging
from typing import Any, Optional

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .state import AgentState, get_initial_state
from .nodes import (
    reasoning_node,
    check_tools_node,
    acting_node,
    observation_node,
)

logger = logging.getLogger(__name__)


def create_react_graph(
    max_iterations: int = 100,
    checkpointer: Optional[MemorySaver] = None,
) -> StateGraph:
    """Create a ReAct agent StateGraph.

    The graph implements the ReAct (Reasoning + Acting) pattern:
    1. reasoning: Model generates thought and potential tool calls
    2. check_tools: Decide whether to execute tools or respond
    3. acting: Execute tools and get results (if tools called)
    4. observation: Process tool results
    5. Loop back to reasoning

    Args:
        max_iterations: Maximum number of reasoning iterations
        checkpointer: Optional MemorySaver for session persistence

    Returns:
        Compiled StateGraph ready for execution
    """
    # Create the graph
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("reasoning", reasoning_node)
    graph.add_node("check_tools", check_tools_node)
    graph.add_node("acting", acting_node)
    graph.add_node("observation", observation_node)

    # Set entry point
    graph.set_entry_point("reasoning")

    # Edge: reasoning -> check_tools
    graph.add_edge("reasoning", "check_tools")

    # Conditional edge: check_tools decides next step
    graph.add_conditional_edges(
        "check_tools",
        check_tools_node,
        {
            "act": "acting",
            "respond": END,
            "error": END,
        },
    )

    # Edge: acting -> observation
    graph.add_edge("acting", "observation")

    # Edge: observation -> reasoning (loop)
    graph.add_edge("observation", "reasoning")

    # Compile with optional checkpointer
    if checkpointer is None:
        checkpointer = MemorySaver()

    compiled_graph = graph.compile(checkpointer=checkpointer)

    logger.info(
        "ReAct graph created with max_iterations=%d",
        max_iterations,
    )

    return compiled_graph


def create_react_graph_with_config(
    session_id: str,
    chat_model: Any,
    tool_registry: Any,
    max_iterations: int = 100,
    system_prompt: str = "",
    user_id: str = "default",
    channel: str = "console",
    agent_id: str = "default",
    checkpointer: Optional[MemorySaver] = None,
) -> tuple[Any, dict]:
    """Create a ReAct graph with configuration and return compiled graph + config.

    This is a convenience function that creates both the graph and
    the initial configuration for execution.

    Args:
        session_id: Session identifier for checkpointing
        chat_model: LangChain chat model instance
        tool_registry: Tool registry instance
        max_iterations: Maximum iterations allowed
        system_prompt: System prompt for the agent
        user_id: User identifier
        channel: Channel name
        agent_id: Agent configuration ID
        checkpointer: Optional MemorySaver for session persistence

    Returns:
        Tuple of (compiled_graph, config_dict)
    """
    # Create the graph
    graph = create_react_graph(
        max_iterations=max_iterations,
        checkpointer=checkpointer,
    )

    # Create config for checkpointing
    config = {
        "configurable": {
            "thread_id": session_id,
        }
    }

    return graph, config


__all__ = [
    "create_react_graph",
    "create_react_graph_with_config",
]