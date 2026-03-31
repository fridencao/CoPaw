# -*- coding: utf-8 -*-
"""ReAct observation node for LangGraph.

This node processes the results from tool execution and prepares
for the next reasoning iteration.
"""

import logging
from typing import Any

from ..state import AgentState

logger = logging.getLogger(__name__)


async def observation_node(state: AgentState) -> AgentState:
    """ReAct observation node - process tool results.

    This node processes the tool execution results and prepares
    the state for the next reasoning iteration. It can be used to:
    - Log tool results
    - Update memory
    - Check for termination conditions

    Args:
        state: Current agent state

    Returns:
        Updated state ready for next iteration
    """
    tool_results = state.get("tool_results", [])
    tool_calls = state.get("tool_calls", [])

    if not tool_results and not tool_calls:
        logger.debug("observation_node: no results to process")
        return state

    # Log tool execution summary
    error_count = sum(1 for r in tool_results if r.get("is_error"))
    success_count = len(tool_results) - error_count

    logger.debug(
        "observation_node: processed %d tool results "
        "(%d success, %d errors)",
        len(tool_results),
        success_count,
        error_count,
    )

    # Clear tool_calls since they've been executed
    # (tool_results remain for potential debugging/logging)

    # Check if we should terminate based on results
    # For example, if a tool indicates the conversation should end
    should_terminate = state.get("should_terminate", False)
    termination_reason = state.get("termination_reason")

    # If terminating, keep the flag
    if should_terminate:
        return state

    # Otherwise, continue the loop
    return {
        **state,
        "tool_calls": [],  # Clear executed tool calls
    }


__all__ = ["observation_node"]