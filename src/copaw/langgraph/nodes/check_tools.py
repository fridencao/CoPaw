# -*- coding: utf-8 -*-
"""Tool check node for LangGraph.

This node determines whether the agent should execute tools or
respond directly based on the presence of tool calls.
"""

import logging
from typing import Literal

from ..state import AgentState

logger = logging.getLogger(__name__)

# Possible next steps after tool check
NextStep = Literal["act", "respond", "error"]


def check_tools_node(state: AgentState) -> NextStep:
    """Check if there are tool calls to execute.

    This node examines the state's tool_calls to determine the next
    step in the ReAct loop:
    - "act": Execute tools
    - "respond": Return response directly (no tools)
    - "error": Something went wrong

    Args:
        state: Current agent state

    Returns:
        Next step identifier
    """
    tool_calls = state.get("tool_calls", [])
    should_terminate = state.get("should_terminate", False)
    termination_reason = state.get("termination_reason")

    # Check if we should terminate
    if should_terminate:
        logger.info(
            "check_tools_node: terminating, reason=%s",
            termination_reason,
        )
        return "respond"

    # Check if there are tool calls
    if tool_calls:
        logger.debug(
            "check_tools_node: %d tool calls to execute",
            len(tool_calls),
        )
        return "act"

    # No tool calls, respond directly
    logger.debug("check_tools_node: no tool calls, responding directly")
    return "respond"


__all__ = ["check_tools_node", "NextStep"]