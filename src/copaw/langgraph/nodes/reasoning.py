# -*- coding: utf-8 -*-
"""ReAct reasoning node for LangGraph.

This node handles the reasoning step where the model generates
thoughts and decides on actions (including tool calls).
"""

import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from ..state import AgentState

logger = logging.getLogger(__name__)

# Default message when max iterations reached
MAX_ITERATIONS_MESSAGE = (
    "已达到最大迭代次数，回复已终止，请继续输入。\n"
    "Maximum iterations reached for this round. "
    "Please send a new message to continue."
)


async def reasoning_node(state: AgentState) -> AgentState:
    """ReAct reasoning node - model generates thought + action.

    This node invokes the chat model with the current conversation
    history and extracts any tool calls from the response.

    Args:
        state: Current agent state

    Returns:
        Updated state with model response and parsed tool calls
    """
    chat_model = state.get("chat_model")
    if chat_model is None:
        logger.error("No chat_model in state")
        return {
            **state,
            "should_terminate": True,
            "termination_reason": "no_chat_model",
        }

    # Check if max iterations reached
    if state["iteration_count"] >= state["max_iterations"]:
        logger.info(
            "Max iterations reached: %d >= %d",
            state["iteration_count"],
            state["max_iterations"],
        )
        return {
            **state,
            "messages": state["messages"]
            + [AIMessage(content=MAX_ITERATIONS_MESSAGE)],
            "should_terminate": True,
            "termination_reason": "max_iterations",
        }

    try:
        # Build messages for the model
        # Include system prompt if available
        messages = list(state["messages"])

        # Invoke the model
        logger.debug(
            "Reasoning node: invoking model with %d messages",
            len(messages),
        )

        response = await chat_model.ainvoke(messages)

        # Extract tool calls from response
        tool_calls = []
        if hasattr(response, "tool_calls") and response.tool_calls:
            for tc in response.tool_calls:
                tool_calls.append({
                    "id": tc.get("id", ""),
                    "name": tc.get("name", ""),
                    "args": tc.get("args", {}),
                })
                logger.debug(
                    "Reasoning node: extracted tool call - %s",
                    tc.get("name", ""),
                )

        # Increment iteration count
        new_iteration_count = state["iteration_count"] + 1

        logger.debug(
            "Reasoning node: iteration %d, tool_calls: %d",
            new_iteration_count,
            len(tool_calls),
        )

        return {
            **state,
            "messages": [response],
            "tool_calls": tool_calls,
            "iteration_count": new_iteration_count,
            "should_terminate": False,
            "termination_reason": None,
        }

    except Exception as e:
        logger.error("Error in reasoning_node: %s", e, exc_info=True)
        return {
            **state,
            "messages": [
                AIMessage(content=f"Error during reasoning: {str(e)}")
            ],
            "tool_calls": [],
            "should_terminate": True,
            "termination_reason": f"error: {str(e)}",
        }


__all__ = ["reasoning_node", "MAX_ITERATIONS_MESSAGE"]