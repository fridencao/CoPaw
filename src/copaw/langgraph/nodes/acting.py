# -*- coding: utf-8 -*-
"""ReAct acting node for LangGraph.

This node executes tools and returns the results as ToolMessage
objects that get added to the conversation.
"""

import logging
from typing import Any

from langchain_core.messages import ToolMessage, AIMessage

from ..state import AgentState

logger = logging.getLogger(__name__)


async def acting_node(state: AgentState) -> AgentState:
    """ReAct acting node - execute tools and return results.

    This node executes the tools specified in state["tool_calls"]
    and returns the results as ToolMessage objects.

    Args:
        state: Current agent state

    Returns:
        Updated state with tool execution results
    """
    tool_registry = state.get("tool_registry")
    if tool_registry is None:
        logger.error("No tool_registry in state")
        return {
            **state,
            "tool_results": [{
                "tool_call_id": "error",
                "content": "Tool registry not available",
                "is_error": True,
            }],
        }

    tool_calls = state.get("tool_calls", [])

    if not tool_calls:
        logger.debug("acting_node: no tool calls to execute")
        return state

    logger.info(
        "acting_node: executing %d tool calls",
        len(tool_calls),
    )

    results = []
    tool_messages = []

    for tc in tool_calls:
        tool_name = tc.get("name", "")
        tool_args = tc.get("args", {})
        tool_call_id = tc.get("id", f"call_{tool_name}")

        # Get tool from registry
        tool_func = tool_registry.get_tool(tool_name)

        if tool_func is None:
            error_msg = f"Tool '{tool_name}' not found"
            logger.warning("acting_node: %s", error_msg)
            results.append({
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "content": error_msg,
                "is_error": True,
            })
            tool_messages.append(
                ToolMessage(
                    tool_call_id=tool_call_id,
                    content=error_msg,
                )
            )
            continue

        # Execute tool
        try:
            logger.debug(
                "acting_node: executing tool '%s' with args: %s",
                tool_name,
                tool_args,
            )

            # Execute tool (can be sync or async)
            if hasattr(tool_func, "ainvoke"):
                result = await tool_func.ainvoke(tool_args)
            elif hasattr(tool_func, "invoke"):
                result = tool_func.invoke(tool_args)
            else:
                # Assume sync function
                result = tool_func(**tool_args)

            result_str = str(result) if result is not None else ""

            results.append({
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "content": result_str,
                "is_error": False,
            })

            tool_messages.append(
                ToolMessage(
                    tool_call_id=tool_call_id,
                    content=result_str,
                )
            )

            logger.debug(
                "acting_node: tool '%s' executed successfully",
                tool_name,
            )

        except Exception as e:
            error_msg = f"Error executing tool '{tool_name}': {str(e)}"
            logger.error(
                "acting_node: %s",
                error_msg,
                exc_info=True,
            )
            results.append({
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "content": error_msg,
                "is_error": True,
            })
            tool_messages.append(
                ToolMessage(
                    tool_call_id=tool_call_id,
                    content=error_msg,
                )
            )

    logger.info(
        "acting_node: executed %d tools, %d errors",
        len(results),
        sum(1 for r in results if r.get("is_error")),
    )

    return {
        **state,
        "tool_results": results,
        # tool_messages will be automatically added to messages
        # via the add_messages reducer in AgentState
        "messages": tool_messages,
    }


__all__ = ["acting_node"]