# -*- coding: utf-8 -*-
"""Tool converter for LangGraph.

This module provides utilities for converting CoPaw/AgentScope
tools to LangChain Tool format.
"""

import inspect
import logging
from typing import Any, Callable, Dict, Optional

from langchain_core.tools import BaseTool, Tool

logger = logging.getLogger(__name__)


def convert_agentscope_tool_to_langchain(
    tool_func: Callable,
    name: Optional[str] = None,
    description: str = "",
) -> Tool:
    """Convert an AgentScope tool function to a LangChain Tool.

    This function inspects a tool function's signature and creates
    a LangChain Tool with proper parameter schema.

    Args:
        tool_func: The tool function to convert
        name: Optional tool name (defaults to function name)
        description: Optional tool description

    Returns:
        LangChain Tool instance
    """
    tool_name = name or tool_func.__name__

    # Inspect function signature to build schema
    sig = inspect.signature(tool_func)
    properties = {}
    required = []

    for param_name, param in sig.parameters.items():
        # Skip 'self' and 'cls' parameters
        if param_name in ("self", "cls"):
            continue

        # Determine parameter type from annotation or default
        param_type = "string"
        if param.annotation is not inspect.Parameter.empty:
            ann = param.annotation
            if ann == int:
                param_type = "integer"
            elif ann == float:
                param_type = "number"
            elif ann == bool:
                param_type = "boolean"
            elif ann == list:
                param_type = "array"
            elif ann == dict:
                param_type = "object"

        # Check if required (no default value)
        if param.default is inspect.Parameter.empty:
            required.append(param_name)

        properties[param_name] = {
            "type": param_type,
            "description": f"Parameter {param_name}",
        }

    # Build args schema
    args_schema = {
        "type": "object",
        "properties": properties,
        "required": required,
    }

    # Create the tool
    tool = Tool(
        name=tool_name,
        description=description or tool_func.__doc__ or f"Tool: {tool_name}",
        args_schema=args_schema,
        func=tool_func,
        coroutine=tool_func,
    )

    logger.debug(
        "Converted tool function '%s' to LangChain Tool "
        "(params: %s, required: %s)",
        tool_name,
        list(properties.keys()),
        required,
    )

    return tool


def convert_builtin_tools() -> Dict[str, Tool]:
    """Convert CoPaw's built-in tools to LangChain Tools.

    Returns:
        Dictionary mapping tool names to LangChain Tool instances
    """
    tools = {}

    try:
        # Import from agentscope and copaw.agents.tools
        from agentscope.tool import execute_python_code, view_text_file, write_text_file
        from copaw.agents.tools import (
            execute_shell_command,
            read_file,
            write_file,
            edit_file,
            grep_search,
            glob_search,
            get_current_time,
            set_user_timezone,
            get_token_usage,
        )

        tool_mappings = {
            "execute_python_code": (
                execute_python_code,
                "Execute Python code in a sandboxed environment. "
                "Returns the output or error.",
            ),
            "view_text_file": (
                view_text_file,
                "View the content of a text file. "
                "Returns the file contents as a string.",
            ),
            "write_text_file": (
                write_text_file,
                "Write text content to a file. "
                "Creates the file if it doesn't exist, "
                "overwrites if it does.",
            ),
            "execute_shell_command": (
                execute_shell_command,
                "Execute a shell command and return the output. "
                "Use this for running system commands.",
            ),
            "read_file": (
                read_file,
                "Read the content of a file. "
                "Returns the file contents as a string.",
            ),
            "write_file": (
                write_file,
                "Write content to a file. "
                "Creates the file if it doesn't exist, "
                "overwrites if it does.",
            ),
            "edit_file": (
                edit_file,
                "Edit a specific part of a file. "
                "Use this to make targeted changes to file content.",
            ),
            "grep_search": (
                grep_search,
                "Search for text patterns in files using grep. "
                "Returns matching lines with file paths.",
            ),
            "glob_search": (
                glob_search,
                "Find files matching a glob pattern. "
                "Useful for locating files by name patterns.",
            ),
            "get_current_time": (
                get_current_time,
                "Get the current date and time. "
                "Useful for timestamps and time-based operations.",
            ),
            "set_user_timezone": (
                set_user_timezone,
                "Set the user's timezone. "
                "Used for timezone-aware time operations.",
            ),
            "get_token_usage": (
                get_token_usage,
                "Get token usage statistics for the current session. "
                "Returns token counts for prompts and completions.",
            ),
        }

        for tool_name, (tool_func, description) in tool_mappings.items():
            try:
                tools[tool_name] = convert_agentscope_tool_to_langchain(
                    tool_func,
                    name=tool_name,
                    description=description,
                )
            except Exception as e:
                logger.warning(
                    "Failed to convert tool '%s': %s",
                    tool_name,
                    e,
                )

        logger.info("Converted %d built-in tools", len(tools))

    except ImportError as e:
        logger.warning(
            "Could not import built-in tools for conversion: %s",
            e,
        )

    return tools


def create_tool_from_function(
    func: Callable,
    name: Optional[str] = None,
    description: str = "",
) -> Tool:
    """Create a LangChain Tool from a function.

    This is an alias for convert_agentscope_tool_to_langchain
    for clearer API usage.

    Args:
        func: The function to convert
        name: Optional tool name
        description: Optional tool description

    Returns:
        LangChain Tool instance
    """
    return convert_agentscope_tool_to_langchain(
        func,
        name=name,
        description=description,
    )


def create_async_tool_from_function(
    func: Callable,
    name: Optional[str] = None,
    description: str = "",
) -> Tool:
    """Create an async LangChain Tool from a function.

    Args:
        func: The async function to convert
        name: Optional tool name
        description: Optional tool description

    Returns:
        LangChain Tool instance with async execution
    """
    tool_name = name or func.__name__

    # For async functions, we use ainvoke
    async def async_wrapper(**kwargs):
        return await func(**kwargs)

    # Build schema similar to sync version
    sig = inspect.signature(func)
    properties = {}
    required = []

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue

        param_type = "string"
        if param.annotation is not inspect.Parameter.empty:
            ann = param.annotation
            if ann == int:
                param_type = "integer"
            elif ann == float:
                param_type = "number"
            elif ann == bool:
                param_type = "boolean"
            elif ann == list:
                param_type = "array"
            elif ann == dict:
                param_type = "object"

        if param.default is inspect.Parameter.empty:
            required.append(param_name)

        properties[param_name] = {
            "type": param_type,
            "description": f"Parameter {param_name}",
        }

    args_schema = {
        "type": "object",
        "properties": properties,
        "required": required,
    }

    return Tool(
        name=tool_name,
        description=description or func.__doc__ or f"Tool: {tool_name}",
        args_schema=args_schema,
        func=None,  # No sync version
        coroutine=async_wrapper,
    )


__all__ = [
    "convert_agentscope_tool_to_langchain",
    "convert_builtin_tools",
    "create_tool_from_function",
    "create_async_tool_from_function",
]