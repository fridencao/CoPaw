# -*- coding: utf-8 -*-
"""Tool registry for LangGraph.

This module provides a registry for managing tools that can be
used by the LangGraph agent.
"""

import logging
from typing import Any, Callable, Dict, Optional

from langchain_core.tools import BaseTool, Tool

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Registry for managing agent tools.

    This class provides a central registry for tools that can be
    used by the LangGraph agent. It supports:
    - Registering tools by name
    - Looking up tools by name
    - Listing all available tools
    - Converting CoPaw built-in tools to LangChain format
    """

    def __init__(self):
        """Initialize an empty tool registry."""
        self._tools: Dict[str, BaseTool] = {}
        self._tool_functions: Dict[str, Callable] = {}

    def register_tool(self, tool: BaseTool) -> None:
        """Register a tool in the registry.

        Args:
            tool: LangChain BaseTool instance
        """
        self._tools[tool.name] = tool
        logger.debug("Registered tool: %s", tool.name)

    def register_tool_function(
        self,
        func: Callable,
        name: Optional[str] = None,
        description: str = "",
    ) -> None:
        """Register a function as a tool.

        Args:
            func: The function to register
            name: Optional tool name (defaults to function name)
            description: Optional tool description
        """
        tool_name = name or func.__name__

        # Create a LangChain Tool from the function
        tool = Tool(
            name=tool_name,
            description=description or func.__doc__ or f"Tool: {tool_name}",
            func=func,
            coroutine=func,
        )

        self._tools[tool_name] = tool
        self._tool_functions[tool_name] = func
        logger.debug(
            "Registered tool function: %s (async=%s)",
            tool_name,
            hasattr(func, '__ainvoke__') or hasattr(func, 'ainvoke'),
        )

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name.

        Args:
            name: Tool name

        Returns:
            Tool instance or None if not found
        """
        return self._tools.get(name)

    def get_tool_function(self, name: str) -> Optional[Callable]:
        """Get a raw tool function by name.

        Args:
            name: Tool name

        Returns:
            Function or None if not found
        """
        return self._tool_functions.get(name)

    def list_tools(self) -> list[str]:
        """List all registered tool names.

        Returns:
            List of tool names
        """
        return list(self._tools.keys())

    def has_tool(self, name: str) -> bool:
        """Check if a tool is registered.

        Args:
            name: Tool name

        Returns:
            True if tool exists
        """
        return name in self._tools

    def unregister_tool(self, name: str) -> bool:
        """Unregister a tool.

        Args:
            name: Tool name

        Returns:
            True if tool was removed, False if not found
        """
        if name in self._tools:
            del self._tools[name]
            self._tool_functions.pop(name, None)
            logger.debug("Unregistered tool: %s", name)
            return True
        return False

    def clear(self) -> None:
        """Clear all registered tools."""
        self._tools.clear()
        self._tool_functions.clear()
        logger.debug("Cleared all tools from registry")

    def __len__(self) -> int:
        """Get number of registered tools."""
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        """Check if tool is registered."""
        return name in self._tools

    def __getitem__(self, name: str) -> BaseTool:
        """Get tool by name (raises KeyError if not found)."""
        return self._tools[name]


# Global default tool registry instance
tool_registry = ToolRegistry()


def get_tool_registry() -> ToolRegistry:
    """Get the global tool registry instance.

    Returns:
        Global ToolRegistry instance
    """
    return tool_registry


def register_builtin_tools() -> None:
    """Register CoPaw's built-in tools to the global registry.

    This function imports and registers all the built-in tools
    from copaw.agents.tools.
    """
    try:
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

        tool_functions = {
            "execute_shell_command": (
                execute_shell_command,
                "Execute a shell command and return the output",
            ),
            "read_file": (
                read_file,
                "Read content from a file",
            ),
            "write_file": (
                write_file,
                "Write content to a file, creating it if needed",
            ),
            "edit_file": (
                edit_file,
                "Edit a specific part of a file",
            ),
            "grep_search": (
                grep_search,
                "Search for text patterns in files using grep",
            ),
            "glob_search": (
                glob_search,
                "Find files matching a glob pattern",
            ),
            "get_current_time": (
                get_current_time,
                "Get the current date and time",
            ),
            "set_user_timezone": (
                set_user_timezone,
                "Set the user's timezone for time-related operations",
            ),
            "get_token_usage": (
                get_token_usage,
                "Get token usage statistics for the current session",
            ),
        }

        for tool_name, (tool_func, description) in tool_functions.items():
            try:
                tool_registry.register_tool_function(
                    tool_func,
                    name=tool_name,
                    description=description,
                )
            except Exception as e:
                logger.warning(
                    "Failed to register built-in tool '%s': %s",
                    tool_name,
                    e,
                )

        logger.info(
            "Registered %d built-in tools",
            len(tool_functions),
        )

    except ImportError as e:
        logger.warning(
            "Could not import built-in tools: %s",
            e,
        )


__all__ = [
    "ToolRegistry",
    "tool_registry",
    "get_tool_registry",
    "register_builtin_tools",
]