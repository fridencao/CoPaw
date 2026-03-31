# -*- coding: utf-8 -*-
"""Tools package for LangGraph integration."""

from .registry import ToolRegistry, tool_registry, register_builtin_tools

# Converter is deprecated - kept for backward compatibility
# Actual tools come from copaw.agents.tools
try:
    from .converter import (
        convert_agentscope_tool_to_langchain,
        convert_builtin_tools,
    )
    _CONVERTER_AVAILABLE = True
except ImportError:
    convert_agentscope_tool_to_langchain = None
    convert_builtin_tools = None
    _CONVERTER_AVAILABLE = False

__all__ = [
    "ToolRegistry",
    "tool_registry",
    "register_builtin_tools",
    "convert_agentscope_tool_to_langchain",
    "convert_builtin_tools",
]