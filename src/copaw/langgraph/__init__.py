# -*- coding: utf-8 -*-
"""LangGraph core module for CoPaw.

This module provides LangGraph-based agent implementation,
replacing the AgentScope ReActAgent.
"""

from .graph import create_react_graph
from .state import AgentState, get_initial_state

# Import nodes for direct access
from .nodes import (
    reasoning_node,
    check_tools_node,
    acting_node,
    observation_node,
)

# Import tools
from .tools import ToolRegistry, tool_registry, register_builtin_tools

# Import adapters
from .adapters import (
    ChannelMessageAdapter,
    create_messages_from_channel_payload,
    LangGraphCheckpointAdapter,
    CoPawCheckpointer,
    MCPClientAdapter,
    MCPClientManagerAdapter,
    create_tools_from_mcp_clients,
)

# Import runner and adapter
from .runner import LangGraphRunner
from .adapter import LangGraphRunnerAdapter, create_langgraph_runner_adapter

# Import prompts
from .prompts import build_system_prompt, build_system_prompt_from_files

# Import factory
from .factory import LangGraphRunnerFactory, get_langgraph_runner_factory

# Import workspace
from .workspace import LangGraphWorkspace, create_langgraph_workspace

__all__ = [
    # Graph and state
    "create_react_graph",
    "AgentState",
    "get_initial_state",
    # Nodes
    "reasoning_node",
    "check_tools_node",
    "acting_node",
    "observation_node",
    # Tools
    "ToolRegistry",
    "tool_registry",
    "register_builtin_tools",
    # Adapters
    "ChannelMessageAdapter",
    "create_messages_from_channel_payload",
    "LangGraphCheckpointAdapter",
    "CoPawCheckpointer",
    "MCPClientAdapter",
    "MCPClientManagerAdapter",
    "create_tools_from_mcp_clients",
    # Runner
    "LangGraphRunner",
    "LangGraphRunnerAdapter",
    "create_langgraph_runner_adapter",
    # Prompts
    "build_system_prompt",
    "build_system_prompt_from_files",
    # Factory
    "LangGraphRunnerFactory",
    "get_langgraph_runner_factory",
    # Workspace
    "LangGraphWorkspace",
    "create_langgraph_workspace",
]