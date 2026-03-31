# -*- coding: utf-8 -*-
"""Adapters for LangGraph integration.

This module provides adapters to bridge between CoPaw's existing components
and the LangGraph-based runner.
"""

from .channel_adapter import (
    ChannelMessageAdapter,
    create_messages_from_channel_payload,
)
from .memory_adapter import (
    LangGraphCheckpointAdapter,
    CoPawCheckpointer,
    messages_to_langgraph_format,
    langgraph_messages_to_copaw_format,
)
from .mcp_adapter import (
    MCPClientAdapter,
    MCPClientManagerAdapter,
    create_tools_from_mcp_clients,
)

__all__ = [
    # Channel adapter
    "ChannelMessageAdapter",
    "create_messages_from_channel_payload",
    # Memory adapter
    "LangGraphCheckpointAdapter",
    "CoPawCheckpointer",
    "messages_to_langgraph_format",
    "langgraph_messages_to_copaw_format",
    # MCP adapter
    "MCPClientAdapter",
    "MCPClientManagerAdapter",
    "create_tools_from_mcp_clients",
]