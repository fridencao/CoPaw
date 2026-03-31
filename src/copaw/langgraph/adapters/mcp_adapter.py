# -*- coding: utf-8 -*-
"""MCP client adapter for LangGraph tool integration.

This module provides adapters to convert MCP clients to LangGraph tools.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Sequence
from dataclasses import dataclass

from langchain_core.tools import BaseTool, Tool
from pydantic import BaseModel, Field

from agentscope.mcp import HttpStatefulClient, StdIOStatefulClient

logger = logging.getLogger(__name__)


class MCPClientAdapter:
    """Adapter to convert MCP clients to LangGraph tools.

    This adapter bridges the gap between:
    - CoPaw's MCPClientManager (using AgentScope MCP clients)
    - LangGraph's tool format (BaseTool)
    """

    def __init__(self, mcp_client: Any):
        """Initialize the MCP client adapter.

        Args:
            mcp_client: AgentScope MCP client instance
        """
        self.mcp_client = mcp_client
        self._tool_schemas: List[Dict[str, Any]] = []
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the adapter and fetch tool schemas."""
        if self._initialized:
            return

        try:
            # Connect if not already connected
            if not hasattr(self.mcp_client, "connected") or not self.mcp_client.connected:
                await self.mcp_client.connect()

            # Get available tools from the MCP server
            # This is typically done via a tools/list call
            tools_response = await self.mcp_client.call_tool("list_tools", {})
            if tools_response and hasattr(tools_response, "content"):
                for content in tools_response.content:
                    if hasattr(content, "text"):
                        import json
                        try:
                            tool_data = json.loads(content.text)
                            self._tool_schemas.append(tool_data)
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse tool schema: {content.text}")

            self._initialized = True
            logger.info(f"MCP client initialized with {len(self._tool_schemas)} tools")

        except Exception as e:
            logger.warning(f"Failed to initialize MCP client: {e}")
            # Don't fail hard - allow lazy initialization

    def get_tools(self) -> List[BaseTool]:
        """Get LangGraph tools from MCP client.

        Returns:
            List of BaseTool instances
        """
        tools = []

        for schema in self._tool_schemas:
            tool = self._create_tool_from_schema(schema)
            if tool:
                tools.append(tool)

        return tools

    def _create_tool_from_schema(self, schema: Dict[str, Any]) -> Optional[BaseTool]:
        """Create a LangGraph tool from MCP schema.

        Args:
            schema: Tool schema from MCP server

        Returns:
            BaseTool instance or None
        """
        try:
            name = schema.get("name", "")
            description = schema.get("description", "")
            input_schema = schema.get("inputSchema", schema.get("parameters", {}))

            if not name:
                return None

            # Create the tool function
            async def _run_mcp_tool(
                tool_name: str = name,
                arguments: Optional[Dict[str, Any]] = None,
            ) -> str:
                """Execute MCP tool call."""
                try:
                    result = await self.mcp_client.call_tool(
                        tool_name,
                        arguments or {},
                    )
                    # Convert result to string
                    if hasattr(result, "content"):
                        outputs = []
                        for content in result.content:
                            if hasattr(content, "text"):
                                outputs.append(content.text)
                            elif hasattr(content, "json"):
                                outputs.append(json.dumps(content.json()))
                            else:
                                outputs.append(str(content))
                        return "\n".join(outputs)
                    return str(result)
                except Exception as e:
                    logger.error(f"MCP tool call failed: {e}")
                    return f"Error: {str(e)}"

            # Create LangGraph tool
            tool = Tool(
                name=name,
                description=description,
                args_schema=self._create_input_model(name, input_schema),
                func=_run_mcp_tool,
            )

            return tool

        except Exception as e:
            logger.warning(f"Failed to create tool from schema: {e}")
            return None

    def _create_input_model(self, name: str, schema: Dict[str, Any]) -> type[BaseModel]:
        """Create a Pydantic input model from JSON schema.

        Args:
            name: Tool name
            schema: JSON schema for tool input

        Returns:
            Pydantic BaseModel class
        """
        properties = schema.get("properties", {})
        required = schema.get("required", [])

        fields = {}
        for prop_name, prop_def in properties.items():
            field_type = str
            if prop_def.get("type") == "integer":
                field_type = int
            elif prop_def.get("type") == "number":
                field_type = float
            elif prop_def.get("type") == "boolean":
                field_type = bool

            default = ... if prop_name in required else None
            description = prop_def.get("description", "")

            fields[prop_name] = Field(default=default, description=description)

        model_name = f"{name}Input"
        return type(model_name, (BaseModel,), fields)


def create_tools_from_mcp_clients(
    mcp_clients: List[Any],
) -> List[BaseTool]:
    """Create LangGraph tools from MCP clients.

    This is the main entry point for converting MCP clients
    to LangGraph-compatible tools.

    Args:
        mcp_clients: List of MCP client instances

    Returns:
        List of BaseTool instances
    """
    all_tools = []

    for client in mcp_clients:
        try:
            adapter = MCPClientAdapter(client)
            # Note: In production, we'd want to initialize asynchronously
            # For now, tools will be created lazily
            tools = adapter.get_tools()
            all_tools.extend(tools)
        except Exception as e:
            logger.warning(f"Failed to create tools from MCP client: {e}")

    return all_tools


class MCPClientManagerAdapter:
    """Adapter for CoPaw's MCPClientManager.

    This class wraps the MCPClientManager to provide LangGraph-compatible
    tool generation with automatic registration to the global tool registry.
    """

    def __init__(
        self,
        mcp_client_manager: Any,
        auto_register: bool = True,
    ):
        """Initialize the adapter.

        Args:
            mcp_client_manager: CoPaw's MCPClientManager instance
            auto_register: Whether to automatically register tools to registry
        """
        self._manager = mcp_client_manager
        self._adapters: Dict[str, MCPClientAdapter] = {}
        self._auto_register = auto_register

    async def get_tools(self) -> List[BaseTool]:
        """Get all tools from all MCP clients.

        Returns:
            Combined list of tools from all clients
        """
        all_tools = []

        try:
            # Get all active clients
            clients = await self._manager.get_clients()

            for client in clients:
                # Get or create adapter for this client
                client_id = id(client)
                if client_id not in self._adapters:
                    adapter = MCPClientAdapter(client)
                    await adapter.initialize()
                    self._adapters[client_id] = adapter
                else:
                    adapter = self._adapters[client_id]

                # Get tools from this client
                tools = adapter.get_tools()
                all_tools.extend(tools)

                # Auto-register tools to global registry
                if self._auto_register:
                    self._register_tools_to_registry(tools)

        except Exception as e:
            logger.warning(f"Failed to get MCP tools: {e}")

        return all_tools

    def _register_tools_to_registry(self, tools: List[BaseTool]) -> None:
        """Register tools to the global tool registry.

        Args:
            tools: List of tools to register
        """
        try:
            from ..tools import tool_registry

            for tool in tools:
                tool_registry.register_tool(tool)
                logger.debug(f"Registered MCP tool: {tool.name}")

        except Exception as e:
            logger.warning(f"Failed to register tools to registry: {e}")

    async def refresh_tools(self) -> List[BaseTool]:
        """Refresh tools from all clients (for hot-reload).

        Returns:
            Updated list of tools
        """
        # Clear cached adapters
        self._adapters.clear()

        # Get fresh tools
        return await self.get_tools()