# -*- coding: utf-8 -*-
"""Channel message adapter for LangGraph integration.

This module provides adapters to convert between Channel message formats
(AgentRequest/AgentResponse) and LangGraph message formats (HumanMessage/AIMessage).
"""

from typing import Any, Dict, List, Optional
import logging

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from agentscope_runtime.engine.schemas.agent_schemas import (
    AgentRequest,
    AgentResponse,
    Message,
    Role,
    MessageType,
    ContentType,
    TextContent,
)

logger = logging.getLogger(__name__)


class ChannelMessageAdapter:
    """Adapter to convert between Channel and LangGraph message formats.

    This adapter bridges the gap between:
    - AgentScope's AgentRequest/AgentResponse (used by Channels)
    - LangGraph's HumanMessage/AIMessage (used by LangGraph)
    """

    @staticmethod
    def agent_request_to_langgraph_messages(
        request: AgentRequest,
    ) -> List[Dict[str, Any]]:
        """Convert AgentRequest to LangGraph message format.

        Args:
            request: AgentRequest from Channel

        Returns:
            List of message dicts compatible with LangGraph
        """
        messages = []

        # Add system prompt if present
        if hasattr(request, "system") and request.system:
            messages.append({
                "type": "system",
                "content": request.system,
            })

        # Process input messages
        if hasattr(request, "input") and request.input:
            for msg in request.input:
                msg_dict = ChannelMessageAdapter._message_to_dict(msg)
                if msg_dict:
                    messages.append(msg_dict)

        return messages

    @staticmethod
    def _message_to_dict(msg: Message) -> Optional[Dict[str, Any]]:
        """Convert a Message to dict format for LangGraph.

        Args:
            msg: Message from AgentRequest

        Returns:
            Dict with 'type' and 'content' keys
        """
        if not hasattr(msg, "role") or not hasattr(msg, "content"):
            return None

        # Convert role
        role_map = {
            Role.SYSTEM: "system",
            Role.USER: "user",
            Role.ASSISTANT: "assistant",
            Role.TOOL: "tool",
        }
        role = role_map.get(msg.role, "user")

        # Convert content
        content = ""
        if msg.content:
            for part in msg.content:
                if hasattr(part, "type") and part.type == ContentType.TEXT:
                    content += getattr(part, "text", "") or ""
                elif isinstance(part, dict):
                    if part.get("type") == "text":
                        content += part.get("text", "") or ""

        if not content:
            return None

        return {
            "type": role,
            "content": content,
        }

    @staticmethod
    def langgraph_messages_to_agent_response(
        messages: List[Any],
        session_id: str,
        user_id: str,
    ) -> AgentResponse:
        """Convert LangGraph messages to AgentResponse for Channel.

        Args:
            messages: List of LangGraph messages
            session_id: Session identifier
            user_id: User identifier

        Returns:
            AgentResponse compatible with Channel
        """
        output_messages = []

        for msg in messages:
            if hasattr(msg, "type"):
                role = Role.ASSISTANT if msg.type == "ai" else Role.USER
            else:
                role = Role.ASSISTANT

            content = []
            msg_content = getattr(msg, "content", "") or ""
            if msg_content:
                content.append(TextContent(
                    type=ContentType.TEXT,
                    text=msg_content,
                ))

            output_messages.append(Message(
                type=MessageType.MESSAGE,
                role=role,
                content=content,
            ))

        return AgentResponse(
            session_id=session_id,
            user_id=user_id,
            output=output_messages,
        )

    @staticmethod
    def extract_user_input(messages: Any) -> str:
        """Extract user input from messages (AgentRequest or list of messages).

        Args:
            messages: AgentRequest or list of message dicts

        Returns:
            User input string
        """
        # If it's an AgentRequest
        if hasattr(messages, "input") and messages.input:
            first_msg = messages.input[0]
            if hasattr(first_msg, "content") and first_msg.content:
                for part in first_msg.content:
                    if hasattr(part, "type") and part.type == ContentType.TEXT:
                        return getattr(part, "text", "") or ""
                    elif isinstance(part, dict) and part.get("type") == "text":
                        return part.get("text", "") or ""

        # If it's a list of messages
        if isinstance(messages, list):
            for msg in messages:
                msg_type = getattr(msg, "type", None) or msg.get("type", None)
                if msg_type == "human" or msg_type == "user":
                    content = getattr(msg, "content", None) or msg.get("content", "")
                    if content:
                        return content if isinstance(content, str) else str(content)

        return ""


def create_messages_from_channel_payload(payload: Any) -> List[Dict[str, Any]]:
    """Create LangGraph-compatible messages from Channel payload.

    This is the main entry point for converting Channel payloads
    to LangGraph message format.

    Args:
        payload: Either AgentRequest or dict with content_parts

    Returns:
        List of message dicts for LangGraph
    """
    # If it's already an AgentRequest
    if hasattr(payload, "session_id") and hasattr(payload, "input"):
        return ChannelMessageAdapter.agent_request_to_langgraph_messages(payload)

    # If it's a dict (native channel payload)
    if isinstance(payload, dict):
        content_parts = payload.get("content_parts", [])
        if content_parts:
            # Get text content
            text = ""
            for part in content_parts:
                if isinstance(part, dict) and part.get("type") == "text":
                    text = part.get("text", "") or ""
                elif hasattr(part, "type") and part.type == "text":
                    text = getattr(part, "text", "") or ""

            if text:
                return [{"type": "human", "content": text}]

    return []