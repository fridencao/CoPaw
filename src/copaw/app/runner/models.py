# -*- coding: utf-8 -*-
"""Chat models for runner with UUID management.

This module provides chat models compatible with LangGraph.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Union
from uuid import uuid4

from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from ..channels.schema import DEFAULT_CHANNEL

# Type alias for messages (LangGraph compatible)
Message = Union[HumanMessage, AIMessage, BaseMessage]


class ChatSpec(BaseModel):
    """Chat specification with UUID identifier.

    Stored in Redis and can be persisted in JSON file.
    """

    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Chat UUID identifier",
    )
    name: str = Field(default="New Chat", description="Chat name")
    session_id: str = Field(
        ...,
        description="Session identifier (channel:user_id format)",
    )
    user_id: str = Field(..., description="User identifier")
    channel: str = Field(default=DEFAULT_CHANNEL, description="Channel name")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Chat creation timestamp",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Chat last update timestamp",
    )
    meta: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )
    status: str = Field(
        default="idle",
        description="Conversation status: idle or running",
    )


class ChatHistory(BaseModel):
    """Complete chat view with spec and state."""

    messages: List[BaseMessage] = Field(default_factory=list)
    status: str = Field(
        default="idle",
        description="Conversation status: idle or running",
    )


class ChatsFile(BaseModel):
    """Chat registry file for JSON repository.

    Stores chat_id (UUID) -> session_id mappings for persistence.
    """

    version: int = 1
    chats: list[ChatSpec] = Field(default_factory=list)