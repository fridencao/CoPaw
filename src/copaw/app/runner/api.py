# -*- coding: utf-8 -*-
"""Chat management API."""
from __future__ import annotations
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from .session import LangGraphSession as SafeJSONSession
from .manager import ChatManager
from .models import (
    ChatSpec,
    ChatHistory,
)
from .utils import langgraph_msg_to_message as agentscope_msg_to_message


router = APIRouter(prefix="/chats", tags=["chats"])


async def get_workspace(request: Request):
    """Get the workspace for the active agent."""
    from ..agent_context import get_agent_for_request

    return await get_agent_for_request(request)


async def get_chat_manager(
    request: Request,
) -> ChatManager:
    """Get the chat manager for the active agent.

    Args:
        request: FastAPI request object

    Returns:
        ChatManager instance for the specified agent

    Raises:
        HTTPException: If manager is not initialized
    """
    workspace = await get_workspace(request)
    return workspace.chat_manager


async def get_session(
    request: Request,
) -> SafeJSONSession:
    """Get the session for the active agent.

    Args:
        request: FastAPI request object

    Returns:
        Session instance
    """
    workspace = await get_workspace(request)
    return workspace.session


# ===== Chat CRUD API Endpoints =====


@router.get("", response_model=list[ChatSpec])
async def list_chats(
    request: Request,
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    channel: Optional[str] = Query(None, description="Filter by channel"),
) -> list[ChatSpec]:
    """List all chat sessions, with optional filters."""
    manager = await get_chat_manager(request)
    return await manager.list_chats(user_id=user_id, channel=channel)


@router.post("", response_model=ChatSpec, status_code=201)
async def create_chat(
    request: Request,
    name: str = Query("New Chat", description="Chat name"),
    session_id: Optional[str] = Query(None, description="Session ID"),
    user_id: str = Query("default", description="User ID"),
    channel: str = Query("console", description="Channel"),
) -> ChatSpec:
    """Create a new chat session."""
    manager = await get_chat_manager(request)
    chat = ChatSpec(
        id=str(uuid4()),
        name=name,
        session_id=session_id or f"{channel}:{user_id}",
        user_id=user_id,
        channel=channel,
    )
    return await manager.create_chat(chat)


@router.get("/{chat_id}", response_model=ChatSpec)
async def get_chat(
    request: Request,
    chat_id: str,
) -> ChatSpec:
    """Get a specific chat by ID."""
    manager = await get_chat_manager(request)
    chat = await manager.get_chat(chat_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    return chat


@router.put("/{chat_id}", response_model=ChatSpec)
async def update_chat(
    request: Request,
    chat_id: str,
    name: Optional[str] = Query(None, description="Chat name"),
    status: Optional[str] = Query(None, description="Chat status"),
) -> ChatSpec:
    """Update a chat session."""
    manager = await get_chat_manager(request)
    chat = await manager.get_chat(chat_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    if name is not None:
        chat.name = name
    if status is not None:
        chat.status = status

    return await manager.update_chat(chat)


@router.delete("/{chat_id}", status_code=204)
async def delete_chat(
    request: Request,
    chat_id: str,
) -> None:
    """Delete a chat session."""
    manager = await get_chat_manager(request)
    deleted = await manager.delete_chats([chat_id])
    if not deleted:
        raise HTTPException(status_code=404, detail="Chat not found")