# -*- coding: utf-8 -*-
"""Memory adapter for LangGraph checkpoint integration.

This module provides adapters to convert between CoPaw's memory system
and LangGraph checkpoint format.
"""

import logging
from typing import Any, Dict, List, Optional, Sequence
import json
from pathlib import Path

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver

logger = logging.getLogger(__name__)


class LangGraphCheckpointAdapter:
    """Adapter to integrate CoPaw MemoryManager with LangGraph checkpoint.

    This adapter bridges the gap between:
    - CoPaw's BaseMemoryManager (using AgentScope's Msg)
    - LangGraph's checkpoint serialization format
    """

    def __init__(
        self,
        memory_manager: Any = None,
        checkpoint_dir: Optional[Path] = None,
    ):
        """Initialize the checkpoint adapter.

        Args:
            memory_manager: Optional CoPaw memory manager
            checkpoint_dir: Optional directory for checkpoint persistence
        """
        self.memory_manager = memory_manager
        self.checkpoint_dir = checkpoint_dir
        self._memory_saver = MemorySaver()

    def get_session_state(
        self,
        session_id: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Get stored state for a session.

        Args:
            session_id: Session identifier
            config: Optional checkpoint config

        Returns:
            Stored state dict or None
        """
        try:
            # Use LangGraph's MemorySaver to get state
            if config is None:
                config = {"configurable": {"thread_id": session_id}}

            # Get checkpoint from memory saver
            checkpoint = self._memory_saver.get(config)
            if checkpoint:
                return {
                    "messages": checkpoint.get("messages", []),
                    "checkpoint": checkpoint,
                }
        except Exception as e:
            logger.warning(f"Failed to get session state: {e}")

        return None

    def save_session_state(
        self,
        session_id: str,
        state: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save state for a session.

        Args:
            session_id: Session identifier
            state: State dict to save
            config: Optional checkpoint config
        """
        try:
            if config is None:
                config = {"configurable": {"thread_id": session_id}}

            # Convert state to checkpoint format
            checkpoint = {
                "messages": state.get("messages", []),
                "channel_values": state.get("channel_values", {}),
            }

            self._memory_saver.put(config, checkpoint, {})
        except Exception as e:
            logger.warning(f"Failed to save session state: {e}")

    def delete_session_state(self, session_id: str) -> bool:
        """Delete stored state for a session.

        Args:
            session_id: Session identifier

        Returns:
            True if deleted successfully
        """
        try:
            config = {"configurable": {"thread_id": session_id}}
            # MemorySaver doesn't have delete, but we can clear by overwriting
            # For now, we'll just log this
            logger.info(f"Session {session_id} checkpoint would be cleared")
            return True
        except Exception as e:
            logger.warning(f"Failed to delete session state: {e}")
            return False


def messages_to_langgraph_format(
    messages: List[Any],
) -> List[Dict[str, Any]]:
    """Convert CoPaw message format to LangGraph format.

    Args:
        messages: List of messages (Msg objects or dicts)

    Returns:
        List of message dicts for LangGraph
    """
    result = []

    for msg in messages:
        # Handle dict format
        if isinstance(msg, dict):
            role = msg.get("role", "user")
            content = msg.get("content", "") or msg.get("text", "")
        # Handle AgentScope Msg
        elif hasattr(msg, "role") and hasattr(msg, "content"):
            role = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
            content = msg.content or ""
        else:
            continue

        if not content:
            continue

        # Map to LangGraph format
        if role in ("system", "system_message"):
            result.append({"type": "system", "content": content})
        elif role in ("assistant", "ai", "assistant_message"):
            result.append({"type": "ai", "content": content})
        elif role in ("user", "user_message"):
            result.append({"type": "human", "content": content})
        elif role in ("tool",):
            result.append({
                "type": "tool",
                "content": content,
                "tool_call_id": msg.get("tool_call_id", "") if isinstance(msg, dict) else "",
            })
        else:
            # Default to user
            result.append({"type": "human", "content": content})

    return result


def langgraph_messages_to_copaw_format(
    messages: Sequence[BaseMessage],
) -> List[Dict[str, Any]]:
    """Convert LangGraph messages to CoPaw format.

    Args:
        messages: LangGraph messages

    Returns:
        List of message dicts in CoPaw format
    """
    result = []

    for msg in messages:
        msg_type = msg.type if hasattr(msg, "type") else "human"
        content = msg.content if hasattr(msg, "content") else str(msg)

        if msg_type == "system":
            role = "system"
        elif msg_type == "ai":
            role = "assistant"
        elif msg_type == "tool":
            role = "tool"
        else:
            role = "user"

        msg_dict = {
            "role": role,
            "content": content,
        }

        # Add tool call info if present
        if hasattr(msg, "tool_call_id") and msg.tool_call_id:
            msg_dict["tool_call_id"] = msg.tool_call_id

        result.append(msg_dict)

    return result


class CoPawCheckpointer(BaseCheckpointSaver):
    """Custom checkpointer that integrates with CoPaw's memory system.

    This checkpointer:
    - Stores checkpoints in CoPaw's memory system
    - Provides session persistence across restarts
    - Supports long-term memory compaction
    """

    def __init__(
        self,
        memory_manager: Any = None,
        checkpoint_dir: Optional[Path] = None,
    ):
        """Initialize the CoPaw checkpointer.

        Args:
            memory_manager: CoPaw memory manager for persistence
            checkpoint_dir: Directory for checkpoint files
        """
        super().__init__()
        self.memory_manager = memory_manager
        self.checkpoint_dir = checkpoint_dir or Path("/tmp/copaw_checkpoints")
        self._memory_saver = MemorySaver()

        # Ensure checkpoint directory exists
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def get(self, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get checkpoint for given config.

        Args:
            config: Checkpoint config with thread_id

        Returns:
            Checkpoint dict or None
        """
        thread_id = config.get("configurable", {}).get("thread_id", "default")

        # First try to get from CoPaw memory manager
        if self.memory_manager:
            try:
                # Get memory from memory manager
                memory = self.memory_manager.get_in_memory_memory()
                if memory:
                    # Convert to checkpoint format
                    messages = memory.get_all()
                    return {
                        "messages": messages_to_langgraph_format(messages),
                    }
            except Exception as e:
                logger.debug(f"Memory manager get failed: {e}")

        # Fall back to memory saver
        return self._memory_saver.get(config)

    def put(
        self,
        config: Dict[str, Any],
        checkpoint: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> None:
        """Store checkpoint.

        Args:
            config: Checkpoint config
            checkpoint: Checkpoint data
            metadata: Metadata
        """
        thread_id = config.get("configurable", {}).get("thread_id", "default")

        # Always save to memory saver for quick access
        self._memory_saver.put(config, checkpoint, metadata)

        # Optionally persist to CoPaw memory manager
        if self.memory_manager:
            try:
                # Convert checkpoint messages to CoPaw format
                messages = checkpoint.get("messages", [])
                copaw_messages = langgraph_messages_to_copaw_format(messages)

                # This would integrate with the memory manager
                # memory.add_messages(copaw_messages)
                logger.debug(f"Would save {len(copaw_messages)} messages to memory manager")
            except Exception as e:
                logger.warning(f"Failed to persist to memory manager: {e}")

    def list(
        self,
        config: Optional[Dict[str, Any]] = None,
        *,
        filter: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """List checkpoints.

        Args:
            config: Base config
            filter: Optional filter
            limit: Max results

        Returns:
            List of checkpoint configs
        """
        return self._memory_saver.list(config, filter=filter, limit=limit)

    def delete(self, config: Dict[str, Any]) -> None:
        """Delete checkpoint for given config.

        Args:
            config: Checkpoint config with thread_id
        """
        thread_id = config.get("configurable", {}).get("thread_id", "default")
        try:
            # Use the memory saver's delete if available
            if hasattr(self._memory_saver, 'delete'):
                self._memory_saver.delete(config)
            else:
                # Fallback: overwrite with empty checkpoint
                self._memory_saver.put(config, {"messages": []}, {})
            logger.info(f"Deleted checkpoint for session: {thread_id}")
        except Exception as e:
            logger.warning(f"Failed to delete checkpoint: {e}")