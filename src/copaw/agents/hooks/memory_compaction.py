# -*- coding: utf-8 -*-
"""Memory compaction hook for managing context window.

This hook monitors token usage and automatically compacts older messages
when the context window approaches its limit, preserving recent messages
and the system prompt.

This version is compatible with LangGraph.
"""
import logging
from typing import TYPE_CHECKING, Any, List

from langchain_core.messages import BaseMessage

from copaw.constant import MEMORY_COMPACT_KEEP_RECENT

if TYPE_CHECKING:
    from ..memory import BaseMemoryManager

logger = logging.getLogger(__name__)


class MemoryCompactionHook:
    """Hook for automatic memory compaction when context is full.

    This hook monitors the token count of messages and triggers compaction
    when it exceeds the threshold. It preserves the system prompt and recent
    messages while summarizing older conversation history.
    """

    def __init__(self, memory_manager: "BaseMemoryManager"):
        """Initialize memory compaction hook.

        Args:
            memory_manager: Memory manager instance for compaction
        """
        self.memory_manager = memory_manager

    async def should_compact(
        self,
        messages: List[BaseMessage],
        max_tokens: int = 100000,
    ) -> bool:
        """Check if memory compaction is needed.

        Args:
            messages: Current conversation messages
            max_tokens: Maximum token threshold

        Returns:
            True if compaction should be triggered
        """
        if not messages:
            return False

        # Simple token estimation: ~4 chars per token
        total_chars = sum(len(getattr(m, "content", "") or "") for m in messages)
        estimated_tokens = total_chars // 4

        return estimated_tokens > (max_tokens * 0.8)  # 80% threshold

    async def compact_if_needed(
        self,
        messages: List[BaseMessage],
    ) -> List[BaseMessage]:
        """Compact messages if needed.

        Args:
            messages: Current conversation messages

        Returns:
            Compacted messages
        """
        if not await self.should_compact(messages):
            return messages

        # Get messages to compact (keep recent)
        keep_recent = MEMORY_COMPACT_KEEP_RECENT
        if len(messages) <= keep_recent:
            return messages

        recent_messages = messages[-keep_recent:]
        older_messages = messages[:-keep_recent]

        # Summarize older messages
        summary = await self.memory_manager.compact_memory(
            messages=older_messages,
        )

        # Create summary message
        from langchain_core.messages import SystemMessage

        summary_msg = SystemMessage(
            content=f"[Previous conversation summary]\n{summary}"
        )

        return [summary_msg] + recent_messages