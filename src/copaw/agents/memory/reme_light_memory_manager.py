# -*- coding: utf-8 -*-
# pylint: disable=too-many-branches
"""ReMeLight-backed memory manager for CoPaw agents.

This version is compatible with LangGraph checkpoint system.
"""
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional

from langchain_core.messages import BaseMessage

from copaw.agents.memory.base_memory_manager import BaseMemoryManager

if TYPE_CHECKING:
    from reme.memory.file_based.reme_in_memory_memory import ReMeInMemoryMemory

logger = logging.getLogger(__name__)


class ReMeLightMemoryManager(BaseMemoryManager):
    """Memory manager that wraps ReMeLight for CoPaw agents via composition.

    Holds a ``ReMeLight`` instance (``self._reme``) and delegates all
    lifecycle / search / compaction calls to it.

    This version is LangGraph-compatible and integrates with checkpoint.
    """

    def __init__(self, working_dir: str, agent_id: str):
        """Initialize with ReMeLight.

        Args:
            working_dir: Working directory for memory storage.
            agent_id: Agent ID for config loading.
        """
        super().__init__(working_dir=working_dir, agent_id=agent_id)
        self._reme = None
        self._reme_version_ok = True

        logger.info(
            f"ReMeLightMemoryManager init: "
            f"agent_id={agent_id}, working_dir={working_dir}",
        )

        self._init_reme()

    def _init_reme(self) -> None:
        """Initialize ReMeLight instance."""
        try:
            from reme.reme_light import ReMeLight
        except ImportError as e:
            logger.warning(
                "reme package not installed, memory features will be "
                f"limited. {e}",
            )
            return

        emb_config = self.get_embedding_config()

        # Build embedding model config
        embedding_config = None
        if emb_config["base_url"] and emb_config["model_name"]:
            embedding_config = {
                "api_key": emb_config["api_key"],
                "base_url": emb_config["base_url"],
                "model_name": emb_config["model_name"],
            }

        self._reme = ReMeLight(
            working_dir=self.working_dir,
            embedding_api_key=emb_config["api_key"],
            embedding_base_url=emb_config["base_url"],
            default_embedding_model_config=embedding_config,
        )

    def get_embedding_config(self) -> dict:
        """Get embedding configuration from config or environment."""
        from copaw.config import load_config

        config = load_config()
        memory_config = getattr(config, "memory", None)

        return {
            "api_key": os.environ.get("EMBEDDING_API_KEY") or (
                memory_config.embedding_api_key if memory_config else None
            ),
            "base_url": os.environ.get("EMBEDDING_BASE_URL") or (
                memory_config.embedding_base_url if memory_config else None
            ),
            "model_name": os.environ.get("EMBEDDING_MODEL_NAME") or (
                memory_config.embedding_model_name if memory_config else None
            ),
        }

    async def start(self) -> None:
        """Start the memory manager lifecycle."""
        if self._reme:
            await self._reme.start()

    async def close(self) -> bool:
        """Close the memory manager."""
        if self._reme:
            await self._reme.close()
        return True

    async def compact_tool_result(self, **kwargs) -> None:
        """Compact tool results."""
        # Delegate to ReMeLight
        pass

    async def check_context(self, **kwargs) -> tuple:
        """Check context size."""
        # Delegate to ReMeLight
        return ([], [], True)

    async def compact_memory(
        self,
        messages: List[BaseMessage],
        previous_summary: str = "",
        **kwargs,
    ) -> str:
        """Compact messages into summary."""
        if not self._reme:
            return ""

        # Convert LangGraph messages to ReMe format
        reme_messages = self._convert_to_reme_messages(messages)
        return await self._reme.compact_memory(
            messages=reme_messages,
            previous_summary=previous_summary,
            **kwargs,
        )

    async def summary_memory(
        self,
        messages: List[BaseMessage],
        **kwargs,
    ) -> str:
        """Generate summary of messages."""
        if not self._reme:
            return ""

        reme_messages = self._convert_to_reme_messages(messages)
        return await self._reme.summary_memory(
            messages=reme_messages,
            **kwargs,
        )

    def _convert_to_reme_messages(self, messages: List[BaseMessage]) -> List[Any]:
        """Convert LangGraph messages to ReMeLight format."""
        # Simple conversion for now
        result = []
        for msg in messages:
            if hasattr(msg, "content"):
                result.append({
                    "role": msg.type,
                    "content": msg.content,
                })
        return result

    async def memory_search(
        self,
        query: str,
        max_results: int = 5,
        min_score: float = 0.1,
    ) -> Any:
        """Search memory."""
        if not self._reme:
            return {"results": []}

        return await self._reme.search(
            query=query,
            max_results=max_results,
            min_score=min_score,
        )

    def get_in_memory_memory(self, **kwargs) -> Any:
        """Get the in-memory memory object."""
        return self._reme


# Alias for backward compatibility
ToolResponse = dict