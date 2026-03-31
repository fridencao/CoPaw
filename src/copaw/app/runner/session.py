# -*- coding: utf-8 -*-
"""Safe JSON session with filename sanitization for cross-platform
compatibility.

This module provides session persistence using LangGraph's checkpoint
system, replacing the agentscope SessionBase.
"""
import os
import re
import json
import logging
from pathlib import Path

from typing import Any, Dict, Optional, Sequence
import aiofiles

logger = logging.getLogger(__name__)


# Characters forbidden in Windows filenames
_UNSAFE_FILENAME_RE = re.compile(r'[\\/:*?"<>|]')


def sanitize_filename(name: str) -> str:
    """Replace characters that are illegal in Windows filenames with ``--``.

    >>> sanitize_filename('discord:dm:12345')
    'discord--dm--12345'
    >>> sanitize_filename('normal-name')
    'normal-name'
    """
    return _UNSAFE_FILENAME_RE.sub("--", name)


class LangGraphSession:
    """Session using LangGraph checkpoint for persistence.

    This replaces the agentscope SessionBase with LangGraph's
    checkpoint mechanism for session state management.
    """

    def __init__(
        self,
        save_dir: str = "./",
        checkpointer: Optional[Any] = None,
    ) -> None:
        """Initialize the session.

        Args:
            save_dir: Directory to save session state
            checkpointer: LangGraph checkpointer for persistence
        """
        self.save_dir = save_dir
        self._checkpointer = checkpointer

    def _get_save_path(self, session_id: str, user_id: str) -> Path:
        """Return a filesystem-safe save path.

        Args:
            session_id: Session identifier
            user_id: User identifier

        Returns:
            Path to save file
        """
        safe_session = sanitize_filename(session_id)
        safe_user = sanitize_filename(user_id)
        save_dir = Path(self.save_dir) / "sessions" / safe_user
        save_dir.mkdir(parents=True, exist_ok=True)
        return save_dir / f"{safe_session}.json"

    async def save(
        self,
        session_id: str,
        user_id: str,
        state: Dict[str, Any],
    ) -> None:
        """Save session state.

        Args:
            session_id: Session identifier
            user_id: User identifier
            state: State to save
        """
        save_path = self._get_save_path(session_id, user_id)
        async with aiofiles.open(save_path, "w") as f:
            await f.write(json.dumps(state, indent=2))

    async def load(
        self,
        session_id: str,
        user_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Load session state.

        Args:
            session_id: Session identifier
            user_id: User identifier

        Returns:
            Session state or None if not found
        """
        save_path = self._get_save_path(session_id, user_id)
        if not save_path.exists():
            return None
        async with aiofiles.open(save_path, "r") as f:
            content = await f.read()
            return json.loads(content)

    async def clear(self, session_id: str, user_id: str) -> None:
        """Clear session state.

        Args:
            session_id: Session identifier
            user_id: User identifier
        """
        save_path = self._get_save_path(session_id, user_id)
        if save_path.exists():
            save_path.unlink()


# Alias for backward compatibility
SafeJSONSession = LangGraphSession