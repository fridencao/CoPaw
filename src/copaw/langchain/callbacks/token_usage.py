# -*- coding: utf-8 -*-
"""Token usage callback handler for LangChain.

This module provides a callback handler that records token usage
similar to the original TokenRecordingModelWrapper.
"""

import logging
from typing import Any, Dict, Optional
from datetime import datetime

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import BaseMessage

logger = logging.getLogger(__name__)


class TokenUsageCallbackHandler(AsyncCallbackHandler):
    """Callback handler for recording token usage in LangChain.

    This handler tracks token usage from model responses and can
    be integrated with CoPaw's token usage tracking system.
    """

    def __init__(
        self,
        provider_id: str = "unknown",
        model_id: str = "unknown",
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ):
        """Initialize the callback handler.

        Args:
            provider_id: Provider identifier for tracking
            model_id: Model identifier for tracking
            session_id: Optional session ID
            user_id: Optional user ID
        """
        super().__init__()
        self.provider_id = provider_id
        self.model_id = model_id
        self.session_id = session_id
        self.user_id = user_id
        self._total_tokens = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._requests = 0

    def reset_counts(self) -> None:
        """Reset token counts for a new conversation."""
        self._total_tokens = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._requests = 0

    @property
    def total_tokens(self) -> int:
        """Get total tokens used."""
        return self._total_tokens

    @property
    def prompt_tokens(self) -> int:
        """Get prompt tokens used."""
        return self._prompt_tokens

    @property
    def completion_tokens(self) -> int:
        """Get completion tokens used."""
        return self._completion_tokens

    @property
    def request_count(self) -> int:
        """Get number of requests made."""
        return self._requests

    def get_usage_dict(self) -> Dict[str, Any]:
        """Get usage statistics as a dictionary.

        Returns:
            Dictionary with usage statistics
        """
        return {
            "provider_id": self.provider_id,
            "model_id": self.model_id,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "total_tokens": self._total_tokens,
            "prompt_tokens": self._prompt_tokens,
            "completion_tokens": self._completion_tokens,
            "requests": self._requests,
            "timestamp": datetime.utcnow().isoformat(),
        }

    async def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: list[BaseMessage],
        *,
        run_id: str,
        parent_run_id: Optional[str] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Run when chat model starts."""
        logger.debug(
            "TokenUsageCallback: on_chat_model_start for model %s",
            self.model_id,
        )

    async def on_chat_model_end(
        self,
        serialized: Dict[str, Any],
        response: Any,
        *,
        run_id: str,
        parent_run_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Run when chat model ends.

        Extract token usage from the response if available.
        """
        self._requests += 1

        # Try to extract token usage from response
        # Different providers return usage in different formats
        try:
            if hasattr(response, "usage") and response.usage:
                usage = response.usage
                if hasattr(usage, "total_tokens"):
                    self._total_tokens += usage.total_tokens or 0
                if hasattr(usage, "prompt_tokens"):
                    self._prompt_tokens += usage.prompt_tokens or 0
                if hasattr(usage, "completion_tokens"):
                    self._completion_tokens += usage.completion_tokens or 0
                elif hasattr(usage, "completion_tokens"):
                    # Some providers use different attribute names
                    self._completion_tokens += usage.completion_tokens or 0

                logger.debug(
                    "TokenUsageCallback: recorded usage - "
                    "total=%d, prompt=%d, completion=%d",
                    self._total_tokens,
                    self._prompt_tokens,
                    self._completion_tokens,
                )
        except Exception as e:
            logger.warning(
                "TokenUsageCallback: failed to extract token usage: %s",
                e,
            )

    async def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: str,
        parent_run_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Run when LLM errors."""
        logger.debug(
            "TokenUsageCallback: on_llm_error for model %s: %s",
            self.model_id,
            error,
        )


class TokenUsageRecorder:
    """Helper class to record token usage to CoPaw's tracking system.

    This can be used to integrate with CoPaw's existing token usage
    tracking infrastructure.
    """

    def __init__(self, provider_id: str):
        """Initialize the recorder.

        Args:
            provider_id: Provider identifier
        """
        self.provider_id = provider_id
        self._records: list[Dict[str, Any]] = []

    def add_usage(
        self,
        model_id: str,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> None:
        """Add a token usage record.

        Args:
            model_id: Model identifier
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            total_tokens: Total tokens used
            session_id: Optional session ID
            user_id: Optional user ID
        """
        record = {
            "provider_id": self.provider_id,
            "model_id": model_id,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "session_id": session_id,
            "user_id": user_id,
            "timestamp": datetime.utcnow().isoformat(),
        }
        self._records.append(record)

    def get_records(self) -> list[Dict[str, Any]]:
        """Get all recorded usage data."""
        return self._records.copy()

    def clear(self) -> None:
        """Clear all records."""
        self._records.clear()

    async def flush(self) -> None:
        """Flush records to the token usage tracking system.

        This method can be extended to write to CoPaw's database
        or send to an external tracking service.
        """
        # For now, just log the records
        logger.info(
            "TokenUsageRecorder: flushing %d records for provider %s",
            len(self._records),
            self.provider_id,
        )
        # TODO: Integrate with copaw.token_usage system
        self.clear()


__all__ = ["TokenUsageCallbackHandler", "TokenUsageRecorder"]