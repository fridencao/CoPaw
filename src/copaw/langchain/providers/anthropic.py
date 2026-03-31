# -*- coding: utf-8 -*-
"""Anthropic provider adapter for LangChain."""

from __future__ import annotations

import logging
from typing import Any, Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.callbacks import AsyncCallbackHandler

from copaw.providers import Provider
from copaw.providers.provider import ModelInfo

logger = logging.getLogger(__name__)


class AnthropicAdapter:
    """Adapter to convert Anthropic Provider config to LangChain ChatAnthropic."""

    def __init__(self, provider: Provider):
        """Initialize adapter with a CoPaw Provider instance.

        Args:
            provider: Anthropic Provider instance
        """
        self.provider = provider

    def get_chat_model(
        self,
        model_id: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        callbacks: Optional[list[AsyncCallbackHandler]] = None,
        **kwargs: Any,
    ) -> ChatAnthropic:
        """Create a LangChain ChatAnthropic instance.

        Args:
            model_id: Model identifier (e.g., 'claude-3-5-sonnet-20241022')
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            callbacks: List of callback handlers
            **kwargs: Additional arguments passed to ChatAnthropic

        Returns:
            ChatAnthropic instance
        """
        # Build the base URL if custom endpoint is set
        base_url = self.provider.base_url if self.provider.base_url else None

        # Merge generate_kwargs from provider
        generate_kwargs = self.provider.generate_kwargs.copy()
        if temperature is not None:
            generate_kwargs["temperature"] = temperature
        if max_tokens is not None:
            generate_kwargs["max_tokens"] = max_tokens
        generate_kwargs.update(kwargs)

        # Build the model kwargs
        model_kwargs = {
            "model": model_id,
            "api_key": self.provider.api_key,
            "temperature": generate_kwargs.get("temperature", 0.7),
            "max_tokens": generate_kwargs.get("max_tokens", 4096),
            "streaming": True,
            "callbacks": callbacks,
        }

        if base_url:
            model_kwargs["base_url"] = base_url

        return ChatAnthropic(**model_kwargs)

    async def check_connection(self, timeout: float = 5) -> tuple[bool, str]:
        """Check if the provider is reachable."""
        try:
            import anthropic

            client = anthropic.AsyncAnthropic(
                api_key=self.provider.api_key,
                base_url=self.provider.base_url or anthropic.DEFAULT_API_URL,
                timeout=timeout,
            )
            await client.models.list()
            return True, ""
        except Exception as e:
            return False, str(e)

    async def fetch_models(self, timeout: float = 5) -> list[ModelInfo]:
        """Fetch available models."""
        try:
            import anthropic

            client = anthropic.AsyncAnthropic(
                api_key=self.provider.api_key,
                base_url=self.provider.base_url or anthropic.DEFAULT_API_URL,
                timeout=timeout,
            )
            payload = await client.models.list()
            models = []

            # Handle different response formats
            if hasattr(payload, "data"):
                rows = payload.data
            elif isinstance(payload, dict):
                rows = payload.get("data", [])
            else:
                rows = []

            for row in rows or []:
                model_id = str(getattr(row, "id", "") or "").strip()
                if not model_id:
                    continue
                model_name = str(
                    getattr(row, "display_name", "") or model_id,
                ).strip()
                models.append(ModelInfo(id=model_id, name=model_name))
            return models
        except Exception:
            return []


__all__ = ["AnthropicAdapter"]