# -*- coding: utf-8 -*-
"""Google Gemini provider adapter for LangChain."""

from __future__ import annotations

import logging
from typing import Any, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.callbacks import AsyncCallbackHandler

from copaw.providers import Provider
from copaw.providers.provider import ModelInfo

logger = logging.getLogger(__name__)


class GeminiAdapter:
    """Adapter to convert Google Gemini Provider config to LangChain."""

    def __init__(self, provider: Provider):
        """Initialize adapter with a CoPaw Provider instance.

        Args:
            provider: Google Gemini Provider instance
        """
        self.provider = provider

    def get_chat_model(
        self,
        model_id: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        callbacks: Optional[list[AsyncCallbackHandler]] = None,
        **kwargs: Any,
    ) -> ChatGoogleGenerativeAI:
        """Create a LangChain ChatGoogleGenerativeAI instance.

        Args:
            model_id: Model identifier (e.g., 'gemini-2.0-flash')
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            callbacks: List of callback handlers
            **kwargs: Additional arguments passed to ChatGoogleGenerativeAI

        Returns:
            ChatGoogleGenerativeAI instance
        """
        # Build connection params
        model_kwargs = {
            "model": model_id,
            "temperature": temperature,
            "streaming": True,
            "callbacks": callbacks,
        }

        if max_tokens is not None:
            model_kwargs["max_output_tokens"] = max_tokens

        # Handle API key - either from provider or use the google-genai package
        if self.provider.api_key:
            model_kwargs["google_api_key"] = self.provider.api_key

        # Handle custom base URL (for compatible endpoints)
        if self.provider.base_url:
            model_kwargs["base_url"] = self.provider.base_url

        # Merge generate_kwargs from provider
        generate_kwargs = self.provider.generate_kwargs.copy()
        generate_kwargs.update(kwargs)

        # Apply generate_kwargs
        for key in ["temperature", "top_p", "top_k"]:
            if key in generate_kwargs:
                model_kwargs[key] = generate_kwargs[key]

        return ChatGoogleGenerativeAI(**model_kwargs)

    async def check_connection(self, timeout: float = 5) -> tuple[bool, str]:
        """Check if the provider is reachable."""
        try:
            from google.genai import errors as genai_errors

            # For Gemini, we can try to list models
            # This is a simplified check
            if self.provider.api_key:
                return True, ""
            return False, "No API key configured"
        except Exception as e:
            return False, str(e)

    async def fetch_models(self, timeout: float = 5) -> list[ModelInfo]:
        """Fetch available models.

        Note: Google AI API doesn't have a standard models list endpoint
        in the same way as OpenAI. We return known model IDs.
        """
        # Return known Gemini models
        known_models = [
            ModelInfo(id="gemini-2.0-flash", name="Gemini 2.0 Flash"),
            ModelInfo(id="gemini-2.0-flash-lite", name="Gemini 2.0 Flash Lite"),
            ModelInfo(id="gemini-1.5-pro", name="Gemini 1.5 Pro"),
            ModelInfo(id="gemini-1.5-flash", name="Gemini 1.5 Flash"),
            ModelInfo(id="gemini-1.5-flash-8b", name="Gemini 1.5 Flash 8B"),
            ModelInfo(id="gemini-pro", name="Gemini Pro (Legacy)"),
            ModelInfo(id="gemini-pro-vision", name="Gemini Pro Vision (Legacy)"),
        ]
        return known_models


__all__ = ["GeminiAdapter"]