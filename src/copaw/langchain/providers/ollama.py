# -*- coding: utf-8 -*-
"""Ollama provider adapter for LangChain."""

from __future__ import annotations

import logging
from typing import Any, Optional

from langchain_ollama import ChatOllama
from langchain_core.callbacks import AsyncCallbackHandler

from copaw.providers import Provider
from copaw.providers.provider import ModelInfo

logger = logging.getLogger(__name__)


class OllamaAdapter:
    """Adapter to convert Ollama Provider config to LangChain ChatOllama."""

    def __init__(self, provider: Provider):
        """Initialize adapter with a CoPaw Provider instance.

        Args:
            provider: Ollama Provider instance
        """
        self.provider = provider

    def get_chat_model(
        self,
        model_id: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        callbacks: Optional[list[AsyncCallbackHandler]] = None,
        **kwargs: Any,
    ) -> ChatOllama:
        """Create a LangChain ChatOllama instance.

        Args:
            model_id: Model identifier (e.g., 'llama3', 'qwen2.5')
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            callbacks: List of callback handlers
            **kwargs: Additional arguments passed to ChatOllama

        Returns:
            ChatOllama instance
        """
        # Extract host from base_url (e.g., "http://localhost:11434")
        base_url = self.provider.base_url or "http://localhost:11434"

        # Merge generate_kwargs from provider
        generate_kwargs = self.provider.generate_kwargs.copy()
        if temperature is not None:
            generate_kwargs["temperature"] = temperature
        if max_tokens is not None:
            generate_kwargs["max_tokens"] = max_tokens
        generate_kwargs.update(kwargs)

        model_kwargs = {
            "model": model_id,
            "base_url": base_url,
            "temperature": generate_kwargs.get("temperature", 0.7),
            "streaming": True,
            "callbacks": callbacks,
        }

        # Add optional parameters
        if "max_tokens" in generate_kwargs:
            model_kwargs["num_predict"] = generate_kwargs["max_tokens"]
        if "top_p" in generate_kwargs:
            model_kwargs["top_p"] = generate_kwargs["top_p"]
        if "top_k" in generate_kwargs:
            model_kwargs["top_k"] = generate_kwargs["top_k"]

        return ChatOllama(**model_kwargs)

    async def check_connection(self, timeout: float = 5) -> tuple[bool, str]:
        """Check if Ollama is reachable."""
        try:
            import httpx

            base_url = self.provider.base_url or "http://localhost:11434"
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(f"{base_url}/api/tags")
                if response.status_code == 200:
                    return True, ""
                return False, f"HTTP {response.status_code}"
        except Exception as e:
            return False, str(e)

    async def fetch_models(self, timeout: float = 5) -> list[ModelInfo]:
        """Fetch available models from Ollama."""
        try:
            import httpx

            base_url = self.provider.base_url or "http://localhost:11434"
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(f"{base_url}/api/tags")
                if response.status_code != 200:
                    return []

                data = response.json()
                models = []
                for model in data.get("models", []):
                    model_id = model.get("name", "")
                    if not model_id:
                        continue
                    # Handle model names with tags (e.g., "llama3:latest")
                    model_name = model_id.split(":")[0]
                    models.append(ModelInfo(id=model_id, name=model_name))
                return models
        except Exception:
            return []


__all__ = ["OllamaAdapter"]