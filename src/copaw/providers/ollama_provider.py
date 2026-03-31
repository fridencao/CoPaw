# -*- coding: utf-8 -*-
"""An OpenAI provider implementation."""

from __future__ import annotations

import os
from typing import Any

from openai import AsyncOpenAI

from copaw.providers.provider import ModelInfo
from copaw.providers.openai_provider import OpenAIProvider


class OllamaProvider(OpenAIProvider):
    """Provider implementation for Ollama local LLM hosting platform."""

    @staticmethod
    def _normalize_base_url(base_url: str) -> str:
        normalized_base_url = (base_url or "").rstrip("/")
        if normalized_base_url.endswith("/v1"):
            # For backwards compatibility, if the URL ends with /v1,
            # we strip it to get the Ollama server base URL.
            normalized_base_url = normalized_base_url[:-3].rstrip("/")
        return normalized_base_url

    def _openai_compatible_base_url(self) -> str:
        return (
            self._normalize_base_url(
                self.base_url,  # type: ignore [has-type]
            )
            + "/v1"
        )

    def model_post_init(self, __context: Any) -> None:
        if not self.base_url:  # type: ignore
            self.base_url = (
                os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434"
            )
        self.base_url = self._normalize_base_url(self.base_url)

    def update_config(self, config: dict[str, Any]) -> None:
        super().update_config(config)
        self.base_url = self._normalize_base_url(self.base_url)

    def _client(self, timeout: float = 5) -> AsyncOpenAI:
        return AsyncOpenAI(
            base_url=self._openai_compatible_base_url(),
            api_key=self.api_key,
            timeout=timeout,
        )

    async def add_model(
        self,
        model_info: ModelInfo,
        target: str = "models",
        timeout: float = 36000,
    ) -> tuple[bool, str]:
        raise NotImplementedError(
            "Please add models directly in Ollama or use "
            "`ollama pull <model>` CLI command.",
        )

    async def delete_model(
        self,
        model_id: str,
        timeout: float = 60,
    ) -> tuple[bool, str]:
        raise NotImplementedError(
            "Please delete models directly in Ollama or use "
            "`ollama rm <model>` CLI command.",
        )

    def get_chat_model_instance(self, model_id: str) -> Any:
        """Get a chat model instance (deprecated, use langchain factory instead)."""
        # For compatibility only - actual models use langchain
        return None

    def get_available_models(self) -> List[ModelInfo]:
        """Get available models (stub for compatibility)."""
        return [ModelInfo(id="llama2", name="Llama 2")]

    def get_default_model(self) -> ModelInfo:
        """Get default model (stub for compatibility)."""
        return ModelInfo(id="llama2", name="Llama 2")
