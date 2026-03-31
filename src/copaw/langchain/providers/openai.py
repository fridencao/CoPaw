# -*- coding: utf-8 -*-
"""OpenAI provider adapter for LangChain."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from langchain_openai import ChatOpenAI
from langchain_core.callbacks import AsyncCallbackHandler

from copaw.providers import Provider
from copaw.providers.provider import ModelInfo

logger = logging.getLogger(__name__)

DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
CODING_DASHSCOPE_BASE_URL = "https://coding.dashscope.aliyuncs.com/v1"


class OpenAIAdapter:
    """Adapter to convert OpenAI Provider config to LangChain ChatOpenAI."""

    def __init__(self, provider: Provider):
        """Initialize adapter with a CoPaw Provider instance.

        Args:
            provider: OpenAI-compatible Provider instance
        """
        self.provider = provider

    def get_chat_model(
        self,
        model_id: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        callbacks: Optional[list[AsyncCallbackHandler]] = None,
        **kwargs: Any,
    ) -> ChatOpenAI:
        """Create a LangChain ChatOpenAI instance.

        Args:
            model_id: Model identifier
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            callbacks: List of callback handlers
            **kwargs: Additional arguments passed to ChatOpenAI

        Returns:
            ChatOpenAI instance
        """
        client_kwargs: Dict[str, Any] = {"base_url": self.provider.base_url}

        # Add custom headers for DashScope
        if self.provider.base_url == DASHSCOPE_BASE_URL:
            import json

            client_kwargs["default_headers"] = {
                "x-dashscope-agentapp": json.dumps(
                    {
                        "agentType": "CoPaw",
                        "deployType": "UnKnown",
                        "moduleCode": "model",
                        "agentCode": "UnKnown",
                    },
                    ensure_ascii=False,
                ),
            }
        elif self.provider.base_url == CODING_DASHSCOPE_BASE_URL:
            import json

            client_kwargs["default_headers"] = {
                "X-DashScope-Cdpl": json.dumps(
                    {
                        "agentType": "CoPaw",
                        "deployType": "UnKnown",
                        "moduleCode": "model",
                        "agentCode": "UnKnown",
                    },
                    ensure_ascii=False,
                ),
            }

        # Merge generate_kwargs from provider
        generate_kwargs = self.provider.generate_kwargs.copy()
        if temperature is not None:
            generate_kwargs["temperature"] = temperature
        if max_tokens is not None:
            generate_kwargs["max_tokens"] = max_tokens
        generate_kwargs.update(kwargs)

        return ChatOpenAI(
            model=model_id,
            api_key=self.provider.api_key,
            temperature=generate_kwargs.get("temperature", 0.7),
            max_tokens=generate_kwargs.get("max_tokens"),
            streaming=True,
            callbacks=callbacks,
            client_kwargs=client_kwargs,
            **generate_kwargs,
        )

    async def check_connection(self, timeout: float = 5) -> tuple[bool, str]:
        """Check if the provider is reachable."""
        try:
            from openai import APIError, AsyncOpenAI

            client = AsyncOpenAI(
                base_url=self.provider.base_url,
                api_key=self.provider.api_key,
                timeout=timeout,
            )
            await client.models.list(timeout=timeout)
            return True, ""
        except Exception as e:
            return False, str(e)

    async def fetch_models(self, timeout: float = 5) -> list[ModelInfo]:
        """Fetch available models."""
        try:
            from openai import APIError, AsyncOpenAI

            client = AsyncOpenAI(
                base_url=self.provider.base_url,
                api_key=self.provider.api_key,
                timeout=timeout,
            )
            payload = await client.models.list(timeout=timeout)
            models = []
            rows = getattr(payload, "data", [])
            for row in rows or []:
                model_id = str(getattr(row, "id", "") or "").strip()
                if not model_id:
                    continue
                model_name = (
                    str(getattr(row, "name", "") or model_id).strip() or model_id
                )
                models.append(ModelInfo(id=model_id, name=model_name))
            return models
        except Exception:
            return []


__all__ = ["OpenAIAdapter"]