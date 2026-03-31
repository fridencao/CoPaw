# -*- coding: utf-8 -*-
"""Factory for creating LangChain chat models from CoPaw providers.

This module provides a unified factory for creating LangChain-compatible
chat model instances based on CoPaw provider configurations.
"""

import logging
from typing import Any, Optional, Tuple

from langchain_core.language_models import BaseChatModel
from langchain_core.callbacks import AsyncCallbackHandler

from copaw.providers import ProviderManager
from copaw.providers.provider import ModelInfo, Provider
from .providers.openai import OpenAIAdapter
from .providers.anthropic import AnthropicAdapter
from .providers.gemini import GeminiAdapter
from .providers.ollama import OllamaAdapter

logger = logging.getLogger(__name__)

# Mapping from provider ID prefix to adapter class
_PROVIDER_ADAPTER_MAP = {
    "openai": OpenAIAdapter,
    "anthropic": AnthropicAdapter,
    "google": GeminiAdapter,
    "gemini": GeminiAdapter,
    "ollama": OllamaAdapter,
    # Add more adapters as needed
}


def get_provider_adapter(provider_id: str) -> Optional[type]:
    """Get the appropriate adapter class for a provider.

    Args:
        provider_id: The provider identifier

    Returns:
        Adapter class or None if not found
    """
    # Try exact match first
    if provider_id in _PROVIDER_ADAPTER_MAP:
        return _PROVIDER_ADAPTER_MAP[provider_id]

    # Try prefix matching
    for prefix, adapter in _PROVIDER_ADAPTER_MAP.items():
        if provider_id.startswith(prefix):
            return adapter

    return None


def create_langchain_model(
    provider_id: str,
    model_id: str,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    callbacks: Optional[list[AsyncCallbackHandler]] = None,
    **kwargs: Any,
) -> Tuple[BaseChatModel, Any]:
    """Factory method to create a LangChain model from provider config.

    This function bridges the CoPaw Provider system with LangChain,
    using the existing provider configurations.

    Args:
        provider_id: Provider identifier (e.g., 'openai', 'anthropic')
        model_id: Model identifier
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        callbacks: List of callback handlers for LangChain
        **kwargs: Additional model configuration

    Returns:
        Tuple of (langchain_model, formatter_placeholder)
        The formatter is a placeholder for now - LangChain handles
        message formatting internally.

    Example:
        >>> model, formatter = create_langchain_model(
        ...     provider_id="openai",
        ...     model_id="gpt-4o",
        ...     temperature=0.7
        ... )
    """
    # Get provider from CoPaw's ProviderManager
    manager = ProviderManager.get_instance()
    provider = manager.get_provider(provider_id)

    if provider is None:
        raise ValueError(f"Provider '{provider_id}' not found")

    # Get appropriate adapter
    adapter_class = get_provider_adapter(provider_id)
    if adapter_class is None:
        # Fallback: try to use OpenAI adapter for unknown providers
        # (many providers are OpenAI-compatible)
        logger.warning(
            "No specific adapter for provider '%s', falling back to OpenAI",
            provider_id,
        )
        adapter_class = OpenAIAdapter

    adapter = adapter_class(provider)

    # Create the LangChain model
    langchain_model = adapter.get_chat_model(
        model_id=model_id,
        temperature=temperature,
        max_tokens=max_tokens,
        callbacks=callbacks,
        **kwargs,
    )

    # Return model with a simple formatter placeholder
    # LangChain handles message formatting internally
    formatter = _SimpleFormatter()

    return langchain_model, formatter


def create_langchain_model_from_provider(
    provider: Provider,
    model_id: str,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    callbacks: Optional[list[AsyncCallbackHandler]] = None,
    **kwargs: Any,
) -> Tuple[BaseChatModel, Any]:
    """Create a LangChain model directly from a Provider instance.

    Args:
        provider: CoPaw Provider instance
        model_id: Model identifier
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        callbacks: List of callback handlers
        **kwargs: Additional model configuration

    Returns:
        Tuple of (langchain_model, formatter_placeholder)
    """
    # Get appropriate adapter based on provider ID
    adapter_class = get_provider_adapter(provider.id)
    if adapter_class is None:
        # Fallback to OpenAI for compatible APIs
        adapter_class = OpenAIAdapter
        logger.warning(
            "No specific adapter for provider '%s', using OpenAI adapter",
            provider.id,
        )

    adapter = adapter_class(provider)

    langchain_model = adapter.get_chat_model(
        model_id=model_id,
        temperature=temperature,
        max_tokens=max_tokens,
        callbacks=callbacks,
        **kwargs,
    )

    formatter = _SimpleFormatter()
    return langchain_model, formatter


def create_langchain_model_by_agent_id(
    agent_id: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    callbacks: Optional[list[AsyncCallbackHandler]] = None,
    **kwargs: Any,
) -> Tuple[BaseChatModel, Any]:
    """Create a LangChain model based on agent configuration.

    This function uses the agent's configured model, falling back
    to the global active model if no agent-specific model is set.

    Args:
        agent_id: Optional agent ID to load agent-specific model config
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        callbacks: List of callback handlers
        **kwargs: Additional model configuration

    Returns:
        Tuple of (langchain_model, formatter_placeholder)
    """
    from copaw.app.agent_context import get_current_agent_id
    from copaw.config.config import load_agent_config

    # Determine agent_id (parameter > context > None)
    if agent_id is None:
        try:
            agent_id = get_current_agent_id()
        except Exception:
            pass

    # Try to get agent-specific model first
    model_slot = None
    if agent_id:
        try:
            agent_config = load_agent_config(agent_id)
            model_slot = agent_config.active_model
        except Exception:
            pass

    if model_slot and model_slot.provider_id and model_slot.model:
        # Use agent-specific model
        return create_langchain_model(
            provider_id=model_slot.provider_id,
            model_id=model_slot.model,
            temperature=temperature,
            max_tokens=max_tokens,
            callbacks=callbacks,
            **kwargs,
        )

    # Fallback to global active model
    manager = ProviderManager.get_instance()
    global_model = manager.get_active_model()
    if not global_model:
        raise ValueError(
            "No active model configured. "
            "Please configure a model using 'copaw models config' "
            "or set an agent-specific model."
        )

    return create_langchain_model(
        provider_id=global_model.provider_id,
        model_id=global_model.model,
        temperature=temperature,
        max_tokens=max_tokens,
        callbacks=callbacks,
        **kwargs,
    )


class _SimpleFormatter:
    """Simple formatter placeholder for LangChain models.

    LangChain handles message formatting internally, so we just
    provide a minimal interface for compatibility.
    """

    @staticmethod
    def convert_messages_to_dict(messages: list) -> list:
        """Convert messages to dict format (LangChain handles this)."""
        return []

    @staticmethod
    def convert_tool_result_to_string(output: Any) -> tuple[str, list]:
        """Convert tool result to string format."""
        if isinstance(output, str):
            return output, []
        return str(output), []


__all__ = [
    "create_langchain_model",
    "create_langchain_model_from_provider",
    "create_langchain_model_by_agent_id",
    "create_model_and_formatter",
    "get_provider_adapter",
]


def create_model_and_formatter(
    agent_id: Optional[str] = None,
    model_slot: Optional[str] = None,
    **kwargs: Any,
) -> Tuple[BaseChatModel, Any]:
    """Factory function to create a model and formatter.

    This is the main entry point for creating chat models in CoPaw.
    Uses LangChain under the hood.

    Args:
        agent_id: Optional agent ID for agent-specific model config
        model_slot: Optional model slot name (e.g., 'model', 'vision')
        **kwargs: Additional model configuration

    Returns:
        Tuple of (chat_model, formatter)

    Example:
        >>> model, formatter = create_model_and_formatter(agent_id="default")
    """
    return create_langchain_model_by_agent_id(
        agent_id=agent_id,
        **kwargs,
    )