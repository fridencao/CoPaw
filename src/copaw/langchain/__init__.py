# -*- coding: utf-8 -*-
"""LangChain adapter layer for CoPaw.

This module provides LangChain-compatible interfaces that replace
the AgentScope-based model layer.
"""

from .factory import create_langchain_model, create_langchain_model_from_provider
from .callbacks.token_usage import TokenUsageCallbackHandler

__all__ = [
    "create_langchain_model",
    "create_langchain_model_from_provider",
    "TokenUsageCallbackHandler",
]