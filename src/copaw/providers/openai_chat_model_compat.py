# -*- coding: utf-8 -*-
"""OpenAI chat model compatibility wrappers.

Deprecated: Use langchain adapters instead.
"""

from __future__ import annotations

from typing import Any


class ChatResponse:
    """Stub for compatibility."""
    def __init__(self, content: str = "", **kwargs: Any):
        self.content = content
        for k, v in kwargs.items():
            setattr(self, k, v)


class OpenAIChatModel:
    """Stub for compatibility."""
    def __init__(self, **kwargs: Any):
        pass


__all__ = ["ChatResponse", "OpenAIChatModel"]