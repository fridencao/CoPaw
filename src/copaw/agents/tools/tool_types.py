# -*- coding: utf-8 -*-
"""Tool response types for LangGraph-compatible tools.

Replaces AgentScope's ToolResponse with a simpler dict-based format
that integrates with LangGraph's ToolMessage system.
"""
from typing import Any, Optional


def text_content(text: str) -> list[dict[str, Any]]:
    """Create a text content block.

    Args:
        text: The text content

    Returns:
        Content block list compatible with ToolResponse
    """
    return [{"type": "text", "text": text}]


def image_content(image_url: str, alt_text: str = "") -> list[dict[str, Any]]:
    """Create an image content block.

    Args:
        image_url: URL or path to the image
        alt_text: Alternative text for the image

    Returns:
        Content block list compatible with ToolResponse
    """
    return [{"type": "image_url", "image_url": {"url": image_url}, "alt": alt_text}]


class ToolResponse(dict):
    """Simple tool response compatible with LangGraph.

    This is a dict subclass that accepts content in a format compatible
    with LangGraph's tool messaging system.

    Usage:
        return ToolResponse(content=text_content("Hello"))
        return ToolResponse(content="Hello")  # Simple string
    """

    def __init__(
        self,
        content: Any = None,
        **kwargs: Any,
    ):
        """Initialize tool response.

        Args:
            content: The response content (str, list, or dict)
            **kwargs: Additional fields
        """
        super().__init__(**kwargs)
        if content is not None:
            self["content"] = content

    @property
    def text(self) -> str:
        """Get text content from response."""
        content = self.get("content", "")
        if isinstance(content, list) and len(content) > 0:
            block = content[0]
            if isinstance(block, dict):
                return block.get("text", str(content))
        if isinstance(content, str):
            return content
        return str(content)

    @property
    def content(self) -> Any:
        """Get raw content."""
        return self.get("content")


# Type alias for backward compatibility
ToolResult = ToolResponse