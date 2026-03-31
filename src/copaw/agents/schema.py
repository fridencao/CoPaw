# -*- coding: utf-8 -*-
"""
Agent tools schema: type definitions for agent tool responses.
"""
from typing import Literal, Optional
from typing_extensions import TypedDict, Required


# Simple source types compatible with LangGraph
class Base64Source(TypedDict, total=False):
    """Base64 encoded source."""
    type: Required[Literal["base64"]]
    media_type: str
    data: str


class URLSource(TypedDict, total=False):
    """URL source."""
    type: Required[Literal["url"]]
    url: str


class FileBlock(TypedDict, total=False):
    """File block for sending files to users."""

    type: Required[Literal["file"]]
    """The type of the block"""

    source: Required[Base64Source | URLSource]
    """The source of the file"""

    filename: Optional[str]
    """The filename of the file"""