# -*- coding: utf-8 -*-
"""Definition of Provider."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, List, Type, Any, Optional
from pydantic import BaseModel, Field


class ModelInfo(BaseModel):
    id: str = Field(..., description="Model identifier used in API calls")
    name: str = Field(..., description="Human-readable model name")
    supports_multimodal: bool | None = Field(
        default=None,
        description="Whether this model supports multimodal input "
        "(image/audio/video). None means not yet probed.",
    )
    supports_image: bool | None = Field(
        default=None,
        description="Whether this model supports image input. "
        "None means not yet probed.",
    )
    supports_video: bool | None = Field(
        default=None,
        description="Whether this model supports video input. "
        "None means not yet probed.",
    )
    probe_source: str | None = Field(
        default=None,
        description=(
            "Probe result source: 'documentation' (from docs)"
            " or 'probed' (actual probe)"
        ),
    )


class ProviderInfo(BaseModel):
    id: str = Field(..., description="Provider identifier")
    name: str = Field(..., description="Human-readable provider name")
    base_url: str = Field(default="", description="API base URL")
    api_key: str = Field(default="", description="API key for authentication")
    chat_model: str = Field(
        default="",
        description="Chat model name",
    )
    model_type: str = Field(
        default="chat",
        description="Model type: 'chat', 'completion', etc.",
    )
    MultimodalModel: Optional[ModelInfo] = Field(
        default=None,
        description="Default multimodal model for this provider",
    )
    VisionModel: Optional[ModelInfo] = Field(
        default=None,
        description="Default vision model for this provider",
    )
    supported_models: List[ModelInfo] = Field(
        default_factory=list,
        description="List of models supported by this provider",
    )


if TYPE_CHECKING:
    from .multimodal_prober import ProbeResult


class Provider(ABC):
    """Abstract base class for LLM providers."""

    type: str = "provider"

    def __init__(
        self,
        id: str,
        name: str,
        base_url: str = "",
        api_key: str = "",
        **kwargs: Any,
    ):
        self.id = id
        self.name = name
        self.base_url = base_url
        self.api_key = api_key

        for k, v in kwargs.items():
            setattr(self, k, v)

    @abstractmethod
    def get_available_models(self) -> List[ModelInfo]:
        """Get list of available models."""

    @abstractmethod
    def get_default_model(self) -> ModelInfo:
        """Get default model for this provider."""

    def to_provider_info(self) -> ProviderInfo:
        """Convert to ProviderInfo."""
        default_model = self.get_default_model()
        return ProviderInfo(
            id=self.id,
            name=self.name,
            base_url=self.base_url,
            api_key=self.api_key,
            chat_model=default_model.id if default_model else "",
            MultimodalModel=default_model,
            supported_models=self.get_available_models(),
        )

    def __repr__(self) -> str:
        return f"<Provider {self.id}: {self.name}>"


__all__ = ["Provider", "ProviderInfo", "ModelInfo"]