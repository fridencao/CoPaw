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
    api_key_prefix: str = Field(default="", description="API key prefix hint")
    require_api_key: bool = Field(
        default=True,
        description="Whether this provider requires an API key",
    )
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
    models: List[ModelInfo] = Field(
        default_factory=list,
        description="Built-in models for this provider",
    )
    extra_models: List[ModelInfo] = Field(
        default_factory=list,
        description="User-added models for this provider",
    )
    is_local: bool = Field(
        default=False,
        description="Whether this is a local provider",
    )
    is_custom: bool = Field(
        default=False,
        description="Whether this is a custom provider",
    )
    support_model_discovery: bool = Field(
        default=False,
        description="Whether this provider supports model discovery",
    )
    support_connection_check: bool = Field(
        default=True,
        description="Whether this provider supports connection check",
    )
    freeze_url: bool = Field(
        default=False,
        description="Whether the base_url is frozen (not editable)",
    )
    generate_kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional generation kwargs",
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
        generate_kwargs: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ):
        self.id = id
        self.name = name
        self.base_url = base_url
        self.api_key = api_key
        self.generate_kwargs = generate_kwargs or {}
        self.chat_model: Optional[str] = None

        for k, v in kwargs.items():
            setattr(self, k, v)

    @abstractmethod
    def get_available_models(self) -> List[ModelInfo]:
        """Get list of available models (built-in)."""

    def get_all_models(self) -> List[ModelInfo]:
        """Get all available models including extra_models."""
        models = self.get_available_models()
        extra = getattr(self, "extra_models", None)
        if extra:
            # Add models that aren't already in the list
            existing_ids = {m.id for m in models}
            for model in extra:
                if model.id not in existing_ids:
                    models.append(model)
        return models

    def has_model(self, model_id: str) -> bool:
        """Check if a model is available in this provider."""
        return any(m.id == model_id for m in self.get_all_models())

    @abstractmethod
    def get_default_model(self) -> ModelInfo:
        """Get default model for this provider."""

    def to_provider_info(self) -> ProviderInfo:
        """Convert to ProviderInfo."""
        default_model = self.get_default_model()
        builtin_models = self.get_available_models()
        extra_models = getattr(self, "extra_models", None) or []
        return ProviderInfo(
            id=self.id,
            name=self.name,
            base_url=self.base_url,
            api_key=self.api_key,
            api_key_prefix=getattr(self, "api_key_prefix", ""),
            require_api_key=getattr(self, "require_api_key", True),
            chat_model=default_model.id if default_model else "",
            MultimodalModel=default_model,
            supported_models=self.get_all_models(),
            models=builtin_models,
            extra_models=extra_models,
            is_local=getattr(self, "is_local", False),
            is_custom=getattr(self, "is_custom", False),
            support_model_discovery=getattr(self, "support_model_discovery", False),
            support_connection_check=getattr(self, "support_connection_check", True),
            freeze_url=getattr(self, "freeze_url", False),
            generate_kwargs=getattr(self, "generate_kwargs", {}),
        )

    def update_config(self, config: dict[str, Any]) -> None:
        """Update provider configuration.

        Args:
            config: Dictionary containing configuration updates.
                   Supported keys: api_key, base_url, chat_model, generate_kwargs
        """
        if "api_key" in config and config["api_key"] is not None:
            self.api_key = config["api_key"]
        if "base_url" in config and config["base_url"] is not None:
            self.base_url = config["base_url"]
        if "chat_model" in config and config["chat_model"] is not None:
            self.chat_model = config["chat_model"]
        if "generate_kwargs" in config and config["generate_kwargs"] is not None:
            self.generate_kwargs = config["generate_kwargs"]

    async def add_model(
        self,
        model_info: ModelInfo,
    ) -> tuple[bool, str]:
        """Add a model to the provider.

        Args:
            model_info: ModelInfo object containing model id and name

        Returns:
            Tuple of (success, message)
        """
        if not hasattr(self, "extra_models"):
            self.extra_models = []
        # Check if model already exists
        for model in self.extra_models:
            if model.id == model_info.id:
                return (False, f"Model '{model_info.id}' already exists")
        self.extra_models.append(model_info)
        return (True, f"Model '{model_info.id}' added successfully")

    async def delete_model(
        self,
        model_id: str,
    ) -> tuple[bool, str]:
        """Delete a model from the provider.

        Args:
            model_id: ID of the model to delete

        Returns:
            Tuple of (success, message)
        """
        if not hasattr(self, "extra_models"):
            self.extra_models = []
        for i, model in enumerate(self.extra_models):
            if model.id == model_id:
                self.extra_models.pop(i)
                return (True, f"Model '{model_id}' deleted successfully")
        return (False, f"Model '{model_id}' not found")

    def __repr__(self) -> str:
        return f"<Provider {self.id}: {self.name}>"


__all__ = ["Provider", "ProviderInfo", "ModelInfo"]