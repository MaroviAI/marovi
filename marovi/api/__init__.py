"""
Marovi API package for accessing LLM and translation services.

This package provides a unified interface for accessing different types of services
through a single router instance.
"""

# Import enums and core types first to avoid circular imports
from .core.base import ServiceType

# Import provider registry and config before other modules that might need them
from .providers.provider_registry import provider_registry
from .config import ProviderType, get_api_key, get_default_model, get_provider_features, settings

# Now import router and client module objects
from .core.router import Router, default_router
from .core.client import MaroviAPI, default_client, api

# Import client classes after router is fully initialized
from .clients.llm import LLMClient, create_llm_client
from .clients.translation import TranslationClient, create_translation_client

# Import schemas
from .schemas.base import BaseRequest, BaseResponse, BatchRequest, BatchResponse
from .schemas.llm import LLMRequest, LLMResponse, LLMBatchRequest, LLMBatchResponse
from .schemas.translation import TranslationRequest, TranslationResponse, TranslationBatchRequest, TranslationBatchResponse

# Import utilities
from .utils.logging import get_logger, setup_logging
from .utils.retry import retry, async_retry
from .utils.cache import cached, async_cached, Cache
from .utils.auth import api_key_manager

# Import version info
__version__ = "0.1.0"

# Set up defaults for easy access
# api and router variables are now imported directly from their respective modules

__all__ = [
    # Core components
    "Router",
    "default_router",
    "ServiceType",
    "ProviderType",
    "MaroviAPI",
    "default_client",
    "api",
    "router",
    "LLMClient",
    "TranslationClient",
    "create_llm_client",
    "create_translation_client",
    "provider_registry",
    
    # Custom endpoint components
    "CustomEndpoint",
    "custom_registry",
    "Pipeline",
    
    # Configuration
    "get_api_key",
    "get_default_model",
    "get_provider_features",
    "settings",
    
    # Schemas
    "BaseRequest",
    "BaseResponse",
    "BatchRequest",
    "BatchResponse",
    "LLMRequest",
    "LLMResponse",
    "LLMBatchRequest",
    "LLMBatchResponse",
    "TranslationRequest",
    "TranslationResponse",
    "TranslationBatchRequest",
    "TranslationBatchResponse",
    
    # Utilities
    "get_logger",
    "setup_logging",
    "retry",
    "async_retry",
    "cached",
    "async_cached",
    "Cache",
    "api_key_manager",
    
    # Version
    "__version__"
]
