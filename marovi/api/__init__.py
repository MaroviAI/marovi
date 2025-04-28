"""
Marovi API package for accessing LLM and translation services.

This package provides a unified interface for accessing different types of services
through a single router instance.
"""

# Import core components
from .router import Router, ServiceType
from .llm import LLMClient, create_llm_client
from .translation import TranslationClient, create_translation_client
from .provider_registry import provider_registry

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

__all__ = [
    # Core components
    "Router",
    "ServiceType",
    "LLMClient",
    "TranslationClient",
    "create_llm_client",
    "create_translation_client",
    "provider_registry",
    
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
