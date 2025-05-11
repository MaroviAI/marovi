"""
Base classes for LLM and translation providers.

This module contains the base classes that providers must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Type, Union, AsyncIterator

# Import from core base to avoid circular imports
from ..core.base import ServiceType

# Import schemas directly for strong typing
from ..schemas.base import BaseRequest, BaseResponse
from ..schemas.llm import LLMRequest, LLMResponse
from ..schemas.translation import TranslationRequest, TranslationResponse

class Provider(ABC):
    """Base class for all providers."""
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the provider with necessary API clients."""
        pass
    
    @abstractmethod
    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about the provider."""
        pass
    
    @abstractmethod
    def get_default_model(self) -> str:
        """Get the default model for this provider."""
        pass
    
    @abstractmethod
    def get_features(self) -> List[str]:
        """Get the features supported by this provider."""
        pass

class LLMProvider(Provider):
    """Base class for LLM providers."""
    
    @abstractmethod
    def complete(self, request: LLMRequest, response_model: Optional[Type] = None) -> LLMResponse:
        """Generate a completion from the LLM."""
        pass
    
    @abstractmethod
    async def acomplete(self, request: LLMRequest, response_model: Optional[Type] = None) -> LLMResponse:
        """Generate a completion from the LLM asynchronously."""
        pass
    
    @abstractmethod
    async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        """Stream a completion from the LLM."""
        pass
    
    def get_service_type(self) -> ServiceType:
        """Get the service type for this provider."""
        return ServiceType.LLM

class TranslationProvider(Provider):
    """Base class for translation providers."""
    
    @abstractmethod
    def translate(self, request: TranslationRequest, response_model: Optional[Type] = None) -> TranslationResponse:
        """Translate text."""
        pass
    
    @abstractmethod
    async def atranslate(self, request: TranslationRequest, response_model: Optional[Type] = None) -> TranslationResponse:
        """Translate text asynchronously."""
        pass
    
    def get_service_type(self) -> ServiceType:
        """Get the service type for this provider."""
        return ServiceType.TRANSLATION
    