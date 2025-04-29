"""
Base provider classes for LLM and translation services.

This module provides abstract base classes that define the interface for all providers.
"""

from abc import ABC, abstractmethod
from typing import Optional, Type, AsyncIterator, List, Dict, Any, Union
from pydantic import BaseModel

from ..schemas.base import BaseRequest, BaseResponse, BatchRequest, BatchResponse
from ..schemas.llm import LLMRequest, LLMResponse, LLMBatchRequest, LLMBatchResponse
from ..schemas.translation import (
    TranslationRequest, TranslationResponse, TranslationBatchRequest, TranslationBatchResponse,
    TranslationFormat, GlossaryEntry
)

class Provider(ABC):
    """Abstract base class for all providers."""
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the provider client."""
        pass
    
    @abstractmethod
    def get_default_model(self) -> str:
        """Get the default model for this provider."""
        pass
    
    @abstractmethod
    def get_supported_models(self) -> List[str]:
        """Get list of supported models."""
        pass
    
    @abstractmethod
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        pass
    
    @abstractmethod
    def get_supported_formats(self) -> List[TranslationFormat]:
        """Get list of supported translation formats."""
        pass
    
    @abstractmethod
    def get_quality_metrics(self) -> List[str]:
        """Get list of supported quality metrics."""
        pass
    
    @abstractmethod
    def get_rate_limits(self) -> Dict[str, Any]:
        """Get rate limits and quotas for this provider."""
        pass

class LLMProvider(Provider):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def complete(self, request: LLMRequest, response_model: Optional[Type[BaseModel]] = None) -> LLMResponse:
        """Generate a completion."""
        pass
    
    @abstractmethod
    async def acomplete(self, request: LLMRequest, response_model: Optional[Type[BaseModel]] = None) -> LLMResponse:
        """Generate a completion asynchronously."""
        pass
    
    @abstractmethod
    async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        """Stream a completion."""
        pass
    
    @abstractmethod
    def batch_complete(self, request: LLMBatchRequest) -> LLMBatchResponse:
        """Generate completions in batch."""
        pass
    
    @abstractmethod
    async def abatch_complete(self, request: LLMBatchRequest) -> LLMBatchResponse:
        """Generate completions in batch asynchronously."""
        pass

class TranslationProvider(Provider):
    """Abstract base class for translation providers."""
    
    @abstractmethod
    def translate(self, request: TranslationRequest) -> TranslationResponse:
        """Translate text."""
        pass
    
    @abstractmethod
    async def atranslate(self, request: TranslationRequest) -> TranslationResponse:
        """Translate text asynchronously."""
        pass
    
    @abstractmethod
    def batch_translate(self, request: TranslationBatchRequest) -> TranslationBatchResponse:
        """Translate texts in batch."""
        pass
    
    @abstractmethod
    async def abatch_translate(self, request: TranslationBatchRequest) -> TranslationBatchResponse:
        """Translate texts in batch asynchronously."""
        pass
    
    @abstractmethod
    def apply_glossary(self, text: str, glossary: List[GlossaryEntry]) -> str:
        """Apply glossary terms to translated text."""
        pass
    
    @abstractmethod
    def refine_translation(self, text: str, base_translation: str) -> str:
        """Refine a base translation."""
        pass
    
    @abstractmethod
    def get_quality_metrics(self, source: str, target: str) -> Dict[str, float]:
        """Get quality metrics for a translation."""
        pass
    
    @abstractmethod
    def detect_language(self, text: str) -> str:
        """Detect the language of the input text."""
        pass
    
    @abstractmethod
    def get_supported_domains(self) -> List[str]:
        """Get list of supported translation domains."""
        pass
    
    @abstractmethod
    def get_supported_quality_preferences(self) -> List[str]:
        """Get list of supported quality preferences."""
        pass
    