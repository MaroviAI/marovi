"""
Translation-specific schema classes for requests and responses.

This module provides schema classes specific to translation services.
"""

from typing import Dict, Optional, List, Union
from pydantic import BaseModel, Field

from .base import BaseRequest, BaseResponse, BatchRequest, BatchResponse

class TranslationRequest(BaseRequest):
    """Schema for translation requests."""
    text: Union[str, List[str]] = Field(..., description="Text or list of texts to translate")
    source_lang: str = Field(..., description="Source language code")
    target_lang: str = Field(..., description="Target language code")
    preserve_formatting: bool = Field(default=True, description="Whether to preserve formatting in translation")
    glossary: Optional[Dict[str, str]] = Field(default=None, description="Custom glossary for translation")

class TranslationResponse(BaseResponse):
    """Schema for translation responses."""
    content: Union[str, List[str]] = Field(..., description="Translated text or list of translated texts")
    source_lang: str = Field(..., description="Source language code")
    target_lang: str = Field(..., description="Target language code")
    detected_lang: Optional[str] = Field(default=None, description="Detected source language if auto-detection was used")
    confidence: Optional[float] = Field(default=None, description="Confidence score for the translation")

class TranslationBatchRequest(BatchRequest):
    """Schema for batch translation requests."""
    items: List[TranslationRequest] = Field(..., description="List of translation requests to process")
    max_concurrency: int = Field(default=5, description="Maximum number of concurrent requests")

class TranslationBatchResponse(BatchResponse):
    """Schema for batch translation responses."""
    items: List[TranslationResponse] = Field(..., description="List of translation responses")
    total_characters: int = Field(..., description="Total characters translated")
    avg_confidence: Optional[float] = Field(default=None, description="Average confidence score across all translations")
