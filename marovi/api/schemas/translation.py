"""
Translation-specific schema classes for requests and responses.

This module provides schema classes specific to translation services.
"""

from typing import Dict, Optional, List, Union, Any
from pydantic import BaseModel, Field, validator
from enum import Enum

from .base import BaseRequest, BaseResponse, BatchRequest, BatchResponse

class TranslationFormat(str, Enum):
    """Supported translation formats."""
    TEXT = "text"
    HTML = "html"
    MARKDOWN = "markdown"
    JSON = "json"
    XML = "xml"

class GlossaryEntry(BaseModel):
    """A single glossary entry."""
    source_term: str = Field(..., description="Source language term")
    target_term: str = Field(..., description="Target language term")
    context: Optional[str] = Field(default=None, description="Optional context for the term")
    case_sensitive: bool = Field(default=True, description="Whether the term matching should be case sensitive")
    exact_match: bool = Field(default=True, description="Whether to require exact matching")

class TranslationRequest(BaseRequest):
    """Schema for translation requests."""
    text: Union[str, List[str]] = Field(..., description="Text or list of texts to translate")
    source_lang: str = Field(..., description="Source language code")
    target_lang: str = Field(..., description="Target language code")
    format: TranslationFormat = Field(default=TranslationFormat.TEXT, description="Format of the input text")
    preserve_formatting: bool = Field(default=True, description="Whether to preserve formatting in translation")
    glossary: Optional[List[GlossaryEntry]] = Field(default=None, description="Custom glossary for translation")
    base_translation: Optional[Union[str, List[str]]] = Field(default=None, description="Optional base translation to refine")
    quality_preference: Optional[str] = Field(default=None, description="Quality preference (e.g., 'speed', 'accuracy')")
    domain: Optional[str] = Field(default=None, description="Domain-specific translation (e.g., 'medical', 'legal')")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata for the translation")

    @validator('text')
    def validate_text(cls, v):
        """Validate text input."""
        if isinstance(v, str) and not v.strip():
            raise ValueError("Text cannot be empty")
        if isinstance(v, list) and not all(t.strip() for t in v):
            raise ValueError("All texts in the list must be non-empty")
        return v

    @validator('glossary')
    def validate_glossary(cls, v):
        """Validate glossary entries."""
        if v is not None:
            if not all(entry.source_term.strip() and entry.target_term.strip() for entry in v):
                raise ValueError("Glossary entries cannot have empty terms")
        return v

class TranslationResponse(BaseResponse):
    """Schema for translation responses."""
    content: Union[str, List[str]] = Field(..., description="Translated text or list of translated texts")
    source_lang: str = Field(..., description="Source language code")
    target_lang: str = Field(..., description="Target language code")
    detected_lang: Optional[str] = Field(default=None, description="Detected source language if auto-detection was used")
    confidence: Optional[float] = Field(default=None, description="Confidence score for the translation")
    quality_metrics: Optional[Dict[str, float]] = Field(default=None, description="Additional quality metrics")
    glossary_applied: Optional[bool] = Field(default=None, description="Whether glossary was applied")
    format: TranslationFormat = Field(..., description="Format of the translated text")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata from the translation")

class TranslationBatchRequest(BatchRequest):
    """Schema for batch translation requests."""
    items: List[TranslationRequest] = Field(..., description="List of translation requests to process")
    max_concurrency: int = Field(default=5, description="Maximum number of concurrent requests")
    batch_size: int = Field(default=10, description="Number of items to process in each batch")
    retry_strategy: Optional[Dict[str, Any]] = Field(default=None, description="Retry strategy for failed requests")

class TranslationBatchResponse(BatchResponse):
    """Schema for batch translation responses."""
    items: List[TranslationResponse] = Field(..., description="List of translation responses")
    total_characters: int = Field(..., description="Total characters translated")
    avg_confidence: Optional[float] = Field(default=None, description="Average confidence score across all translations")
    quality_metrics: Optional[Dict[str, float]] = Field(default=None, description="Aggregated quality metrics")
    failed_items: Optional[List[Dict[str, Any]]] = Field(default=None, description="Details of failed translations")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional batch metadata")
