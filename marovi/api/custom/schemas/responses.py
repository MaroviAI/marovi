"""
Response schemas for custom endpoints.

This module contains Pydantic models for response schemas used by custom endpoints.
"""

from typing import Dict, Any, List, Optional, Union
from enum import Enum
from pydantic import BaseModel, Field

# Translation response
class TranslationResponse(BaseModel):
    """
    Response schema for translation.
    
    This model defines the output structure for translation results.
    """
    text: str = Field(..., description="The original text content")
    translated_text: str = Field(..., description="The translated text content")
    source_lang: str = Field(..., description="The source language code")
    target_lang: str = Field(..., description="The target language code")
    success: bool = Field(True, description="Whether the translation was successful")
    error: Optional[str] = Field(None, description="Error message if translation failed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata about the translation")

# Format conversion response
class FormatConversionResponse(BaseModel):
    """
    Response schema for format conversion.
    
    This model defines the output structure for format conversion results.
    """
    text: str = Field(..., description="The original text content")
    converted_text: str = Field(..., description="The converted text content")
    source_format: str = Field(..., description="The source format of the original text")
    target_format: str = Field(..., description="The target format of the converted text")
    success: bool = Field(True, description="Whether the conversion was successful")
    error: Optional[str] = Field(None, description="Error message if conversion failed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata about the conversion")

# Summarization response
class SummarizationResponse(BaseModel):
    """
    Response schema for text summarization.
    
    This model defines the output structure for summarization results.
    """
    text: str = Field(..., description="The original text content")
    summary: str = Field(..., description="The generated summary")
    style: str = Field(..., description="The style of summary that was generated")
    word_count_original: Optional[int] = Field(None, description="Approximate word count of the original text")
    word_count_summary: Optional[int] = Field(None, description="Approximate word count of the summary")
    keywords: Optional[List[str]] = Field(None, description="Key terms or keywords extracted from the text")
    success: bool = Field(True, description="Whether the summarization was successful")
    error: Optional[str] = Field(None, description="Error message if summarization failed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata about the summarization") 