"""
Request schemas for custom endpoints.

This module contains Pydantic models for request schemas used by custom endpoints.
"""

from typing import Dict, Any, List, Optional, Union
from enum import Enum
from pydantic import BaseModel, Field, validator

# Shared enums
class SupportedFormat(str, Enum):
    """Supported formats for the format converter."""
    HTML = "html"
    MARKDOWN = "markdown"
    MD = "md"
    WIKI = "wiki"
    LATEX = "latex"
    TEX = "tex"
    RST = "rst"
    ORG = "org"
    ASCIIDOC = "asciidoc"
    PLAIN = "plain"
    TEXT = "text"

class SummaryStyle(str, Enum):
    """Supported summary styles."""
    BULLET = "bullet"
    PARAGRAPH = "paragraph"
    STRUCTURED = "structured"
    CONCISE = "concise"

# Translation request
class TranslationRequest(BaseModel):
    """
    Request schema for translation.
    
    This model defines the input parameters for translating text between languages.
    """
    text: str = Field(..., description="The text content to translate")
    source_lang: str = Field(..., description="The source language code")
    target_lang: str = Field(..., description="The target language code")
    options: Optional[Dict[str, Any]] = Field(None, description="Additional options for the translation")

# Format conversion request
class FormatConversionRequest(BaseModel):
    """
    Request schema for format conversion.
    
    This model defines the input parameters for converting text between different formats.
    """
    text: str = Field(..., description="The text content to convert")
    source_format: SupportedFormat = Field(..., description="The source format of the text")
    target_format: SupportedFormat = Field(..., description="The target format to convert to")
    preserve_structure: bool = Field(True, description="Whether to preserve document structure")
    preserve_links: bool = Field(True, description="Whether to preserve links in the text")
    preserve_images: bool = Field(True, description="Whether to preserve image references")
    options: Optional[Dict[str, Any]] = Field(None, description="Additional options for the conversion")
    
    class Config:
        use_enum_values = True  # Use string values when serializing enums

# Summarization request
class SummarizationRequest(BaseModel):
    """
    Request schema for text summarization.
    
    This model defines the input parameters for summarizing text.
    """
    text: str = Field(..., description="The text content to summarize")
    style: Optional[SummaryStyle] = Field(SummaryStyle.CONCISE, description="The style of summary to generate")
    max_length: Optional[int] = Field(None, description="The approximate maximum length of the summary in words")
    include_keywords: bool = Field(False, description="Whether to include key terms/keywords in the summary")
    preserve_quotes: bool = Field(False, description="Whether to preserve important quotes in the summary")
    language: Optional[str] = Field("en", description="The language to generate the summary in (ISO code)")
    options: Optional[Dict[str, Any]] = Field(None, description="Additional options for the summarization")
    
    class Config:
        use_enum_values = True  # Use string values when serializing enums 