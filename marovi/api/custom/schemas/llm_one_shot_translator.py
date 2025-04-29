"""
Schemas for LLM one-shot translation.

This module provides schema classes specific to LLM-based translation with structured output.
"""

from typing import Optional
from pydantic import BaseModel, Field

class TranslationOutput(BaseModel):
    """Structured output format for LLM translation responses."""
    translated_text: str = Field(..., description="The translated text")
    confidence: float = Field(..., description="Confidence score for the translation (0-1)")
    detected_language: Optional[str] = Field(default=None, description="Detected source language if auto-detection was used")
    notes: Optional[str] = Field(default=None, description="Any additional notes about the translation") 