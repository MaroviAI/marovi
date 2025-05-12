"""
Schema definitions for custom endpoints.

This module provides schemas for custom endpoint requests and responses.
"""

from .requests import (
    TranslationRequest,
    FormatConversionRequest,
    SummarizationRequest,
    CleanTextRequest,
    SupportedFormat,
    SummaryStyle,
)

from .responses import (
    TranslationResponse,
    FormatConversionResponse,
    SummarizationResponse,
    CleanTextResponse,
)

# Export all schemas
__all__ = [
    # Request schemas
    "TranslationRequest",
    "FormatConversionRequest",
    "SummarizationRequest",
    "CleanTextRequest",
    # Response schemas
    "TranslationResponse",
    "FormatConversionResponse",
    "SummarizationResponse",
    "CleanTextResponse",
    # Enums
    "SupportedFormat",
    "SummaryStyle",
]
