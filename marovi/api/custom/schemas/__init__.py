"""
Schema definitions for custom endpoints.

This module provides schemas for custom endpoint requests and responses.
"""

from .requests import (
    TranslationRequest,
    FormatConversionRequest,
    SummarizationRequest,
    SupportedFormat,
    SummaryStyle,
)

from .responses import (
    TranslationResponse,
    FormatConversionResponse,
    SummarizationResponse,
)

# Export all schemas
__all__ = [
    # Request schemas
    "TranslationRequest",
    "FormatConversionRequest",
    "SummarizationRequest",
    # Response schemas
    "TranslationResponse",
    "FormatConversionResponse",
    "SummarizationResponse",
    # Enums
    "SupportedFormat",
    "SummaryStyle",
]
