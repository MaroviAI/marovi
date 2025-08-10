"""
Custom Endpoints for the Marovi API.

This module provides custom endpoints that extend the Marovi API functionality.
"""

# Import from schemas module, not directly from schemas.py which is deleted
from .schemas import (
    TranslationRequest,
    TranslationResponse,
    FormatConversionRequest,
    FormatConversionResponse,
    SummarizationRequest,
    SummarizationResponse,
    CleanTextRequest,
    CleanTextResponse,
    SupportedFormat,
    SummaryStyle
)

# Import base translator and derived classes
from .endpoints.llm_translate import (
    LLMTranslate,
    create_llm_translate
)

# Import format converter
from .endpoints.convert_format import (
    FormatConverter,
    FormatConversionError
)

# Import summarizer
from .endpoints.summarize import (
    Summarizer,
    SummarizationError
)

# Import text cleaner
from .endpoints.clean_text import CleanText

# Import registry functionality
from .core.registry import (
    register_endpoint,
    register_default_endpoints,
    default_registry
)

__all__ = [
    # Translation
    "LLMTranslate",
    "create_llm_translate",
    
    # Format converter
    "FormatConverter",
    "FormatConversionError",

    # Summarizer
    "Summarizer",
    "SummarizationError",

    # Text cleaner
    "CleanText",

    # Registry functionality
    "register_endpoint",
    "register_default_endpoints",
    "default_registry",
    
    # Schemas
    "TranslationRequest",
    "TranslationResponse",
    "FormatConversionRequest",
    "FormatConversionResponse",
    "SummarizationRequest",
    "SummarizationResponse",
    "CleanTextRequest",
    "CleanTextResponse",
    "SupportedFormat",
    "SummaryStyle"
]

# Register all default endpoints during package initialization
register_default_endpoints(default_registry)
