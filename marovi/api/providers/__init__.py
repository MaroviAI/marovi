"""
LLM and translation provider implementations.

This module contains implementations for various LLM and translation providers.
"""

from .base import LLMProvider, TranslationProvider, Provider
from .openai import OpenAIProvider
from .anthropic import AnthropicProvider
# Handle optional dependencies with try/except
try:
    from .google import GoogleTranslateProvider, GeminiProvider
    _has_google = True
except ImportError:
    _has_google = False
    # Define stub classes that raise errors when used
    class GoogleTranslateProvider:
        def __init__(self, *args, **kwargs):
            raise ImportError("Google Translate provider requires the 'requests' package")
            
    class GeminiProvider:
        def __init__(self, *args, **kwargs):
            raise ImportError("Gemini provider requires the 'google.generativeai' package")

# Handle REST-based Google providers
try:
    from .google_rest import GoogleTranslateRestProvider, GeminiRestProvider
    _has_google_rest = True
except ImportError:
    _has_google_rest = False
    # Define stub classes that raise errors when used
    class GoogleTranslateRestProvider:
        def __init__(self, *args, **kwargs):
            raise ImportError("GoogleTranslateRestProvider requires the 'requests' package")
            
    class GeminiRestProvider:
        def __init__(self, *args, **kwargs):
            raise ImportError("GeminiRestProvider requires the 'google.generativeai' package")

# Handle DeepL dependency            
try:
    from .deepl import DeepLProvider
    _has_deepl = True
except ImportError:
    _has_deepl = False
    # Define stub class that raises error when used
    class DeepLProvider:
        def __init__(self, *args, **kwargs):
            raise ImportError("DeepL provider requires the 'requests' package")

from .custom import ChatGPTTranslationProvider

__all__ = [
    # Base classes
    "Provider",
    "LLMProvider",
    "TranslationProvider",
    
    # LLM providers
    "OpenAIProvider",
    "AnthropicProvider",
    "GeminiProvider",
    "GeminiRestProvider",
    
    # Translation providers
    "GoogleTranslateProvider",
    "GoogleTranslateRestProvider",
    "DeepLProvider",
    "ChatGPTTranslationProvider"
]
