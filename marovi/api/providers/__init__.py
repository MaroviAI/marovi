"""
LLM and translation provider implementations.

This module contains implementations for various LLM and translation providers.
"""

# Import only base classes at module level to avoid circular imports
from .base import LLMProvider, TranslationProvider, Provider
from .provider_registry import provider_registry

# Use lazy imports for concrete provider implementations to avoid circular imports
def get_openai_provider():
    from .openai import OpenAIProvider
    return OpenAIProvider

def get_anthropic_provider():
    from .anthropic import AnthropicProvider
    return AnthropicProvider

def get_google_translate_provider():
    from .google import GoogleTranslateProvider
    return GoogleTranslateProvider

def get_gemini_provider():
    from .google import GeminiProvider
    return GeminiProvider

def get_google_translate_rest_provider():
    from .google_rest import GoogleTranslateRestProvider
    return GoogleTranslateRestProvider

def get_gemini_rest_provider():
    from .google_rest import GeminiRestProvider
    return GeminiRestProvider

def get_deepl_provider():
    from .deepl import DeepLProvider
    return DeepLProvider

def register_default_providers(router):
    """
    Register all default providers with a router.
    
    This function will register providers and raise an exception if any provider
    cannot be initialized due to implementation issues.
    
    Args:
        router: Router instance to register providers with
        
    Returns:
        True if any providers were registered successfully
    """
    import logging
    import inspect
    logger = logging.getLogger(__name__)
    
    # Track if we registered any providers successfully
    registered_any_provider = False
    
    # Register OpenAI provider (defaults for both LLM and translation)
    try:
        openai_provider = get_openai_provider()()
        router.add_provider(openai_provider)
        registered_any_provider = True
    except Exception as e:
        logger.warning(f"Failed to register OpenAI provider: {e}")
    
    # Register Google Translate provider
    try:
        GoogleTranslateProvider = get_google_translate_provider()
        if not inspect.isabstract(GoogleTranslateProvider):
            router.add_provider(GoogleTranslateProvider())
            registered_any_provider = True
    except Exception as e:
        logger.warning(f"Failed to register Google Translate provider: {e}")
    
    # Register DeepL provider
    try:
        DeepLProvider = get_deepl_provider()
        if not inspect.isabstract(DeepLProvider):
            router.add_provider(DeepLProvider())
            registered_any_provider = True
    except Exception as e:
        logger.warning(f"Failed to register DeepL provider: {e}")
    
    # Log a warning if no providers were registered
    if not registered_any_provider:
        logger.warning("No providers were registered. The API will have limited functionality.")
        
    return registered_any_provider

# Create class aliases to make them directly importable 
# This preserves lazy imports while allowing direct imports like "from ..providers import GoogleTranslateProvider"
GoogleTranslateProvider = get_google_translate_provider()
GeminiProvider = get_gemini_provider()
OpenAIProvider = get_openai_provider()
AnthropicProvider = get_anthropic_provider()
DeepLProvider = get_deepl_provider()
GoogleTranslateRestProvider = get_google_translate_rest_provider()
GeminiRestProvider = get_gemini_rest_provider()

__all__ = [
    # Base classes
    "Provider",
    "LLMProvider",
    "TranslationProvider",
    
    # Provider registry
    "provider_registry",
    
    # Functions
    "register_default_providers",
    
    # Provider getters
    "get_openai_provider",
    "get_anthropic_provider",
    "get_google_translate_provider",
    "get_gemini_provider",
    "get_google_translate_rest_provider",
    "get_gemini_rest_provider",
    "get_deepl_provider",
    
    # Provider classes
    "GoogleTranslateProvider",
    "GeminiProvider",
    "OpenAIProvider",
    "AnthropicProvider", 
    "DeepLProvider",
    "GoogleTranslateRestProvider",
    "GeminiRestProvider"
]
