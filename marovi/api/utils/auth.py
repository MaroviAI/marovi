"""
Authentication utilities for the API.

This module provides authentication functionality for API requests.
"""

import os
from typing import Optional, Dict, Any, Callable
from functools import wraps

from ..config import settings, get_api_key

class APIKeyManager:
    """Manager for API keys."""
    
    def __init__(self):
        """Initialize API key manager."""
        self._api_keys: Dict[str, str] = {}
    
    def get_api_key(self, provider: str) -> Optional[str]:
        """Get API key for a provider.
        
        Args:
            provider: Provider name (e.g., "openai", "anthropic")
            
        Returns:
            API key if available, otherwise None
        """
        # First check for unified API key
        if self._api_keys.get("marovi"):
            return self._api_keys.get("marovi")
            
        # Then check environment variables using the config utility
        env_key = get_api_key(provider)
        if env_key:
            return env_key
        
        # Finally check stored keys
        return self._api_keys.get(provider)
    
    def set_api_key(self, provider: str, api_key: str) -> None:
        """Set API key for a provider."""
        self._api_keys[provider] = api_key
        
    def set_unified_api_key(self, api_key: str) -> None:
        """Set unified API key for all providers."""
        self._api_keys["marovi"] = api_key
    
    def remove_api_key(self, provider: str) -> None:
        """Remove API key for a provider."""
        if provider in self._api_keys:
            del self._api_keys[provider]
    
    def clear_api_keys(self) -> None:
        """Clear all stored API keys."""
        self._api_keys.clear()

def require_api_key(provider: str) -> Callable:
    """Decorator to require API key for a function."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            api_key = api_key_manager.get_api_key(provider)
            if not api_key:
                raise ValueError(f"API key required for {provider}")
            return func(*args, **kwargs)
        return wrapper
    return decorator

def require_api_key_async(provider: str) -> Callable:
    """Decorator to require API key for an async function."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            api_key = api_key_manager.get_api_key(provider)
            if not api_key:
                raise ValueError(f"API key required for {provider}")
            return await func(*args, **kwargs)
        return wrapper
    return decorator

# Create global API key manager instance
api_key_manager = APIKeyManager()

# Initialize with default API keys from settings
if settings.MAROVI_API_KEY:
    api_key_manager.set_unified_api_key(settings.MAROVI_API_KEY)
if settings.OPENAI_API_KEY:
    api_key_manager.set_api_key("openai", settings.OPENAI_API_KEY)
if settings.ANTHROPIC_API_KEY:
    api_key_manager.set_api_key("anthropic", settings.ANTHROPIC_API_KEY)
if settings.GOOGLE_TRANSLATE_API_KEY:
    api_key_manager.set_api_key("google", settings.GOOGLE_TRANSLATE_API_KEY)
if settings.DEEPL_API_KEY:
    api_key_manager.set_api_key("deepl", settings.DEEPL_API_KEY)
