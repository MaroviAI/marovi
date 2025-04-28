"""
Cache utilities for the API.

This module provides caching functionality for API responses.
"""

import time
import hashlib
import json
from typing import Any, Optional, Dict, Callable
from functools import wraps
from threading import Lock

from ..config import settings

class Cache:
    """Simple in-memory cache with TTL support."""
    
    def __init__(self, max_size: int = 100, ttl: int = 3600):
        """Initialize cache."""
        self.max_size = max_size
        self.ttl = ttl
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = Lock()
    
    def _generate_key(self, *args: Any, **kwargs: Any) -> str:
        """Generate a cache key from arguments."""
        # Convert args and kwargs to a stable string representation
        key_parts = [str(arg) for arg in args]
        key_parts.extend(f"{k}:{v}" for k, v in sorted(kwargs.items()))
        key_str = "|".join(key_parts)
        
        # Generate a hash of the key string
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache."""
        with self._lock:
            if key not in self._cache:
                return None
            
            entry = self._cache[key]
            if time.time() > entry["expires_at"]:
                del self._cache[key]
                return None
            
            return entry["value"]
    
    def set(self, key: str, value: Any) -> None:
        """Set a value in the cache."""
        with self._lock:
            # Remove oldest entry if cache is full
            if len(self._cache) >= self.max_size:
                oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k]["expires_at"])
                del self._cache[oldest_key]
            
            self._cache[key] = {
                "value": value,
                "expires_at": time.time() + self.ttl
            }
    
    def delete(self, key: str) -> None:
        """Delete a value from the cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
    
    def clear(self) -> None:
        """Clear all values from the cache."""
        with self._lock:
            self._cache.clear()

def cached(max_size: Optional[int] = None, ttl: Optional[int] = None) -> Callable:
    """Decorator for caching function results."""
    def decorator(func: Callable) -> Callable:
        cache = Cache(
            max_size=max_size or settings.DEFAULT_CACHE_SIZE,
            ttl=ttl or 3600
        )
        
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not settings.ENABLE_CACHE:
                return func(*args, **kwargs)
            
            key = cache._generate_key(*args, **kwargs)
            result = cache.get(key)
            
            if result is None:
                result = func(*args, **kwargs)
                cache.set(key, result)
            
            return result
        
        return wrapper
    return decorator

def async_cached(max_size: Optional[int] = None, ttl: Optional[int] = None) -> Callable:
    """Decorator for caching async function results."""
    def decorator(func: Callable) -> Callable:
        cache = Cache(
            max_size=max_size or settings.DEFAULT_CACHE_SIZE,
            ttl=ttl or 3600
        )
        
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not settings.ENABLE_CACHE:
                return await func(*args, **kwargs)
            
            key = cache._generate_key(*args, **kwargs)
            result = cache.get(key)
            
            if result is None:
                result = await func(*args, **kwargs)
                cache.set(key, result)
            
            return result
        
        return wrapper
    return decorator

# Create default cache instance
default_cache = Cache(
    max_size=settings.DEFAULT_CACHE_SIZE,
    ttl=3600
) 