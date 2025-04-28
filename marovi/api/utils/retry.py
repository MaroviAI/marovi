"""
Retry utilities for the API.

This module provides retry functionality for API requests.
"""

import time
import asyncio
import random
from typing import Callable, Any, Optional, Type, Union, List, Dict
from functools import wraps

from ..config import settings, get_retry_config

def should_retry(error: Exception, attempt: int, retry_config: Optional[Dict[str, Any]] = None) -> bool:
    """Determine if a request should be retried based on the error and attempt number."""
    config = retry_config or get_retry_config()
    
    if attempt >= config["max_retries"]:
        return False
    
    error_type = type(error).__name__.lower()
    error_msg = str(error).lower()
    
    # Check if error matches any retryable error patterns
    for retryable_error in config["retryable_errors"]:
        if retryable_error in error_type or retryable_error in error_msg:
            return True
    
    return False

def calculate_backoff(attempt: int, retry_config: Optional[Dict[str, Any]] = None) -> float:
    """Calculate backoff time for retry attempt."""
    config = retry_config or get_retry_config()
    
    backoff_time = min(
        config["initial_backoff"] * (config["backoff_factor"] ** attempt),
        config["max_backoff"]
    )
    
    # Add jitter to prevent thundering herd
    jitter = backoff_time * 0.1 * (2 * random.random() - 1)
    return backoff_time + jitter

def retry(max_retries: Optional[int] = None,
          retry_config: Optional[Dict[str, Any]] = None,
          on_retry: Optional[Callable[[Exception, int], None]] = None) -> Callable:
    """Decorator for retrying functions."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            config = retry_config or get_retry_config()
            max_attempts = max_retries or config["max_retries"]
            attempt = 0
            
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempt += 1
                    
                    if not should_retry(e, attempt, config):
                        raise
                    
                    if on_retry:
                        on_retry(e, attempt)
                    
                    backoff_time = calculate_backoff(attempt - 1, config)
                    time.sleep(backoff_time)
        
        return wrapper
    return decorator

def async_retry(max_retries: Optional[int] = None,
                retry_config: Optional[Dict[str, Any]] = None,
                on_retry: Optional[Callable[[Exception, int], None]] = None) -> Callable:
    """Decorator for retrying async functions."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            config = retry_config or get_retry_config()
            max_attempts = max_retries or config["max_retries"]
            attempt = 0
            
            while True:
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    attempt += 1
                    
                    if not should_retry(e, attempt, config):
                        raise
                    
                    if on_retry:
                        on_retry(e, attempt)
                    
                    backoff_time = calculate_backoff(attempt - 1, config)
                    await asyncio.sleep(backoff_time)
        
        return wrapper
    return decorator

class RetryContext:
    """Context manager for retrying operations."""
    
    def __init__(self,
                 max_retries: Optional[int] = None,
                 retry_config: Optional[Dict[str, Any]] = None,
                 on_retry: Optional[Callable[[Exception, int], None]] = None):
        """Initialize retry context."""
        self.config = retry_config or get_retry_config()
        self.max_attempts = max_retries or self.config["max_retries"]
        self.on_retry = on_retry
        self.attempt = 0
    
    def __enter__(self) -> 'RetryContext':
        """Enter retry context."""
        return self
    
    def __exit__(self, exc_type: Optional[Type[Exception]], exc_val: Optional[Exception], exc_tb: Any) -> bool:
        """Exit retry context."""
        if exc_val is None:
            return True
        
        self.attempt += 1
        
        if not should_retry(exc_val, self.attempt, self.config):
            return False
        
        if self.on_retry:
            self.on_retry(exc_val, self.attempt)
        
        backoff_time = calculate_backoff(self.attempt - 1, self.config)
        time.sleep(backoff_time)
        
        return True

class AsyncRetryContext:
    """Context manager for retrying async operations."""
    
    def __init__(self,
                 max_retries: Optional[int] = None,
                 retry_config: Optional[Dict[str, Any]] = None,
                 on_retry: Optional[Callable[[Exception, int], None]] = None):
        """Initialize async retry context."""
        self.config = retry_config or get_retry_config()
        self.max_attempts = max_retries or self.config["max_retries"]
        self.on_retry = on_retry
        self.attempt = 0
    
    async def __aenter__(self) -> 'AsyncRetryContext':
        """Enter async retry context."""
        return self
    
    async def __aexit__(self, exc_type: Optional[Type[Exception]], exc_val: Optional[Exception], exc_tb: Any) -> bool:
        """Exit async retry context."""
        if exc_val is None:
            return True
        
        self.attempt += 1
        
        if not should_retry(exc_val, self.attempt, self.config):
            return False
        
        if self.on_retry:
            self.on_retry(exc_val, self.attempt)
        
        backoff_time = calculate_backoff(self.attempt - 1, self.config)
        await asyncio.sleep(backoff_time)
        
        return True
