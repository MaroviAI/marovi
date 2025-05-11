"""
Translation client for interacting with various translation providers.

This module provides a unified interface for making requests to different translation
providers with support for batching, retries, and comprehensive observability.
"""

import time
import logging
import asyncio
from typing import List, Dict, Optional, Union, Type, Any

from ..providers.base import TranslationProvider, TranslationRequest, TranslationResponse
from ..config import get_api_key

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TranslationError(Exception):
    """Exception raised for translation errors."""
    pass

# Valid provider strings
PROVIDER_GOOGLE = "google"
PROVIDER_GOOGLE_REST = "google_rest"
PROVIDER_DEEPL = "deepl"
PROVIDER_CUSTOM = "custom"

VALID_PROVIDERS = [PROVIDER_GOOGLE, PROVIDER_GOOGLE_REST, PROVIDER_DEEPL, PROVIDER_CUSTOM]

# Default retry configuration
DEFAULT_RETRY_CONFIG = {
    "max_retries": 3,
    "initial_backoff": 1.0,
    "max_backoff": 10.0,
    "backoff_factor": 2.0,
    "retryable_errors": [
        "rate_limit_exceeded",
        "server_error",
        "connection_error",
        "timeout"
    ]
}

# Global default provider and client cache
_default_provider = PROVIDER_GOOGLE
_client_cache: Dict[str, 'TranslationClient'] = {}

def set_default_provider(provider: str):
    """
    Set the default translation provider for the application.
    
    Args:
        provider: Provider identifier string
    
    Raises:
        ValueError: If provider is not valid
    """
    global _default_provider
    
    if provider not in VALID_PROVIDERS:
        raise ValueError(f"Unknown provider: {provider}. Valid options are: {VALID_PROVIDERS}")
    
    _default_provider = provider
    logger.info(f"Default provider set to {_default_provider}")

def get_client(provider: Optional[str] = None) -> 'TranslationClient':
    """
    Get a translation client for the specified provider.
    
    Args:
        provider: Provider identifier string (uses default if None)
        
    Returns:
        TranslationClient instance
        
    Raises:
        ValueError: If provider is not valid
    """
    global _client_cache, _default_provider
    
    provider = provider or _default_provider
    
    if provider not in VALID_PROVIDERS:
        raise ValueError(f"Unknown provider: {provider}. Valid options are: {VALID_PROVIDERS}")
    
    if provider not in _client_cache:
        _client_cache[provider] = TranslationClient(provider=provider)
    
    return _client_cache[provider]

class TranslationClient:
    """
    A unified client for interacting with various translation providers.
    
    Features:
    - Support for multiple providers (Google Translate, DeepL, etc.)
    - Batch translation support
    - Async and sync interfaces
    - Automatic logging
    - Comprehensive observability
    - Retry logic for transient failures
    """
    
    def __init__(self, provider: str = PROVIDER_GOOGLE, api_key: Optional[str] = None, 
                custom_provider: Optional[TranslationProvider] = None,
                retry_config: Optional[Dict] = None):
        """
        Initialize a translation client.
        
        Args:
            provider: Provider identifier string
            api_key: Optional API key (if None, will be fetched from config)
            custom_provider: Optional custom provider implementation
            retry_config: Configuration for retry logic
            
        Raises:
            ValueError: If provider is not valid
        """
        self.provider_type = provider
        
        if provider not in VALID_PROVIDERS:
            raise ValueError(f"Unknown provider: {provider}. Valid options are: {VALID_PROVIDERS}")
        
        # Get API key from config if not provided
        if api_key is None and provider != PROVIDER_CUSTOM:
            api_key = get_api_key(provider)
        
        # Initialize the provider
        if custom_provider:
            self.provider = custom_provider
        else:
            if provider == PROVIDER_GOOGLE:
                from ..providers import GoogleTranslateProvider
                self.provider = GoogleTranslateProvider(api_key=api_key)
            elif provider == PROVIDER_GOOGLE_REST:
                from ..providers import GoogleTranslateRestProvider
                self.provider = GoogleTranslateRestProvider(api_key=api_key)
            elif provider == PROVIDER_DEEPL:
                from ..providers import DeepLProvider
                self.provider = DeepLProvider(api_key=api_key)
            elif provider == PROVIDER_CUSTOM:
                raise ValueError("Custom provider requires a provider implementation")
            else:
                raise ValueError(f"Unsupported provider: {provider}")
        
        # Set retry configuration
        self.retry_config = retry_config or DEFAULT_RETRY_CONFIG
        
        logger.info(f"Initialized TranslationClient with provider={provider}")
    
    def _should_retry(self, error: Exception, attempt: int) -> bool:
        """
        Determine if a request should be retried based on the error and attempt number.
        
        Args:
            error: The exception that occurred
            attempt: The current attempt number (starting from 1)
            
        Returns:
            True if the request should be retried, False otherwise
        """
        if attempt >= self.retry_config.get("max_retries", 3):
            return False
        
        error_type = type(error).__name__.lower()
        error_msg = str(error).lower()
        
        # Check if error matches any retryable error patterns
        retryable_errors = self.retry_config.get("retryable_errors", [
            "timeout", "connection", "server_error", "rate_limit"
        ])
        
        for retryable_error in retryable_errors:
            if retryable_error in error_type or retryable_error in error_msg:
                return True
        
        return False
    
    async def _backoff(self, attempt: int) -> None:
        """Implement exponential backoff for retries."""
        backoff_time = min(
            self.retry_config["initial_backoff"] * (self.retry_config["backoff_factor"] ** attempt),
            self.retry_config["max_backoff"]
        )
        await asyncio.sleep(backoff_time)
    
    def _sync_backoff(self, attempt: int) -> None:
        """Implement synchronous exponential backoff for retries."""
        backoff_time = min(
            self.retry_config["initial_backoff"] * (self.retry_config["backoff_factor"] ** attempt),
            self.retry_config["max_backoff"]
        )
        time.sleep(backoff_time)
    
    def translate(self, 
                 text: Union[str, List[str]], 
                 source_lang: str, 
                 target_lang: str,
                 provider: Optional[str] = None) -> Union[str, List[str]]:
        """
        Translate text from one language to another.
        
        Args:
            text: Text to translate (string or list of strings)
            source_lang: Source language code
            target_lang: Target language code
            provider: Optional provider override for this call
            
        Returns:
            Translated text (string or list of strings)
            
        Raises:
            TranslationError: If translation fails
            ValueError: If provider is not valid
        """
        # If a different provider is specified, use that client instead
        if provider is not None and provider != self.provider_type:
            return get_client(provider).translate(text, source_lang, target_lang)
            
        start_time = time.time()
        
        # Prepare request
        request = TranslationRequest(
            text=text,
            source_lang=source_lang,
            target_lang=target_lang,
            metadata=None
        )
        
        # Track attempts for retry logic
        attempt = 1
        max_retries = self.retry_config.get("max_retries", 3)
        
        while True:
            try:
                # Execute translation
                response = self.provider.translate(request)
                
                # Check for successful response
                if not response.success:
                    raise TranslationError(response.error or "Translation failed without specific error")
                
                logger.info(f"Translation successful: {self.provider_type}, "
                            f"source={source_lang}, target={target_lang}, "
                            f"latency: {time.time() - start_time:.2f}s")
                
                # Return translated text
                return response.content
                
            except Exception as e:
                # Log error
                logger.error(f"Translation error: {self.provider_type}, "
                             f"source={source_lang}, target={target_lang}, "
                             f"attempt {attempt}/{max_retries}, error: {str(e)}")
                
                # Check if we should retry
                if self._should_retry(e, attempt):
                    # Implement backoff before retry
                    self._sync_backoff(attempt)
                    attempt += 1
                else:
                    # If this is a known error type or we've exceeded retries, re-raise
                    if isinstance(e, TranslationError):
                        raise
                    else:
                        # Wrap unknown errors in TranslationError
                        raise TranslationError(str(e)) from e
    
    async def atranslate(self, 
                        text: Union[str, List[str]], 
                        source_lang: str, 
                        target_lang: str,
                        provider: Optional[str] = None) -> Union[str, List[str]]:
        """
        Translate text from one language to another asynchronously.
        
        Args:
            text: Text to translate (string or list of strings)
            source_lang: Source language code
            target_lang: Target language code
            provider: Optional provider override for this call
            
        Returns:
            Translated text (string or list of strings)
            
        Raises:
            TranslationError: If translation fails
            ValueError: If provider is not valid
        """
        # If a different provider is specified, use that client instead
        if provider is not None and provider != self.provider_type:
            return await get_client(provider).atranslate(text, source_lang, target_lang)
            
        start_time = time.time()
        
        # Prepare request
        request = TranslationRequest(
            text=text,
            source_lang=source_lang,
            target_lang=target_lang,
            metadata=None
        )
        
        # Track attempts for retry logic
        attempt = 1
        max_retries = self.retry_config.get("max_retries", 3)
        
        while True:
            try:
                # Execute translation
                response = await self.provider.atranslate(request)
                
                # Check for successful response
                if not response.success:
                    raise TranslationError(response.error or "Translation failed without specific error")
                
                logger.info(f"Async translation successful: {self.provider_type}, "
                            f"source={source_lang}, target={target_lang}, "
                            f"latency: {time.time() - start_time:.2f}s")
                
                # Return translated text
                return response.content
                
            except Exception as e:
                # Log error
                logger.error(f"Async translation failed after {attempt} attempts: {str(e)}")
                
                # Check if we should retry
                if self._should_retry(e, attempt):
                    # Implement backoff before retry
                    await self._backoff(attempt)
                    attempt += 1
                else:
                    # If this is a known error type or we've exceeded retries, re-raise
                    if isinstance(e, TranslationError):
                        raise
                    else:
                        # Wrap unknown errors in TranslationError
                        raise TranslationError(str(e)) from e
    
    def batch_translate(self,
                       texts: List[str],
                       source_lang: str,
                       target_lang: str,
                       max_concurrency: int = 5,
                       provider: Optional[str] = None) -> List[str]:
        """
        Translate multiple texts in batch.
        
        Args:
            texts: List of texts to translate
            source_lang: Source language code
            target_lang: Target language code
            max_concurrency: Maximum number of concurrent requests
            provider: Optional provider override for this call
            
        Returns:
            List of translated texts
            
        Raises:
            TranslationError: If translation fails
            ValueError: If provider is not valid
        """
        # If a different provider is specified, use that client instead
        if provider is not None and provider != self.provider_type:
            return get_client(provider).batch_translate(texts, source_lang, target_lang, max_concurrency)
            
        batch_start_time = time.time()
        
        # Prepare requests
        requests = [
            TranslationRequest(
                text=text,
                source_lang=source_lang,
                target_lang=target_lang,
                metadata=None
            )
            for i, text in enumerate(texts)
        ]
        
        # Process in batches to respect concurrency limits
        results = []
        for i in range(0, len(requests), max_concurrency):
            batch = requests[i:i + max_concurrency]
            batch_results = self.provider.batch_translate(batch)
            results.extend(batch_results)
        
        batch_time = time.time() - batch_start_time
        logger.info(f"Batch translation completed: {self.provider_type}, "
                   f"source={source_lang}, target={target_lang}, "
                   f"total_time: {batch_time:.2f}s, batch_size: {len(texts)}")
        
        # Extract translated texts from responses
        translated_texts = [r.content for r in results]
        return translated_texts
    
    async def abatch_translate(self,
                             texts: List[str],
                             source_lang: str,
                             target_lang: str,
                             max_concurrency: int = 5,
                             provider: Optional[str] = None) -> List[str]:
        """
        Translate multiple texts asynchronously in batch.
        
        Args:
            texts: List of texts to translate
            source_lang: Source language code
            target_lang: Target language code
            max_concurrency: Maximum number of concurrent requests
            provider: Optional provider override for this call
            
        Returns:
            List of translated texts
            
        Raises:
            TranslationError: If translation fails
            ValueError: If provider is not valid
        """
        # If a different provider is specified, use that client instead
        if provider is not None and provider != self.provider_type:
            return await get_client(provider).abatch_translate(texts, source_lang, target_lang, max_concurrency)
            
        batch_start_time = time.time()
        
        # Prepare requests
        requests = [
            TranslationRequest(
                text=text,
                source_lang=source_lang,
                target_lang=target_lang,
                metadata=None
            )
            for i, text in enumerate(texts)
        ]
        
        # Process in batches to respect concurrency limits
        results = []
        for i in range(0, len(requests), max_concurrency):
            batch = requests[i:i + max_concurrency]
            batch_results = await self.provider.abatch_translate(batch)
            results.extend(batch_results)
        
        batch_time = time.time() - batch_start_time
        logger.info(f"Async batch translation completed: {self.provider_type}, "
                   f"source={source_lang}, target={target_lang}, "
                   f"total_time: {batch_time:.2f}s, batch_size: {len(texts)}")
        
        # Extract translated texts from responses
        translated_texts = [r.content for r in results]
        return translated_texts
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported language codes."""
        return self.provider.get_supported_languages()


def create_translation_client(provider: Optional[str] = None) -> TranslationClient:
    """
    Factory function to create a translation client.
    
    Args:
        provider: Translation provider identifier (uses default if None)
        
    Returns:
        TranslationClient instance
        
    Raises:
        ValueError: If provider is not valid
    """
    return get_client(provider)
