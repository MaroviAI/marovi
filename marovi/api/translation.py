"""
Translation client for interacting with various translation providers.

This module provides a unified interface for making requests to different translation
providers with support for batching, retries, and comprehensive observability.
"""

import time
import logging
import asyncio
from typing import List, Dict, Optional, Union, Type
from enum import Enum

from .providers.base import TranslationProvider, TranslationRequest, TranslationResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TranslationError(Exception):
    """Exception raised for translation errors."""
    pass

class ProviderType(Enum):
    """Supported translation providers."""
    GOOGLE = "google"
    GOOGLE_REST = "google_rest"
    DEEPL = "deepl"
    CUSTOM = "custom"

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

class TranslationClient:
    """
    A unified client for interacting with various translation providers.
    
    Features:
    - Support for multiple providers (Google Translate, DeepL, etc.)
    - Batch translation support
    - Async and sync interfaces
    - Automatic logging to pipeline context
    - Comprehensive observability
    - Retry logic for transient failures
    """
    
    def __init__(self, 
                provider: Union[str, ProviderType] = ProviderType.GOOGLE,
                api_key: Optional[str] = None,
                custom_provider: Optional[TranslationProvider] = None,
                retry_config: Optional[Dict] = None):
        """
        Initialize a translation client.
        
        Args:
            provider: Provider type or name
            api_key: Optional API key
            custom_provider: Optional custom provider implementation
            retry_config: Configuration for retry logic
        """
        # Convert string provider to enum
        if isinstance(provider, str):
            try:
                self.provider_type = ProviderType(provider)
            except ValueError:
                raise ValueError(f"Unknown provider: {provider}")
        else:
            self.provider_type = provider
        
        # Initialize the provider
        if custom_provider:
            self.provider = custom_provider
        else:
            if self.provider_type == ProviderType.GOOGLE:
                from .providers import GoogleTranslateProvider
                self.provider = GoogleTranslateProvider(api_key=api_key)
            elif self.provider_type == ProviderType.GOOGLE_REST:
                from .providers import GoogleTranslateRestProvider
                self.provider = GoogleTranslateRestProvider(api_key=api_key)
            elif self.provider_type == ProviderType.DEEPL:
                from .providers import DeepLProvider
                self.provider = DeepLProvider(api_key=api_key)
            else:
                raise ValueError(f"Unsupported provider: {self.provider_type}")
        
        # Set retry configuration
        self.retry_config = retry_config or {
            "max_retries": 3,
            "base_delay": 1,
            "max_delay": 10
        }
        
        logger.info(f"Initialized TranslationClient with provider={self.provider_type.value}")
    
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
                 step_name: Optional[str] = None) -> Union[str, List[str]]:
        """
        Translate text from one language to another.
        
        Args:
            text: Text to translate (string or list of strings)
            source_lang: Source language code
            target_lang: Target language code
            step_name: Name of the step (for context logging)
            
        Returns:
            Translated text (string or list of strings)
        """
        start_time = time.time()
        
        # Prepare metadata
        metadata = {"step_name": step_name} if step_name else None
        
        # Prepare request
        request = TranslationRequest(
            text=text,
            source_lang=source_lang,
            target_lang=target_lang,
            metadata=metadata
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
                
                # Log success if context provided
                if step_name:
                    logger.info(f"Translation successful: {self.provider_type.value}, "
                                f"source={source_lang}, target={target_lang}, "
                                f"latency: {time.time() - start_time:.2f}s")
                
                # Return translated text
                return response.content
                
            except Exception as e:
                # Log error
                logger.error(f"Translation error: {self.provider_type.value}, "
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
                        step_name: Optional[str] = None) -> Union[str, List[str]]:
        """
        Translate text from one language to another asynchronously.
        
        Args:
            text: Text to translate (string or list of strings)
            source_lang: Source language code
            target_lang: Target language code
            step_name: Name of the step (for context logging)
            
        Returns:
            Translated text (string or list of strings)
        """
        start_time = time.time()
        
        # Prepare metadata
        metadata = {"step_name": step_name} if step_name else None
        
        # Prepare request
        request = TranslationRequest(
            text=text,
            source_lang=source_lang,
            target_lang=target_lang,
            metadata=metadata
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
                
                # Log success if context provided
                if step_name:
                    logger.info(f"Async translation successful: {self.provider_type.value}, "
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
                       step_name: Optional[str] = None,
                       max_concurrency: int = 5) -> List[str]:
        """
        Translate multiple texts in batch.
        
        Args:
            texts: List of texts to translate
            source_lang: Source language code
            target_lang: Target language code
            step_name: Name of the step (for context logging)
            max_concurrency: Maximum number of concurrent requests
            
        Returns:
            List of translated texts
        """
        batch_start_time = time.time()
        
        # Prepare requests
        requests = [
            TranslationRequest(
                text=text,
                source_lang=source_lang,
                target_lang=target_lang,
                metadata={"step_name": f"{step_name}_{i}"} if step_name else None
            )
            for i, text in enumerate(texts)
        ]
        
        # Process in batches to respect concurrency limits
        results = []
        for i in range(0, len(requests), max_concurrency):
            batch = requests[i:i + max_concurrency]
            batch_results = self.provider.batch_translate(batch)
            results.extend(batch_results)
        
        # Log batch metrics if context provided
        if step_name:
            batch_time = time.time() - batch_start_time
            
            logger.info(f"Batch translation completed: {self.provider_type.value}, "
                       f"source={source_lang}, target={target_lang}, "
                       f"total_time: {batch_time:.2f}s, avg_time: {batch_time / len(texts):.2f}s, batch_size: {len(texts)}")
            
            # Log detailed batch info
            batch_info = {
                "total_time": batch_time,
                "avg_time": batch_time / len(texts),
                "batch_size": len(texts),
                "concurrent_limit": max_concurrency
            }
            
            logger.info(f"Batch translation summary: {batch_info}")
        
        # Extract translated texts from responses
        translated_texts = [r.content for r in results]
        return translated_texts
    
    async def abatch_translate(self,
                             texts: List[str],
                             source_lang: str,
                             target_lang: str,
                             step_name: Optional[str] = None,
                             max_concurrency: int = 5) -> List[str]:
        """
        Translate multiple texts asynchronously in batch.
        
        Args:
            texts: List of texts to translate
            source_lang: Source language code
            target_lang: Target language code
            step_name: Name of the step (for context logging)
            max_concurrency: Maximum number of concurrent requests
            
        Returns:
            List of translated texts
        """
        batch_start_time = time.time()
        
        # Prepare requests
        requests = [
            TranslationRequest(
                text=text,
                source_lang=source_lang,
                target_lang=target_lang,
                metadata={"step_name": f"{step_name}_{i}"} if step_name else None
            )
            for i, text in enumerate(texts)
        ]
        
        # Process in batches to respect concurrency limits
        results = []
        for i in range(0, len(requests), max_concurrency):
            batch = requests[i:i + max_concurrency]
            batch_results = await self.provider.abatch_translate(batch)
            results.extend(batch_results)
        
        # Log batch metrics if context provided
        if step_name:
            batch_time = time.time() - batch_start_time
            
            logger.info(f"Batch translation completed: {self.provider_type.value}, "
                       f"source={source_lang}, target={target_lang}, "
                       f"total_time: {batch_time:.2f}s, avg_time: {batch_time / len(texts):.2f}s, batch_size: {len(texts)}")
            
            # Log detailed batch info
            batch_info = {
                "total_time": batch_time,
                "avg_time": batch_time / len(texts),
                "batch_size": len(texts),
                "concurrent_limit": max_concurrency
            }
            
            logger.info(f"Batch translation summary: {batch_info}")
        
        # Extract translated texts from responses
        translated_texts = [r.content for r in results]
        return translated_texts
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported language codes."""
        return self.provider.get_supported_languages()


def create_translation_client(provider: Union[str, ProviderType] = ProviderType.GOOGLE,
                            api_key: Optional[str] = None,
                            custom_provider: Optional[TranslationProvider] = None,
                            retry_config: Optional[Dict] = None) -> TranslationClient:
    """
    Factory function to create a translation client with optimized configuration.
    
    Args:
        provider: Translation provider ("google", "deepl", or ProviderType enum)
        api_key: Optional API key (if not provided, will use environment variables)
        custom_provider: Optional custom provider implementation
        retry_config: Configuration for retry logic
        
    Returns:
        TranslationClient instance
    """
    return TranslationClient(
        provider=provider,
        api_key=api_key,
        custom_provider=custom_provider,
        retry_config=retry_config
    )
