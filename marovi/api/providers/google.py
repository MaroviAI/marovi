"""
Google provider implementations.

This module provides implementations for Google's Translation and LLM (Gemini) services.
"""

import os
import time
import logging
import json
import random
import asyncio
from typing import List, Dict, Optional, Type, Any, Union, AsyncIterator

import google.generativeai as genai
import requests
from pydantic import BaseModel

from ..schemas.llm import LLMRequest, LLMResponse, LLMBatchRequest, LLMBatchResponse
from ..schemas.translation import TranslationRequest, TranslationResponse, TranslationBatchRequest, TranslationBatchResponse
from .base import LLMProvider, TranslationProvider
from ..utils.logging import get_logger
from ..utils.retry import retry, async_retry, calculate_backoff
from ..utils.cache import cached, async_cached

# Configure logging
logger = get_logger(__name__)

#
# Google Translation Provider
#
class GoogleTranslateProvider(TranslationProvider):
    """
    Translation provider using Google Translate V2 REST API.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Google Translate provider.
        
        Args:
            api_key: Optional API key (if not provided, will use environment variables)
        """
        self.api_key = api_key or os.getenv("GOOGLE_TRANSLATE_API_KEY")
        if not self.api_key:
            raise ValueError("Google Translate API key not provided and not found in environment variables")
        
        self.base_url = "https://translation.googleapis.com/language/translate/v2"
        self.session = requests.Session()
    
    def initialize(self) -> None:
        """Initialize the provider. No special initialization needed for REST API."""
        pass
    
    def get_default_model(self) -> str:
        """Get the default model for this provider."""
        return "nmt"  # Neural Machine Translation
    
    def get_supported_models(self) -> List[str]:
        """Get list of supported models."""
        return ["nmt"]
    
    def get_rate_limits(self) -> Dict[str, int]:
        """Get rate limits for the provider."""
        return {
            "requests_per_minute": 1000,
            "characters_per_minute": 1000000
        }
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported language codes."""
        # Google Translate supports a large number of languages
        return [
            'af', 'am', 'ar', 'az', 'be', 'bg', 'bn', 'bs', 'ca', 'ceb', 'co', 'cs', 'cy', 'da', 'de', 'el', 'en', 'eo',
            'es', 'et', 'eu', 'fa', 'fi', 'fr', 'fy', 'ga', 'gd', 'gl', 'gu', 'ha', 'haw', 'he', 'hi', 'hmn', 'hr', 'ht',
            'hu', 'hy', 'id', 'ig', 'is', 'it', 'iw', 'ja', 'jw', 'ka', 'kk', 'km', 'kn', 'ko', 'ku', 'ky', 'la', 'lb',
            'lo', 'lt', 'lv', 'mg', 'mi', 'mk', 'ml', 'mn', 'mr', 'ms', 'mt', 'my', 'ne', 'nl', 'no', 'ny', 'or', 'pa',
            'pl', 'ps', 'pt', 'ro', 'ru', 'rw', 'si', 'sk', 'sl', 'sm', 'sn', 'so', 'sq', 'sr', 'st', 'su', 'sv', 'sw',
            'ta', 'te', 'tg', 'th', 'tk', 'tl', 'tr', 'tt', 'ug', 'uk', 'ur', 'uz', 'vi', 'xh', 'yi', 'yo', 'zh', 'zu'
        ]
    
    def get_features(self) -> List[str]:
        """Get the features supported by this provider."""
        return [
            "text_translation",
            "language_detection",
            "batch_translation"
        ]
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about the provider."""
        return {
            "id": "google_translate",
            "name": "Google Translate",
            "version": "v2",
            "api_type": "REST",
            "documentation_url": "https://cloud.google.com/translate/docs/reference/rest/v2/translate",
            "service_type": "translation",
            "rate_limits": self.get_rate_limits(),
            "supported_models": self.get_supported_models(),
            "default_model": self.get_default_model(),
            "features": self.get_features()
        }
    
    @retry()
    def translate(self, request: TranslationRequest) -> TranslationResponse:
        """
        Translate a single text using Google Translate V2 REST API.
        
        Args:
            request: Translation request containing text and language info
            
        Returns:
            TranslationResponse with translated text
        """
        start_time = time.time()
        
        # Handle single text or list of texts
        is_batch = isinstance(request.text, list)
        texts = request.text if is_batch else [request.text]
        
        # Prepare API request parameters
        params = {
            "key": self.api_key,
            "q": texts,
            "source": request.source_lang,
            "target": request.target_lang,
            "format": "text"
        }
        
        try:
            # Make API request
            response = self.session.post(self.base_url, params=params)
            response.raise_for_status()
            
            # Parse response
            data = response.json()
            if "data" not in data or "translations" not in data["data"]:
                raise ValueError(f"Unexpected API response format: {data}")
            
            translations = [t["translatedText"] for t in data["data"]["translations"]]
            
            # Return single text or list based on input
            translated_text = translations if is_batch else translations[0]
            
            return TranslationResponse(
                content=translated_text,
                source_lang=request.source_lang,
                target_lang=request.target_lang,
                detected_lang=None,  # Google V2 API doesn't return detected language in this endpoint
                confidence=None,  # No confidence score in basic API
                metadata=request.metadata,
                timestamp=time.time(),
                latency=time.time() - start_time,
                success=True,
                format=request.format
            )
            
        except Exception as e:
            logger.error(f"Google Translate API request failed: {str(e)}")
            return TranslationResponse(
                content="",
                source_lang=request.source_lang,
                target_lang=request.target_lang,
                timestamp=time.time(),
                latency=time.time() - start_time,
                success=False,
                error=str(e),
                format=request.format
            )
    
    @async_retry()
    async def atranslate(self, request: TranslationRequest) -> TranslationResponse:
        """
        Asynchronously translate text.
        
        Uses an async wrapper around the synchronous implementation for now.
        """
        # This is an async wrapper for the synchronous implementation
        # In a production environment, this should be properly implemented using async HTTP client
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.translate, request)
    
    def batch_translate(self, request: TranslationBatchRequest) -> TranslationBatchResponse:
        """
        Translate multiple texts in batch using Google Translate V2 REST API.
        """
        start_time = time.time()
        responses = []
        total_chars = 0
        errors = []
        
        # Group requests by language pair for efficiency
        grouped_requests = {}
        for item in request.items:
            lang_pair = (item.source_lang, item.target_lang)
            if lang_pair not in grouped_requests:
                grouped_requests[lang_pair] = []
            grouped_requests[lang_pair].append(item)
        
        # Process each language pair group
        for (source_lang, target_lang), items in grouped_requests.items():
            batch_request = TranslationRequest(
                text=[item.text for item in items],
                source_lang=source_lang,
                target_lang=target_lang,
                metadata=request.metadata
            )
            
            try:
                batch_response = self.translate(batch_request)
                
                # Create individual responses
                if batch_response.success and isinstance(batch_response.content, list):
                    for i, item in enumerate(items):
                        if i < len(batch_response.content):
                            text = item.text
                            total_chars += len(text) if isinstance(text, str) else sum(len(t) for t in text)
                            
                            response = TranslationResponse(
                                content=batch_response.content[i],
                                source_lang=source_lang,
                                target_lang=target_lang,
                                detected_lang=batch_response.detected_lang,
                                confidence=batch_response.confidence,
                                metadata=item.metadata,
                                timestamp=time.time(),
                                latency=batch_response.latency,
                                success=True,
                                format=item.format
                            )
                            responses.append(response)
                else:
                    # Handle error
                    errors.append(batch_response.error or "Unknown error in batch translation")
                    
                    # Create error responses for all items in this group
                    for item in items:
                        text = item.text
                        total_chars += len(text) if isinstance(text, str) else sum(len(t) for t in text)
                        
                        response = TranslationResponse(
                            content="",
                            source_lang=source_lang,
                            target_lang=target_lang,
                            detected_lang=None,
                            confidence=None,
                            metadata=item.metadata,
                            timestamp=time.time(),
                            latency=0,
                            success=False,
                            error=batch_response.error,
                            format=item.format
                        )
                        responses.append(response)
                    
            except Exception as e:
                errors.append(str(e))
                
                # Create error responses for all items in this group
                for item in items:
                    text = item.text
                    total_chars += len(text) if isinstance(text, str) else sum(len(t) for t in text)
                    
                    response = TranslationResponse(
                        content="",
                        source_lang=source_lang,
                        target_lang=target_lang,
                        detected_lang=None,
                        confidence=None,
                        metadata=item.metadata,
                        timestamp=time.time(),
                        latency=0,
                        success=False,
                        error=str(e),
                        format=item.format
                    )
                    responses.append(response)
        
        batch_time = time.time() - start_time
        
        return TranslationBatchResponse(
            items=responses,
            total_characters=total_chars,
            avg_confidence=None,  # No confidence score in basic API
            metadata=request.metadata,
            timestamp=time.time(),
            total_latency=batch_time,
            avg_latency=batch_time / len(request.items) if request.items else 0,
            success=len(errors) == 0,
            errors=errors if errors else None
        )
    
    async def abatch_translate(self, request: TranslationBatchRequest) -> TranslationBatchResponse:
        """
        Asynchronously translate multiple texts in batch.
        
        Uses an async wrapper around the synchronous implementation for now.
        """
        # This is an async wrapper for the synchronous implementation
        # In a production environment, this should be properly implemented using async HTTP client
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.batch_translate, request)

#
# Google Gemini Provider
#
class GeminiProvider(LLMProvider):
    """
    LLM provider implementation for Google's Gemini models.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Gemini provider."""
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key not provided and GOOGLE_API_KEY environment variable not set")
        self.is_initialized = False
    
    def initialize(self) -> None:
        """Initialize the Gemini client."""
        if not self.is_initialized:
            genai.configure(api_key=self.api_key)
            self.is_initialized = True
            logger.info("Initialized Gemini provider")
    
    def get_default_model(self) -> str:
        """Get the default model for Gemini."""
        return "gemini-1.5-pro"
    
    def get_supported_models(self) -> List[str]:
        """Get list of supported models."""
        return ["gemini-1.5-pro", "gemini-1.5-flash"]
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        # Gemini supports many languages but there's no official list
        # This is a subset of commonly supported languages
        return ["en", "es", "fr", "de", "it", "pt", "nl", "pl", "ru", "zh", "ja", "ko", "ar", "hi"]
    
    def get_features(self) -> List[str]:
        """Get the features supported by this provider."""
        return [
            "text_generation",
            "streaming",
            "system_prompts",
            "json_output"
        ]
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about the provider."""
        return {
            "id": "gemini",
            "name": "Google Gemini",
            "version": "1.0",
            "api_type": "API",
            "documentation_url": "https://ai.google.dev/",
            "service_type": "llm",
            "supported_models": self.get_supported_models(),
            "default_model": self.get_default_model(),
            "features": self.get_features()
        }
    
    def _prepare_gemini_request(self, request: LLMRequest) -> dict:
        """Prepare parameters for Gemini API call."""
        # Ensure initialization
        self.initialize()
        
        # Prepare generation config
        generation_config = {
            "temperature": request.temperature,
            "max_output_tokens": request.max_tokens,
            "top_p": request.top_p if request.top_p is not None else 0.95,
        }
        
        # Add safety settings (default conservative settings)
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            }
        ]
        
        # Prepare messages
        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "parts": [request.system_prompt]})
        messages.append({"role": "user", "parts": [request.prompt]})
        
        return {
            "model": request.model or self.get_default_model(),
            "messages": messages,
            "generation_config": generation_config,
            "safety_settings": safety_settings
        }
    
    @retry()
    def complete(self, request: LLMRequest, response_model: Optional[Type[BaseModel]] = None) -> LLMResponse:
        """Generate a completion using Gemini API."""
        start_time = time.time()
        
        try:
            # Prepare request parameters
            params = self._prepare_gemini_request(request)
            model_name = params["model"]
            
            # Create model instance
            model = genai.GenerativeModel(model_name=model_name)
            
            # Generate response
            response = model.generate_content(
                params["messages"],
                generation_config=params["generation_config"],
                safety_settings=params["safety_settings"]
            )
            
            # Extract text content
            raw_content = response.text
            
            # Parse structured output if model provided
            if response_model:
                try:
                    content = response_model.model_validate_json(raw_content)
                except Exception as e:
                    logger.error(f"Failed to parse response as {response_model.__name__}: {str(e)}")
                    logger.debug(f"Raw response: {raw_content}")
                    raise
            else:
                content = raw_content
            
            # Calculate tokens (Gemini API doesn't return token counts directly)
            # This is an estimation
            prompt_tokens = sum(len(msg.get("parts", [""])[0]) for msg in params["messages"]) // 4
            completion_tokens = len(raw_content) // 4
            
            # Create usage metrics
            usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
            
            # Create response
            return LLMResponse(
                content=content,
                model=model_name,
                usage=usage,
                latency=time.time() - start_time,
                raw_response=response,
                finish_reason="stop",  # Gemini API doesn't provide this directly
                metadata=request.metadata,
                timestamp=time.time(),
                success=True
            )
            
        except Exception as e:
            logger.error(f"Gemini API call failed: {str(e)}")
            return LLMResponse(
                content="",
                model=request.model or self.get_default_model(),
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                latency=time.time() - start_time,
                raw_response=None,
                finish_reason="error",
                metadata=request.metadata,
                timestamp=time.time(),
                success=False,
                error=str(e)
            )
    
    @async_retry()
    async def acomplete(self, request: LLMRequest, response_model: Optional[Type[BaseModel]] = None) -> LLMResponse:
        """Generate a completion asynchronously using Gemini API."""
        # This is an async wrapper for the synchronous implementation
        # In a production environment, this should be properly implemented using async API
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.complete, request, response_model)
    
    async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        """Stream a completion from Gemini API."""
        # Ensure initialization
        self.initialize()
        
        try:
            # Prepare request parameters
            params = self._prepare_gemini_request(request)
            model_name = params["model"]
            
            # Create model instance
            model = genai.GenerativeModel(model_name=model_name)
            
            # Generate streaming response
            response = model.generate_content(
                params["messages"],
                generation_config=params["generation_config"],
                safety_settings=params["safety_settings"],
                stream=True
            )
            
            # Yield chunks
            async for chunk in response:
                if hasattr(chunk, "text") and chunk.text:
                    yield chunk.text
                    
        except Exception as e:
            logger.error(f"Gemini streaming API call failed: {str(e)}")
            raise
    
    def batch_complete(self, request: LLMBatchRequest) -> LLMBatchResponse:
        """Generate completions in batch."""
        start_time = time.time()
        responses = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        errors = []
        
        # Process each request sequentially
        # For better concurrency, this should be implemented using asyncio
        for item in request.items:
            try:
                response = self.complete(item)
                responses.append(response)
                
                if response.success:
                    total_prompt_tokens += response.usage.get("prompt_tokens", 0)
                    total_completion_tokens += response.usage.get("completion_tokens", 0)
                else:
                    errors.append(response.error or "Unknown error")
            except Exception as e:
                errors.append(str(e))
                # Create error response
                responses.append(LLMResponse(
                    content="",
                    model=item.model or self.get_default_model(),
                    usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                    latency=0,
                    raw_response=None,
                    finish_reason="error",
                    metadata=item.metadata,
                    timestamp=time.time(),
                    success=False,
                    error=str(e)
                ))
        
        batch_time = time.time() - start_time
        
        return LLMBatchResponse(
            items=responses,
            total_tokens=total_prompt_tokens + total_completion_tokens,
            avg_tokens=(total_prompt_tokens + total_completion_tokens) / len(request.items) if request.items else 0,
            metadata=request.metadata,
            timestamp=time.time(),
            total_latency=batch_time,
            avg_latency=batch_time / len(request.items) if request.items else 0,
            success=len(errors) == 0,
            errors=errors if errors else None
        )
    
    async def abatch_complete(self, request: LLMBatchRequest) -> LLMBatchResponse:
        """Generate completions in batch asynchronously."""
        start_time = time.time()
        responses = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        errors = []
        
        # Process requests concurrently with limits
        tasks = []
        semaphore = asyncio.Semaphore(request.max_concurrency)
        
        async def process_request(item):
            async with semaphore:
                return await self.acomplete(item)
        
        # Create tasks for all requests
        for item in request.items:
            tasks.append(process_request(item))
        
        # Wait for all tasks to complete
        for task in asyncio.as_completed(tasks):
            try:
                response = await task
                responses.append(response)
                
                if response.success:
                    total_prompt_tokens += response.usage.get("prompt_tokens", 0)
                    total_completion_tokens += response.usage.get("completion_tokens", 0)
                else:
                    errors.append(response.error or "Unknown error")
            except Exception as e:
                errors.append(str(e))
                # We would need to identify which request failed to create a proper error response
                # This is a simplification
                responses.append(LLMResponse(
                    content="",
                    model=self.get_default_model(),
                    usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                    latency=0,
                    raw_response=None,
                    finish_reason="error",
                    metadata=None,
                    timestamp=time.time(),
                    success=False,
                    error=str(e)
                ))
        
        batch_time = time.time() - start_time
        
        return LLMBatchResponse(
            items=responses,
            total_tokens=total_prompt_tokens + total_completion_tokens,
            avg_tokens=(total_prompt_tokens + total_completion_tokens) / len(request.items) if request.items else 0,
            metadata=request.metadata,
            timestamp=time.time(),
            total_latency=batch_time,
            avg_latency=batch_time / len(request.items) if request.items else 0,
            success=len(errors) == 0,
            errors=errors if errors else None
        )
