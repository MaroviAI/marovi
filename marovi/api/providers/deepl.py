"""
DeepL provider implementation.

This module provides implementation for DeepL's Translation service using their REST API.
"""

import os
import time
import logging
import json
import asyncio
from typing import List, Dict, Optional, Type, Any, Union, AsyncIterator

import requests
from pydantic import BaseModel

from ..schemas.translation import TranslationRequest, TranslationResponse, TranslationBatchRequest, TranslationBatchResponse
from .base import TranslationProvider
from ..utils.logging import get_logger
from ..utils.retry import retry, async_retry, calculate_backoff
from ..utils.cache import cached, async_cached

# Configure logging
logger = get_logger(__name__)

class DeepLProvider(TranslationProvider):
    """
    Translation provider using DeepL API.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the DeepL provider.
        
        Args:
            api_key: Optional API key (if not provided, will use environment variables)
        """
        self.api_key = api_key or os.getenv("DEEPL_API_KEY")
        if not self.api_key:
            raise ValueError("DeepL API key not provided and not found in environment variables")
        
        # Check if using free tier API by examining key
        if self.api_key.endswith(":fx"):
            self.base_url = "https://api-free.deepl.com/v2"
        else:
            self.base_url = "https://api.deepl.com/v2"
        
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"DeepL-Auth-Key {self.api_key}"})
    
    def initialize(self) -> None:
        """Initialize the provider. No special initialization needed for REST API."""
        pass
    
    def get_default_model(self) -> str:
        """Get the default model for this provider."""
        return "default"  # DeepL doesn't expose different models through the API
    
    def get_supported_models(self) -> List[str]:
        """Get list of supported models."""
        return ["default"]
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported language codes."""
        # DeepL supported languages as of 2024
        return [
            'bg', 'cs', 'da', 'de', 'el', 'en', 'es', 'et', 'fi', 'fr', 'hu', 'id', 
            'it', 'ja', 'ko', 'lt', 'lv', 'nb', 'nl', 'pl', 'pt', 'ro', 'ru', 'sk', 
            'sl', 'sv', 'tr', 'uk', 'zh'
        ]
    
    @retry()
    def translate(self, request: TranslationRequest) -> TranslationResponse:
        """
        Translate a single text using DeepL API.
        
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
        data = {
            "text": texts,
            "source_lang": request.source_lang.upper(),
            "target_lang": request.target_lang.upper(),
        }
        
        # Add optional formality parameter if provided in metadata
        if request.metadata and "formality" in request.metadata:
            data["formality"] = request.metadata["formality"]
        
        try:
            # Make API request
            response = self.session.post(f"{self.base_url}/translate", json=data)
            response.raise_for_status()
            
            # Parse response
            data = response.json()
            if "translations" not in data:
                raise ValueError(f"Unexpected API response format: {data}")
            
            translations = [t["text"] for t in data["translations"]]
            detected_languages = [t.get("detected_source_language", "").lower() for t in data["translations"]]
            detected_lang = detected_languages[0] if detected_languages and len(detected_languages) > 0 else None
            
            # Return single text or list based on input
            translated_text = translations if is_batch else translations[0]
            
            return TranslationResponse(
                content=translated_text,
                source_lang=request.source_lang,
                target_lang=request.target_lang,
                detected_lang=detected_lang,
                confidence=None,  # DeepL doesn't provide confidence scores
                metadata=request.metadata,
                timestamp=time.time(),
                latency=time.time() - start_time,
                success=True
            )
            
        except Exception as e:
            logger.error(f"DeepL API request failed: {str(e)}")
            return TranslationResponse(
                content="",
                source_lang=request.source_lang,
                target_lang=request.target_lang,
                timestamp=time.time(),
                latency=time.time() - start_time,
                success=False,
                error=str(e)
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
        Translate multiple texts in batch using DeepL API.
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
                                confidence=None,  # DeepL doesn't provide confidence scores
                                metadata=item.metadata,
                                timestamp=time.time(),
                                latency=batch_response.latency,
                                success=True
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
                            error=batch_response.error
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
                        error=str(e)
                    )
                    responses.append(response)
        
        batch_time = time.time() - start_time
        
        return TranslationBatchResponse(
            items=responses,
            total_characters=total_chars,
            avg_confidence=None,  # DeepL doesn't provide confidence scores
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
