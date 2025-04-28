"""
Custom provider implementations.

This module provides custom provider implementations for specific use cases.
"""

import time
from typing import Dict, Optional, List, Any, Union
from pydantic import BaseModel

from ..schemas.llm import LLMRequest, LLMResponse, LLMBatchRequest, LLMBatchResponse
from ..schemas.translation import TranslationRequest, TranslationResponse, TranslationBatchRequest, TranslationBatchResponse
from ..providers.base import LLMProvider, TranslationProvider
from ..utils.logging import request_logger
from ..utils.retry import retry, async_retry
from ..utils.cache import cached, async_cached
from ..utils.auth import require_api_key, require_api_key_async

class ChatGPTTranslationProvider(TranslationProvider):
    """Translation provider that uses ChatGPT for translation."""
    
    def __init__(self, llm_provider: LLMProvider):
        """Initialize the ChatGPT translation provider."""
        self.llm_provider = llm_provider
        self.initialize()
    
    def initialize(self) -> None:
        """Initialize the provider."""
        pass
    
    def get_default_model(self) -> str:
        """Get the default model for this provider."""
        return self.llm_provider.get_default_model()
    
    def get_supported_models(self) -> List[str]:
        """Get list of supported models."""
        return self.llm_provider.get_supported_models()
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        # ChatGPT can translate between any languages
        return ["auto", "en", "es", "fr", "de", "it", "pt", "nl", "pl", "ru", "zh", "ja", "ko", "ar", "hi"]
    
    def _create_translation_prompt(self, text: str, source_lang: str, target_lang: str) -> str:
        """Create a prompt for translation."""
        return f"""Translate the following text from {source_lang} to {target_lang}. 
        Preserve any formatting, special characters, and maintain the original meaning.
        
        Text to translate:
        {text}
        
        Translation:"""
    
    @retry()
    def translate(self, request: TranslationRequest) -> TranslationResponse:
        """Translate text using ChatGPT."""
        start_time = time.time()
        
        # Create translation prompt
        prompt = self._create_translation_prompt(
            request.text,
            request.source_lang,
            request.target_lang
        )
        
        # Create LLM request
        llm_request = LLMRequest(
            prompt=prompt,
            model=self.get_default_model(),
            temperature=0.1,  # Low temperature for more consistent translations
            max_tokens=len(request.text) * 2,  # Estimate max tokens needed
            system_prompt="You are a professional translator. Provide only the translation without any additional text or explanations.",
            metadata=request.metadata
        )
        
        # Get translation from LLM
        llm_response = self.llm_provider.complete(llm_request)
        
        # Create translation response
        response = TranslationResponse(
            content=llm_response.content.strip(),
            source_lang=request.source_lang,
            target_lang=request.target_lang,
            detected_lang=request.source_lang if request.source_lang == "auto" else None,
            confidence=1.0,  # ChatGPT doesn't provide confidence scores
            metadata=llm_response.metadata,
            timestamp=llm_response.timestamp,
            latency=time.time() - start_time,
            success=True
        )
        
        # Log the translation
        request_logger.log_response(
            service="chatgpt_translation",
            response=response.dict(),
            latency=response.latency,
            metadata=response.metadata
        )
        
        return response
    
    @async_retry()
    async def atranslate(self, request: TranslationRequest) -> TranslationResponse:
        """Translate text asynchronously using ChatGPT."""
        start_time = time.time()
        
        # Create translation prompt
        prompt = self._create_translation_prompt(
            request.text,
            request.source_lang,
            request.target_lang
        )
        
        # Create LLM request
        llm_request = LLMRequest(
            prompt=prompt,
            model=self.get_default_model(),
            temperature=0.1,
            max_tokens=len(request.text) * 2,
            system_prompt="You are a professional translator. Provide only the translation without any additional text or explanations.",
            metadata=request.metadata
        )
        
        # Get translation from LLM
        llm_response = await self.llm_provider.acomplete(llm_request)
        
        # Create translation response
        response = TranslationResponse(
            content=llm_response.content.strip(),
            source_lang=request.source_lang,
            target_lang=request.target_lang,
            detected_lang=request.source_lang if request.source_lang == "auto" else None,
            confidence=1.0,
            metadata=llm_response.metadata,
            timestamp=llm_response.timestamp,
            latency=time.time() - start_time,
            success=True
        )
        
        # Log the translation
        request_logger.log_response(
            service="chatgpt_translation",
            response=response.dict(),
            latency=response.latency,
            metadata=response.metadata
        )
        
        return response
    
    def batch_translate(self, request: TranslationBatchRequest) -> TranslationBatchResponse:
        """Translate texts in batch using ChatGPT."""
        start_time = time.time()
        responses = []
        total_chars = 0
        
        for item in request.items:
            response = self.translate(item)
            responses.append(response)
            total_chars += len(item.text) if isinstance(item.text, str) else sum(len(t) for t in item.text)
        
        batch_time = time.time() - start_time
        
        return TranslationBatchResponse(
            items=responses,
            total_characters=total_chars,
            avg_confidence=1.0,
            metadata=request.metadata,
            timestamp=time.time(),
            total_latency=batch_time,
            avg_latency=batch_time / len(request.items),
            success=True
        )
    
    async def abatch_translate(self, request: TranslationBatchRequest) -> TranslationBatchResponse:
        """Translate texts in batch asynchronously using ChatGPT."""
        start_time = time.time()
        responses = []
        total_chars = 0
        
        for item in request.items:
            response = await self.atranslate(item)
            responses.append(response)
            total_chars += len(item.text) if isinstance(item.text, str) else sum(len(t) for t in item.text)
        
        batch_time = time.time() - start_time
        
        return TranslationBatchResponse(
            items=responses,
            total_characters=total_chars,
            avg_confidence=1.0,
            metadata=request.metadata,
            timestamp=time.time(),
            total_latency=batch_time,
            avg_latency=batch_time / len(request.items),
            success=True
        ) 