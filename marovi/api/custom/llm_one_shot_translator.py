"""
LLM-based one-shot translation provider.

This module provides a translation provider that uses any LLM provider with structured response support
to perform translations through a carefully crafted prompt and response format.
"""

import time
import os
from typing import Dict, Optional, List, Any, Union
from jinja2 import Environment, FileSystemLoader

from ..schemas.llm import LLMRequest, LLMResponse, LLMBatchRequest, LLMBatchResponse
from ..schemas.translation import (
    TranslationRequest, TranslationResponse, TranslationBatchRequest, TranslationBatchResponse,
    TranslationFormat, GlossaryEntry
)
from ..providers.base import LLMProvider, TranslationProvider
from ..utils.logging import request_logger
from ..utils.retry import retry, async_retry
from ..utils.cache import cached, async_cached
from .schemas.llm_one_shot_translator import TranslationOutput
from .prompts.system_prompts import TranslationSystemPrompts

# Initialize Jinja2 environment
template_dir = os.path.join(os.path.dirname(__file__), "prompts")
env = Environment(loader=FileSystemLoader(template_dir))
template = env.get_template("llm_one_shot_translator.jinja")

class LLMOneShotTranslator(TranslationProvider):
    """
    Translation provider that uses any LLM provider with structured response support.
    
    This provider uses a carefully crafted prompt and structured response format to ensure
    high-quality translations while maintaining consistency and reliability.
    """
    
    def __init__(self, llm_provider: LLMProvider):
        """
        Initialize the LLM one-shot translator.
        
        Args:
            llm_provider: LLM provider that supports structured responses
        """
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
        # LLMs can translate between any languages
        return ["auto", "en", "es", "fr", "de", "it", "pt", "nl", "pl", "ru", "zh", "ja", "ko", "ar", "hi"]
    
    def get_supported_formats(self) -> List[TranslationFormat]:
        """Get list of supported translation formats."""
        return [
            TranslationFormat.TEXT,
            TranslationFormat.HTML,
            TranslationFormat.MARKDOWN,
            TranslationFormat.JSON,
            TranslationFormat.XML
        ]
    
    def get_quality_metrics(self) -> List[str]:
        """Get list of supported quality metrics."""
        return ["confidence", "fluency", "adequacy", "terminology"]
    
    def get_rate_limits(self) -> Dict[str, Any]:
        """Get rate limits and quotas for this provider."""
        return self.llm_provider.get_rate_limits()
    
    def get_supported_domains(self) -> List[str]:
        """Get list of supported translation domains."""
        return ["general", "technical", "medical", "legal", "financial"]
    
    def get_supported_quality_preferences(self) -> List[str]:
        """Get list of supported quality preferences."""
        return ["speed", "accuracy", "balanced"]
    
    def _create_translation_prompt(self, request: TranslationRequest) -> str:
        """Create a prompt for translation using the Jinja template."""
        return template.render(
            text=request.text,
            source_lang=request.source_lang,
            target_lang=request.target_lang,
            format=request.format.value,
            preserve_formatting=request.preserve_formatting,
            glossary=request.glossary,
            base_translation=request.base_translation,
            quality_preference=request.quality_preference,
            domain=request.domain
        )
    
    @retry()
    def translate(self, request: TranslationRequest) -> TranslationResponse:
        """Translate text using the LLM provider."""
        start_time = time.time()
        
        # Create translation prompt
        prompt = self._create_translation_prompt(request)
        
        # Create LLM request with structured output format
        llm_request = LLMRequest(
            prompt=prompt,
            model=self.get_default_model(),
            temperature=0.1,  # Low temperature for more consistent translations
            max_tokens=len(request.text) * 2,  # Estimate max tokens needed
            system_prompt=TranslationSystemPrompts.TRANSLATION,
            response_format={"type": "json_object"},
            metadata=request.metadata
        )
        
        # Get translation from LLM
        llm_response = self.llm_provider.complete(llm_request)
        
        # Parse structured response
        try:
            translation_output = TranslationOutput.parse_raw(llm_response.content)
        except Exception as e:
            raise ValueError(f"Failed to parse LLM translation response: {str(e)}")
        
        # Apply glossary if provided
        if request.glossary:
            translation_output.translated_text = self.apply_glossary(
                translation_output.translated_text,
                request.glossary
            )
        
        # Get quality metrics
        quality_metrics = self._get_quality_metrics(
            request.text,
            translation_output.translated_text
        )
        
        # Create translation response
        response = TranslationResponse(
            content=translation_output.translated_text,
            source_lang=request.source_lang,
            target_lang=request.target_lang,
            detected_lang=translation_output.detected_language,
            confidence=translation_output.confidence,
            quality_metrics=quality_metrics,
            glossary_applied=bool(request.glossary),
            format=request.format,
            metadata=llm_response.metadata,
            timestamp=llm_response.timestamp,
            latency=time.time() - start_time,
            success=True
        )
        
        # Log the translation
        request_logger.log_response(
            service="llm_translation",
            response=response.dict(),
            latency=response.latency,
            metadata=response.metadata
        )
        
        return response
    
    @async_retry()
    async def atranslate(self, request: TranslationRequest) -> TranslationResponse:
        """Translate text asynchronously using the LLM provider."""
        start_time = time.time()
        
        # Create translation prompt
        prompt = self._create_translation_prompt(request)
        
        # Create LLM request with structured output format
        llm_request = LLMRequest(
            prompt=prompt,
            model=self.get_default_model(),
            temperature=0.1,
            max_tokens=len(request.text) * 2,
            system_prompt=TranslationSystemPrompts.TRANSLATION,
            response_format={"type": "json_object"},
            metadata=request.metadata
        )
        
        # Get translation from LLM
        llm_response = await self.llm_provider.acomplete(llm_request)
        
        # Parse structured response
        try:
            translation_output = TranslationOutput.parse_raw(llm_response.content)
        except Exception as e:
            raise ValueError(f"Failed to parse LLM translation response: {str(e)}")
        
        # Apply glossary if provided
        if request.glossary:
            translation_output.translated_text = self.apply_glossary(
                translation_output.translated_text,
                request.glossary
            )
        
        # Get quality metrics
        quality_metrics = self._get_quality_metrics(
            request.text,
            translation_output.translated_text
        )
        
        # Create translation response
        response = TranslationResponse(
            content=translation_output.translated_text,
            source_lang=request.source_lang,
            target_lang=request.target_lang,
            detected_lang=translation_output.detected_language,
            confidence=translation_output.confidence,
            quality_metrics=quality_metrics,
            glossary_applied=bool(request.glossary),
            format=request.format,
            metadata=llm_response.metadata,
            timestamp=llm_response.timestamp,
            latency=time.time() - start_time,
            success=True
        )
        
        # Log the translation
        request_logger.log_response(
            service="llm_translation",
            response=response.dict(),
            latency=response.latency,
            metadata=response.metadata
        )
        
        return response
    
    def batch_translate(self, request: TranslationBatchRequest) -> TranslationBatchResponse:
        """Translate texts in batch using the LLM provider."""
        start_time = time.time()
        responses = []
        total_chars = 0
        failed_items = []
        
        for item in request.items:
            try:
                response = self.translate(item)
                responses.append(response)
                total_chars += len(item.text) if isinstance(item.text, str) else sum(len(t) for t in item.text)
            except Exception as e:
                failed_items.append({
                    "request": item.dict(),
                    "error": str(e)
                })
        
        batch_time = time.time() - start_time
        
        # Calculate aggregated quality metrics
        quality_metrics = {}
        if responses:
            for metric in self.get_quality_metrics():
                values = [r.quality_metrics.get(metric, 0) for r in responses if r.quality_metrics]
                if values:
                    quality_metrics[metric] = sum(values) / len(values)
        
        return TranslationBatchResponse(
            items=responses,
            total_characters=total_chars,
            avg_confidence=sum(r.confidence for r in responses) / len(responses) if responses else None,
            quality_metrics=quality_metrics,
            failed_items=failed_items if failed_items else None,
            metadata=request.metadata,
            timestamp=time.time(),
            total_latency=batch_time,
            avg_latency=batch_time / len(request.items),
            success=True
        )
    
    async def abatch_translate(self, request: TranslationBatchRequest) -> TranslationBatchResponse:
        """Translate texts in batch asynchronously using the LLM provider."""
        start_time = time.time()
        responses = []
        total_chars = 0
        failed_items = []
        
        for item in request.items:
            try:
                response = await self.atranslate(item)
                responses.append(response)
                total_chars += len(item.text) if isinstance(item.text, str) else sum(len(t) for t in item.text)
            except Exception as e:
                failed_items.append({
                    "request": item.dict(),
                    "error": str(e)
                })
        
        batch_time = time.time() - start_time
        
        # Calculate aggregated quality metrics
        quality_metrics = {}
        if responses:
            for metric in self.get_quality_metrics():
                values = [r.quality_metrics.get(metric, 0) for r in responses if r.quality_metrics]
                if values:
                    quality_metrics[metric] = sum(values) / len(values)
        
        return TranslationBatchResponse(
            items=responses,
            total_characters=total_chars,
            avg_confidence=sum(r.confidence for r in responses) / len(responses) if responses else None,
            quality_metrics=quality_metrics,
            failed_items=failed_items if failed_items else None,
            metadata=request.metadata,
            timestamp=time.time(),
            total_latency=batch_time,
            avg_latency=batch_time / len(request.items),
            success=True
        )
    
    def apply_glossary(self, text: str, glossary: List[GlossaryEntry]) -> str:
        """Apply glossary terms to translated text."""
        # Sort glossary entries by length (longest first) to handle overlapping terms
        sorted_glossary = sorted(glossary, key=lambda x: len(x.source_term), reverse=True)
        
        for entry in sorted_glossary:
            if entry.case_sensitive:
                text = text.replace(entry.source_term, entry.target_term)
            else:
                # Case-insensitive replacement
                import re
                pattern = re.compile(re.escape(entry.source_term), re.IGNORECASE)
                text = pattern.sub(entry.target_term, text)
        
        return text
    
    def refine_translation(self, text: str, base_translation: str) -> str:
        """Refine a base translation."""
        prompt = f"""You are a professional translator. Refine the following translation to improve its quality.
        
        Original text:
        {text}
        
        Base translation:
        {base_translation}
        
        Provide your refined translation in the following JSON format:
        {{
            "translated_text": "your refined translation here",
            "confidence": 0.95,  # confidence score between 0 and 1
            "notes": "any additional notes about the refinement"
        }}
        
        Refined translation:"""
        
        llm_request = LLMRequest(
            prompt=prompt,
            model=self.get_default_model(),
            temperature=0.1,
            max_tokens=len(text) * 2,
            system_prompt=TranslationSystemPrompts.REFINEMENT,
            response_format={"type": "json_object"}
        )
        
        llm_response = self.llm_provider.complete(llm_request)
        translation_output = TranslationOutput.parse_raw(llm_response.content)
        
        return translation_output.translated_text
    
    def _get_quality_metrics(self, source: str, target: str) -> Dict[str, float]:
        """Get quality metrics for a translation."""
        # This is a simplified implementation. In practice, you would want to use
        # more sophisticated metrics like BLEU, TER, or custom quality metrics.
        return {
            "confidence": 0.95,  # Placeholder
            "fluency": 0.9,     # Placeholder
            "adequacy": 0.85,   # Placeholder
            "terminology": 0.9  # Placeholder
        }
    
    def detect_language(self, text: str) -> str:
        """Detect the language of the input text."""
        prompt = f"""Detect the language of the following text and provide the ISO 639-1 language code.
        
        Text:
        {text}
        
        Provide your response in the following JSON format:
        {{
            "language_code": "detected language code",
            "confidence": 0.95  # confidence score between 0 and 1
        }}
        
        Language:"""
        
        llm_request = LLMRequest(
            prompt=prompt,
            model=self.get_default_model(),
            temperature=0.1,
            max_tokens=100,
            system_prompt=TranslationSystemPrompts.LANGUAGE_DETECTION,
            response_format={"type": "json_object"}
        )
        
        llm_response = self.llm_provider.complete(llm_request)
        response = llm_response.content
        
        # Parse response and extract language code
        import json
        try:
            result = json.loads(response)
            return result["language_code"]
        except Exception as e:
            raise ValueError(f"Failed to parse language detection response: {str(e)}")
