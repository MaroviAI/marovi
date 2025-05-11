"""
OpenAI provider implementation for the LLM client.

This module provides the OpenAIProvider class for interacting with OpenAI's API.
"""

import os
import time
import logging
from typing import List, Dict, Optional, Type, Any, AsyncIterator

from openai import OpenAI, AsyncOpenAI
from pydantic import BaseModel

from ..schemas.llm import LLMRequest, LLMResponse
from .base import LLMProvider

# Configure logging
logger = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
    """OpenAI API provider implementation."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the OpenAI provider."""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided and OPENAI_API_KEY environment variable not set")
        self.client = None
        self.async_client = None
    
    def initialize(self) -> None:
        """Initialize the OpenAI client."""
        self.client = OpenAI(api_key=self.api_key)
        self.async_client = AsyncOpenAI(api_key=self.api_key)
    
    def get_default_model(self) -> str:
        """Get the default model for OpenAI."""
        return "gpt-4o-2024-08-06"
    
    def get_supported_models(self) -> List[str]:
        """Get list of supported models."""
        return [
            "gpt-4o-2024-08-06",
            "gpt-4o-mini-2024-07-18",
            "gpt-4-turbo-2024-04-09",
            "gpt-3.5-turbo-0125"
        ]
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported language codes."""
        # OpenAI doesn't have a concept of language codes since it's not a translation service
        return ["en"]  # Return English as default
    
    def get_features(self) -> List[str]:
        """Get the features supported by this provider."""
        return [
            "chat_completions",
            "streaming",
            "json_mode",
            "function_calling",
            "vision"
        ]
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about the provider."""
        return {
            "id": "openai",
            "name": "OpenAI",
            "description": "OpenAI API provider for text and chat completion",
            "website": "https://openai.com",
            "auth_type": "api_key",
            "services": [
                {
                    "type": "llm",
                    "models": [
                        {
                            "name": "gpt-4o-2024-08-06",
                            "description": "Most advanced GPT-4 model with vision capabilities",
                            "max_tokens": 128000,
                            "supports_streaming": True,
                            "supports_json_mode": True
                        },
                        {
                            "name": "gpt-4o-mini-2024-07-18",
                            "description": "Smaller, faster GPT-4 model with good performance and lower cost",
                            "max_tokens": 128000,
                            "supports_streaming": True,
                            "supports_json_mode": True
                        },
                        {
                            "name": "gpt-4-turbo-2024-04-09",
                            "description": "Previous generation GPT-4 model",
                            "max_tokens": 128000,
                            "supports_streaming": True,
                            "supports_json_mode": True
                        },
                        {
                            "name": "gpt-3.5-turbo-0125",
                            "description": "Efficient GPT-3.5 model",
                            "max_tokens": 16385,
                            "supports_streaming": True,
                            "supports_json_mode": True
                        }
                    ],
                    "features": self.get_features()
                }
            ]
        }
    
    def get_rate_limits(self) -> Dict[str, Any]:
        """Get rate limits and quotas for this provider."""
        return {
            "requests_per_minute": 60,
            "tokens_per_minute": 90000,
            "max_tokens_per_request": 128000,
            "max_requests_per_day": 100000
        }
    
    def _prepare_messages(self, request: LLMRequest) -> List[Dict[str, str]]:
        """Prepare messages for OpenAI API."""
        messages = []
        
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        
        messages.append({"role": "user", "content": request.prompt})
        return messages
    
    def _prepare_params(self, request: LLMRequest, response_model: Optional[Type[BaseModel]] = None) -> Dict[str, Any]:
        """Prepare parameters for OpenAI API call."""
        params = {
            "model": request.model,
            "messages": self._prepare_messages(request),
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        
        # Add optional parameters if provided
        if request.top_p is not None:
            params["top_p"] = request.top_p
        if request.frequency_penalty is not None:
            params["frequency_penalty"] = request.frequency_penalty
        if request.presence_penalty is not None:
            params["presence_penalty"] = request.presence_penalty
        if request.stop_sequences:
            params["stop"] = request.stop_sequences
        if request.seed is not None:
            params["seed"] = request.seed
        
        # Add response format for structured output
        if response_model:
            params["response_format"] = {"type": "json_object"}
        elif request.response_format:
            params["response_format"] = request.response_format
        
        return params
    
    def complete(self, request: LLMRequest, response_model: Optional[Type[BaseModel]] = None) -> LLMResponse:
        """Generate a completion using OpenAI API."""
        if not self.client:
            self.initialize()
        
        start_time = time.time()
        params = self._prepare_params(request, response_model)
        
        try:
            response = self.client.chat.completions.create(**params)
            raw_content = response.choices[0].message.content
            
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
            
            # Extract usage statistics
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
            
            return LLMResponse(
                content=content,
                model=response.model,
                usage=usage,
                latency=time.time() - start_time,
                raw_response=response,
                finish_reason=response.choices[0].finish_reason,
                success=True
            )
            
        except Exception as e:
            logger.error(f"OpenAI API call failed: {str(e)}")
            raise
    
    async def acomplete(self, request: LLMRequest, response_model: Optional[Type[BaseModel]] = None) -> LLMResponse:
        """Generate a completion asynchronously using OpenAI API."""
        if not self.async_client:
            self.initialize()
        
        start_time = time.time()
        params = self._prepare_params(request, response_model)
        
        try:
            response = await self.async_client.chat.completions.create(**params)
            raw_content = response.choices[0].message.content
            
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
            
            # Extract usage statistics
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
            
            return LLMResponse(
                content=content,
                model=response.model,
                usage=usage,
                latency=time.time() - start_time,
                raw_response=response,
                finish_reason=response.choices[0].finish_reason,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Async OpenAI API call failed: {str(e)}")
            raise
    
    async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        """Stream a completion from OpenAI API."""
        if not self.async_client:
            self.initialize()
        
        params = self._prepare_params(request)
        params["stream"] = True
        
        try:
            response_stream = await self.async_client.chat.completions.create(**params)
            
            async for chunk in response_stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"OpenAI streaming failed: {str(e)}")
            raise
            
    def batch_complete(self, requests: List[LLMRequest], response_model: Optional[Type[BaseModel]] = None) -> List[LLMResponse]:
        """Generate completions for a batch of LLM requests."""
        results = []
        for request in requests:
            results.append(self.complete(request, response_model))
        return results
    
    async def abatch_complete(self, requests: List[LLMRequest], response_model: Optional[Type[BaseModel]] = None) -> List[LLMResponse]:
        """Generate completions for a batch of LLM requests asynchronously."""
        results = []
        for request in requests:
            results.append(await self.acomplete(request, response_model))
        return results
