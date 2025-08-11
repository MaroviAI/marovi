"""LiteLLM provider implementation using self-hosted gateway."""

from __future__ import annotations

import os
import time
import logging
from typing import Dict, List, Optional, Type, Any, AsyncIterator

import litellm
from pydantic import BaseModel

from .base import LLMProvider
from ..schemas.llm import LLMRequest, LLMResponse

logger = logging.getLogger(__name__)


class LiteLLMProvider(LLMProvider):
    """LLM provider that routes requests through a LiteLLM gateway."""

    def __init__(self, api_key: Optional[str] = None, api_base: Optional[str] = None) -> None:
        self.api_key = api_key or os.getenv("MAROVI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "MAROVI_API_KEY not provided and environment variable not set"
            )
        self.api_base = api_base or os.getenv("LITELLM_API_BASE")
        if not self.api_base:
            raise ValueError(
                "LITELLM_API_BASE environment variable not set"
            )

    def initialize(self) -> None:  # pragma: no cover - no initialization required
        """LiteLLM uses direct function calls; no client initialization needed."""
        return None

    def get_default_model(self) -> str:
        return "gpt-4o-mini"

    def get_features(self) -> List[str]:
        return [
            "chat_completions",
            "streaming",
            "json_mode",
            "function_calling",
        ]

    def get_provider_info(self) -> Dict[str, Any]:
        return {
            "id": "litellm",
            "name": "LiteLLM",
            "description": "Self-hosted LiteLLM gateway", 
            "auth_type": "api_key",
            "services": [
                {
                    "type": "llm",
                    "models": [
                        {
                            "name": self.get_default_model(),
                            "description": "Gateway default model",
                            "max_tokens": 128000,
                            "supports_streaming": True,
                            "supports_json_mode": True,
                        }
                    ],
                    "features": self.get_features(),
                }
            ],
        }

    def _prepare_messages(self, request: LLMRequest) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})
        return messages

    def _prepare_params(
        self, request: LLMRequest, response_model: Optional[Type[BaseModel]] = None
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "model": request.model,
            "messages": self._prepare_messages(request),
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "api_base": self.api_base,
            "api_key": self.api_key,
        }
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

        if response_model:
            params["response_format"] = {"type": "json_object"}
        elif request.response_format:
            params["response_format"] = request.response_format
        return params

    def complete(
        self, request: LLMRequest, response_model: Optional[Type[BaseModel]] = None
    ) -> LLMResponse:
        start_time = time.time()
        params = self._prepare_params(request, response_model)
        response = litellm.completion(**params)
        raw_content = response["choices"][0]["message"]["content"]
        if response_model:
            content = response_model.model_validate_json(raw_content)
        else:
            content = raw_content
        usage = response.get("usage", {})
        return LLMResponse(
            content=content,
            model=response.get("model", request.model),
            usage=usage,
            latency=time.time() - start_time,
            raw_response=response,
            finish_reason=response["choices"][0].get("finish_reason"),
            success=True,
        )

    async def acomplete(
        self, request: LLMRequest, response_model: Optional[Type[BaseModel]] = None
    ) -> LLMResponse:
        start_time = time.time()
        params = self._prepare_params(request, response_model)
        response = await litellm.acompletion(**params)
        raw_content = response["choices"][0]["message"]["content"]
        if response_model:
            content = response_model.model_validate_json(raw_content)
        else:
            content = raw_content
        usage = response.get("usage", {})
        return LLMResponse(
            content=content,
            model=response.get("model", request.model),
            usage=usage,
            latency=time.time() - start_time,
            raw_response=response,
            finish_reason=response["choices"][0].get("finish_reason"),
            success=True,
        )

    async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        params = self._prepare_params(request)
        async for chunk in litellm.astream_completion(**params):
            if "choices" in chunk:
                delta = chunk["choices"][0]["delta"]
                if delta and delta.get("content"):
                    yield delta["content"]
