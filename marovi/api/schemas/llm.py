"""
LLM-specific schema classes for requests and responses.

This module provides schema classes specific to LLM services.
"""

from typing import Dict, Optional, List, Any, Union
from pydantic import BaseModel, Field

from .base import BaseRequest, BaseResponse, BatchRequest, BatchResponse

class LLMRequest(BaseRequest):
    """Schema for LLM completion requests."""
    prompt: str = Field(..., description="The prompt to send to the LLM")
    model: Optional[str] = Field(default=None, description="The model to use")
    temperature: float = Field(default=0.1, description="Sampling temperature")
    max_tokens: int = Field(default=8000, description="Maximum number of tokens to generate")
    system_prompt: Optional[str] = Field(default=None, description="System prompt for the model")
    stop_sequences: Optional[List[str]] = Field(default=None, description="Sequences that stop generation")
    top_p: Optional[float] = Field(default=None, description="Top-p sampling parameter")
    frequency_penalty: Optional[float] = Field(default=None, description="Frequency penalty parameter")
    presence_penalty: Optional[float] = Field(default=None, description="Presence penalty parameter")
    seed: Optional[int] = Field(default=None, description="Random seed for generation")
    response_format: Optional[Dict[str, Any]] = Field(default=None, description="Format for the response")

class LLMResponse(BaseResponse):
    """Schema for LLM completion responses."""
    content: str = Field(..., description="Generated text content")
    model: str = Field(..., description="Model used for generation")
    usage: Dict[str, int] = Field(..., description="Token usage statistics")
    finish_reason: Optional[str] = Field(default=None, description="Reason for generation completion")
    raw_response: Optional[Any] = Field(default=None, description="Raw response from the provider")

class LLMBatchRequest(BatchRequest):
    """Schema for batch LLM completion requests."""
    items: List[LLMRequest] = Field(..., description="List of LLM requests to process")
    max_concurrency: int = Field(default=5, description="Maximum number of concurrent requests")

class LLMBatchResponse(BatchResponse):
    """Schema for batch LLM completion responses."""
    items: List[LLMResponse] = Field(..., description="List of LLM responses")
    total_tokens: int = Field(..., description="Total tokens used across all requests")
    avg_tokens: float = Field(..., description="Average tokens used per request")
