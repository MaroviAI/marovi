"""
Base schema classes for requests and responses.

This module provides common schema classes that are used across different services.
"""

from typing import Dict, Optional, Any, List
from pydantic import BaseModel, Field
from datetime import datetime

class BaseRequest(BaseModel):
    """Base class for all service requests."""
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata for the request")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Request timestamp")

class BaseResponse(BaseModel):
    """Base class for all service responses."""
    content: Any = Field(..., description="Response content")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata for the response")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    latency: float = Field(..., description="Request latency in seconds")
    success: bool = Field(..., description="Whether the request was successful")
    error: Optional[str] = Field(default=None, description="Error message if request failed")

class BatchRequest(BaseModel):
    """Base class for batch requests."""
    items: List[Any] = Field(..., description="List of items to process")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata for the batch request")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Request timestamp")

class BatchResponse(BaseModel):
    """Base class for batch responses."""
    items: List[Any] = Field(..., description="List of processed items")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata for the batch response")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    total_latency: float = Field(..., description="Total request latency in seconds")
    avg_latency: float = Field(..., description="Average request latency in seconds")
    success: bool = Field(..., description="Whether all items were processed successfully")
    errors: Optional[List[str]] = Field(default=None, description="List of error messages for failed items") 