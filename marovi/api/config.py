"""
Configuration module for the API.

This module handles environment variables and settings for the API.
"""

import os
from typing import Dict, Any, Optional
from pydantic import BaseSettings, Field

class APISettings(BaseSettings):
    """API settings."""
    
    # General settings
    DEBUG: bool = Field(default=False, description="Enable debug mode")
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    
    # OpenAI settings
    OPENAI_API_KEY: Optional[str] = Field(default=None, description="OpenAI API key")
    OPENAI_DEFAULT_MODEL: str = Field(default="gpt-4", description="Default OpenAI model")
    
    # Anthropic settings
    ANTHROPIC_API_KEY: Optional[str] = Field(default=None, description="Anthropic API key")
    ANTHROPIC_DEFAULT_MODEL: str = Field(default="claude-3-sonnet-20240229", description="Default Anthropic model")
    
    # Google Translate settings
    GOOGLE_TRANSLATE_API_KEY: Optional[str] = Field(default=None, description="Google Translate API key")
    
    # DeepL settings
    DEEPL_API_KEY: Optional[str] = Field(default=None, description="DeepL API key")
    
    # Retry settings
    DEFAULT_MAX_RETRIES: int = Field(default=3, description="Default maximum number of retries")
    DEFAULT_INITIAL_BACKOFF: float = Field(default=1.0, description="Default initial backoff time in seconds")
    DEFAULT_MAX_BACKOFF: float = Field(default=10.0, description="Default maximum backoff time in seconds")
    DEFAULT_BACKOFF_FACTOR: float = Field(default=2.0, description="Default backoff factor")
    
    # Cache settings
    DEFAULT_CACHE_SIZE: int = Field(default=100, description="Default cache size")
    ENABLE_CACHE: bool = Field(default=False, description="Enable response caching")
    
    # Concurrency settings
    DEFAULT_MAX_CONCURRENCY: int = Field(default=5, description="Default maximum number of concurrent requests")
    
    class Config:
        """Pydantic config."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

# Create global settings instance
settings = APISettings()

def get_retry_config() -> Dict[str, Any]:
    """Get default retry configuration."""
    return {
        "max_retries": settings.DEFAULT_MAX_RETRIES,
        "initial_backoff": settings.DEFAULT_INITIAL_BACKOFF,
        "max_backoff": settings.DEFAULT_MAX_BACKOFF,
        "backoff_factor": settings.DEFAULT_BACKOFF_FACTOR,
        "retryable_errors": [
            "rate_limit_exceeded",
            "server_error",
            "connection_error",
            "timeout"
        ]
    }
