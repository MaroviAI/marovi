"""
Configuration module for the API.

This module handles environment variables and settings for the API.
"""

import os
from typing import Dict, Any, Optional, cast

# Try to import from pydantic-settings, but fall back to simple dict if unavailable
try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
    from pydantic import Field
    
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
        
        model_config = SettingsConfigDict(
            env_file=".env",
            env_file_encoding="utf-8",
            case_sensitive=True
        )
        
    # Create global settings instance
    settings = APISettings()
    
except ImportError:
    # Fall back to a simple dict-based settings object when pydantic-settings is not available
    import os
    
    class SimpleSettings:
        """Simple settings implementation without pydantic dependency."""
        
        def __init__(self):
            # General settings
            self.DEBUG = self._get_bool_env("DEBUG", False)
            self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
            
            # OpenAI settings
            self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
            self.OPENAI_DEFAULT_MODEL = os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4")
            
            # Anthropic settings
            self.ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
            self.ANTHROPIC_DEFAULT_MODEL = os.getenv("ANTHROPIC_DEFAULT_MODEL", "claude-3-sonnet-20240229")
            
            # Google Translate settings
            self.GOOGLE_TRANSLATE_API_KEY = os.getenv("GOOGLE_TRANSLATE_API_KEY")
            
            # DeepL settings
            self.DEEPL_API_KEY = os.getenv("DEEPL_API_KEY")
            
            # Retry settings
            self.DEFAULT_MAX_RETRIES = int(os.getenv("DEFAULT_MAX_RETRIES", "3"))
            self.DEFAULT_INITIAL_BACKOFF = float(os.getenv("DEFAULT_INITIAL_BACKOFF", "1.0"))
            self.DEFAULT_MAX_BACKOFF = float(os.getenv("DEFAULT_MAX_BACKOFF", "10.0"))
            self.DEFAULT_BACKOFF_FACTOR = float(os.getenv("DEFAULT_BACKOFF_FACTOR", "2.0"))
            
            # Cache settings
            self.DEFAULT_CACHE_SIZE = int(os.getenv("DEFAULT_CACHE_SIZE", "100"))
            self.ENABLE_CACHE = self._get_bool_env("ENABLE_CACHE", False)
            
            # Concurrency settings
            self.DEFAULT_MAX_CONCURRENCY = int(os.getenv("DEFAULT_MAX_CONCURRENCY", "5"))
        
        def _get_bool_env(self, key: str, default: bool) -> bool:
            """Helper to get boolean from environment variable."""
            val = os.getenv(key, str(default).lower())
            return val.lower() in ("true", "1", "yes", "y", "t")
    
    # Create global settings instance
    settings = SimpleSettings()

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
