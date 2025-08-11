"""Configuration management for the Marovi API.

This module handles configuration for services and providers, including
API key management and provider-specific settings.
"""

import os
try:
    import yaml
except Exception:  # pragma: no cover - allow running without PyYAML
    yaml = None
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass
from enum import Enum

# Try to import from pydantic-settings, but fall back to simple dict if unavailable
try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
    from pydantic import Field
    
    class APISettings(BaseSettings):
        """API settings."""
        
        # General settings
        DEBUG: bool = Field(default=False, description="Enable debug mode")
        LOG_LEVEL: str = Field(default="INFO", description="Logging level")
        
        # Unified API key (takes precedence over provider-specific keys)
        MAROVI_API_KEY: Optional[str] = Field(default="marovi-api-key", description="Single API key for all services")
        
        # OpenAI settings
        OPENAI_API_KEY: Optional[str] = Field(default=None, description="OpenAI API key")
        OPENAI_DEFAULT_MODEL: str = Field(default="gpt-4o-2024-08-06", description="Default OpenAI model")
        
        # Anthropic settings
        ANTHROPIC_API_KEY: Optional[str] = Field(default=None, description="Anthropic API key")
        ANTHROPIC_DEFAULT_MODEL: str = Field(default="claude-3-sonnet-20240229", description="Default Anthropic model")
        
        # Google settings
        GOOGLE_API_KEY: Optional[str] = Field(default=None, description="Google API key")
        GOOGLE_TRANSLATE_API_KEY: Optional[str] = Field(default=None, description="Google Translate API key")
        
        # DeepL settings
        DEEPL_API_KEY: Optional[str] = Field(default=None, description="DeepL API key")
        
        # Default providers
        DEFAULT_LLM_PROVIDER: str = Field(default="openai", description="Default LLM provider")
        DEFAULT_TRANSLATION_PROVIDER: str = Field(default="google", description="Default translation provider")
        
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
            
            # Unified API key
            self.MAROVI_API_KEY = os.getenv("MAROVI_API_KEY")
            
            # OpenAI settings
            self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
            self.OPENAI_DEFAULT_MODEL = os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4o-2024-08-06")
            
            # Anthropic settings
            self.ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
            self.ANTHROPIC_DEFAULT_MODEL = os.getenv("ANTHROPIC_DEFAULT_MODEL", "claude-3-sonnet-20240229")
            
            # Google settings
            self.GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
            self.GOOGLE_TRANSLATE_API_KEY = os.getenv("GOOGLE_TRANSLATE_API_KEY")
            
            # DeepL settings
            self.DEEPL_API_KEY = os.getenv("DEEPL_API_KEY")
            
            # Default providers
            self.DEFAULT_LLM_PROVIDER = os.getenv("DEFAULT_LLM_PROVIDER", "openai")
            self.DEFAULT_TRANSLATION_PROVIDER = os.getenv("DEFAULT_TRANSLATION_PROVIDER", "google")
            
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

class ProviderType(Enum):
    """Types of providers available."""
    LLM = "llm"
    TRANSLATION = "translation"
    CUSTOM = "custom"

@dataclass
class ProviderInfo:
    """Information about a provider."""
    name: str
    type: ProviderType
    models: List[str]
    default_model: str
    features: List[str]
    requires_key: bool = True

class ServiceConfig:
    """Configuration for Marovi services."""
    
    # Default provider configurations
    DEFAULT_PROVIDERS = {
        "openai": ProviderInfo(
            name="openai",
            type=ProviderType.LLM,
            models=["gpt-4o-2024-08-06", "gpt-4o-mini-2024-07-18", "gpt-4-turbo-2024-04-09", "gpt-3.5-turbo-0125"],
            default_model="gpt-4o-2024-08-06",
            features=["completion", "streaming", "function_calling"]
        ),
        "anthropic": ProviderInfo(
            name="anthropic",
            type=ProviderType.LLM,
            models=["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"],
            default_model="claude-3-sonnet-20240229",
            features=["completion", "streaming"]
        ),
        "google": ProviderInfo(
            name="google",
            type=ProviderType.TRANSLATION,
            models=["nmt"],
            default_model="nmt",
            features=["translation", "language_detection", "batch_translation"]
        ),
        "google_llm": ProviderInfo(
            name="google",
            type=ProviderType.LLM,
            models=["gemini-1.5-pro", "gemini-1.5-flash"],
            default_model="gemini-1.5-pro",
            features=["completion", "streaming"]
        ),
        "deepl": ProviderInfo(
            name="deepl",
            type=ProviderType.TRANSLATION,
            models=["v2"],
            default_model="v2",
            features=["translation", "glossary_support", "batch_translation", "formality_control"]
        ),
        "custom": ProviderInfo(
            name="custom",
            type=ProviderType.CUSTOM,
            models=["default"],
            default_model="default",
            features=["translation", "refinement", "glossary_support"],
            requires_key=False
        )
    }
    
    def __init__(self, api_key: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        """Initialize service configuration.
        
        Args:
            api_key: Single API key for all services
            config: Optional additional configuration
        """
        self.api_key = api_key or settings.MAROVI_API_KEY
        self.config = config or {}
        
        # Load provider settings
        self.default_llm_provider = self.config.get("default_llm_provider", settings.DEFAULT_LLM_PROVIDER)
        self.default_translation_provider = self.config.get("default_translation_provider", settings.DEFAULT_TRANSLATION_PROVIDER)
        
        # Load model settings
        self.default_models = self.config.get("default_models", {})
        
        # Provider-specific configurations
        self.provider_configs = self.config.get("provider_configs", {})
        
        # Load provider info from registry.yaml if available
        self._load_provider_registry()
    
    def _load_provider_registry(self):
        """Load provider registry data from registry.yaml if it exists."""
        registry_path = os.path.join(os.path.dirname(__file__), "providers", "registry.yaml")
        if os.path.exists(registry_path):
            try:
                with open(registry_path, 'r') as f:
                    registry_data = yaml.safe_load(f)
                    if registry_data and 'providers' in registry_data:
                        # Update provider info based on registry
                        for provider_id, provider_data in registry_data['providers'].items():
                            services = provider_data.get('services', [])
                            for service in services:
                                if isinstance(service, dict):
                                    service_type = service.get('type')
                                    if service_type:
                                        key = f"{provider_id}" if service_type == "llm" else f"{provider_id}_{service_type}"
                                        models = service.get('models', [])
                                        model_names = []
                                        if models and isinstance(models[0], dict):
                                            model_names = [model.get('name') for model in models if isinstance(model, dict) and 'name' in model]
                                        else:
                                            model_names = models
                                        
                                        self.DEFAULT_PROVIDERS[key] = ProviderInfo(
                                            name=provider_data.get('name', provider_id),
                                            type=getattr(ProviderType, service_type.upper()) if hasattr(ProviderType, service_type.upper()) else ProviderType.CUSTOM,
                                            models=model_names,
                                            default_model=service.get('default_model', model_names[0] if model_names else "default"),
                                            features=service.get('features', []),
                                            requires_key=bool(provider_data.get('env_vars', []))
                                        )
            except Exception as e:
                import logging
                logging.warning(f"Failed to load provider registry from {registry_path}: {e}")
    
    def get_api_key(self, provider_name: str) -> Optional[str]:
        """Get API key for a provider."""
        if not self.DEFAULT_PROVIDERS.get(provider_name, ProviderInfo(
            name="unknown", 
            type=ProviderType.CUSTOM, 
            models=[], 
            default_model="", 
            features=[]
        )).requires_key:
            return None
            
        # First try unified API key
        if self.api_key:
            return self.api_key
        
        # Then try provider-specific key from environment
        return get_api_key(provider_name)
    
    def get_provider_info(self, provider_name: str) -> ProviderInfo:
        """Get information about a provider."""
        return self.DEFAULT_PROVIDERS.get(provider_name, None)
    
    def get_default_model(self, provider_name: str) -> str:
        """Get default model for a provider."""
        provider_info = self.get_provider_info(provider_name)
        if not provider_info:
            return "default"
        
        return (
            self.default_models.get(provider_name) or
            provider_info.default_model
        )
    
    def get_available_providers(self, provider_type: Optional[ProviderType] = None) -> List[str]:
        """Get list of available providers."""
        if provider_type:
            return [
                name for name, info in self.DEFAULT_PROVIDERS.items()
                if info.type == provider_type
            ]
        return list(self.DEFAULT_PROVIDERS.keys())
    
    def get_available_models(self, provider_name: str) -> List[str]:
        """Get list of available models for a provider."""
        provider_info = self.get_provider_info(provider_name)
        if not provider_info:
            return []
        return provider_info.models
    
    def get_provider_features(self, provider_name: str) -> List[str]:
        """Get list of features supported by a provider."""
        provider_info = self.get_provider_info(provider_name)
        if not provider_info:
            return []
        return provider_info.features

def load_config(config_path: Optional[str] = None) -> ServiceConfig:
    """Load configuration from file or environment."""
    if config_path and os.path.exists(config_path) and yaml:
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    
    return ServiceConfig(config.get("api_key"), config)

def get_api_key(provider: str) -> Optional[str]:
    """Get API key for a provider.
    
    First checks for unified MAROVI_API_KEY, then provider-specific keys.
    
    Args:
        provider: Provider name (e.g., "openai", "anthropic", "google")
        
    Returns:
        API key if found, otherwise None
    """
    # Check unified API key first
    if settings.MAROVI_API_KEY:
        return settings.MAROVI_API_KEY
    
    # Fall back to provider-specific keys
    if provider == "openai":
        return settings.OPENAI_API_KEY
    elif provider == "anthropic":
        return settings.ANTHROPIC_API_KEY
    elif provider in ("google", "google_llm", "gemini"):
        return settings.GOOGLE_API_KEY or settings.GOOGLE_TRANSLATE_API_KEY
    elif provider == "deepl":
        return settings.DEEPL_API_KEY
    
    return None

def get_default_model(provider: str) -> str:
    """Get default model for a provider.
    
    Args:
        provider: Provider name
        
    Returns:
        Default model name
    """
    if provider == "openai":
        return settings.OPENAI_DEFAULT_MODEL
    elif provider == "anthropic":
        return settings.ANTHROPIC_DEFAULT_MODEL
    elif provider == "google_llm" or provider == "gemini":
        return "gemini-1.5-pro"
    elif provider == "google":
        return "nmt"
    elif provider == "deepl":
        return "v2"
    else:
        return "default"

def get_provider_features(provider: str) -> List[str]:
    """Get supported features for a provider.
    
    Args:
        provider: Provider name
        
    Returns:
        List of supported features
    """
    features = {
        "openai": ["completion", "streaming", "function_calling"],
        "anthropic": ["completion", "streaming"],
        "google": ["translation", "language_detection", "batch_translation"],
        "google_llm": ["completion", "streaming"],
        "gemini": ["completion", "streaming"],
        "deepl": ["translation", "batch_translation", "glossary_support", "formality_control"],
        "custom": ["translation", "refinement", "glossary_support"]
    }
    
    return features.get(provider, [])

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
