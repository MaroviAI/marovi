"""Provider registry for managing provider creation and configuration.

This module provides a centralized registry for creating and managing providers,
ensuring consistent provider initialization and configuration across the API.
"""

import os
import yaml
from typing import Dict, Any, Optional, Type, Union, List
from enum import Enum

from ..config import ProviderType, get_api_key

class ProviderRegistry:
    """Registry for managing provider creation and configuration."""
    
    def __init__(self):
        """Initialize the provider registry."""
        self._providers_info: Dict[str, Dict[str, Any]] = {}
        self._load_registry()
    
    def _load_registry(self) -> None:
        """Load provider registry data from registry.yaml if it exists, or use defaults."""
        registry_path = os.path.join(os.path.dirname(__file__), "providers", "registry.yaml")
        if os.path.exists(registry_path):
            try:
                with open(registry_path, 'r') as f:
                    registry_data = yaml.safe_load(f)
                    if registry_data and 'providers' in registry_data:
                        self._providers_info = registry_data['providers']
                        return
            except Exception as e:
                import logging
                logging.warning(f"Failed to load provider registry from {registry_path}: {e}")
        
        # Fall back to default provider registry
        self._providers_info = {
            "openai": {
                "name": "OpenAI",
                "description": "OpenAI API provider for GPT models",
                "services": [
                    {
                        "type": "llm",
                        "implementation": "marovi.api.providers.openai.OpenAIProvider",
                        "default_model": "gpt-4o-2024-08-06",
                        "models": [
                            {"name": "gpt-4o-2024-08-06", "max_tokens": 128000},
                            {"name": "gpt-4o-mini-2024-07-18", "max_tokens": 128000},
                            {"name": "gpt-4-turbo-2024-04-09", "max_tokens": 128000},
                            {"name": "gpt-3.5-turbo-0125", "max_tokens": 16385}
                        ]
                    }
                ],
                "env_vars": ["OPENAI_API_KEY"]
            },
            "anthropic": {
                "name": "Anthropic",
                "description": "Anthropic API provider for Claude models",
                "services": [
                    {
                        "type": "llm",
                        "implementation": "marovi.api.providers.anthropic.AnthropicProvider",
                        "default_model": "claude-3-sonnet-20240229",
                        "models": [
                            {"name": "claude-3-opus-20240229", "max_tokens": 200000},
                            {"name": "claude-3-sonnet-20240229", "max_tokens": 200000},
                            {"name": "claude-3-haiku-20240307", "max_tokens": 200000}
                        ]
                    }
                ],
                "env_vars": ["ANTHROPIC_API_KEY"]
            },
            "google": {
                "name": "Google",
                "description": "Google AI & Translation services",
                "services": [
                    {
                        "type": "translation",
                        "implementation": "marovi.api.providers.google.GoogleTranslateProvider",
                        "features": ["batch_translation", "language_detection"]
                    },
                    {
                        "type": "llm",
                        "implementation": "marovi.api.providers.google.GeminiProvider",
                        "default_model": "gemini-1.5-pro",
                        "models": [
                            {"name": "gemini-1.5-pro", "max_tokens": 32768},
                            {"name": "gemini-1.5-flash", "max_tokens": 32768}
                        ]
                    }
                ],
                "env_vars": ["GOOGLE_API_KEY", "GOOGLE_TRANSLATE_API_KEY"]
            },
            "deepl": {
                "name": "DeepL",
                "description": "DeepL Translation API",
                "services": [
                    {
                        "type": "translation",
                        "implementation": "marovi.api.providers.deepl.DeepLProvider",
                        "features": ["batch_translation", "formality_control"]
                    }
                ],
                "env_vars": ["DEEPL_API_KEY"]
            },
            "custom": {
                "name": "Custom",
                "description": "Custom implementations",
                "services": [
                    {
                        "type": "translation",
                        "implementation": "marovi.api.providers.custom.ChatGPTTranslationProvider",
                        "description": "Translation provider that uses ChatGPT for translation",
                        "requires": [{"type": "llm", "provider": "openai"}]
                    }
                ]
            }
        }
    
    def get_provider_info(self, provider_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific provider."""
        return self._providers_info.get(provider_id)
    
    def get_all_providers(self) -> Dict[str, Any]:
        """Get all registered providers."""
        return self._providers_info
    
    def get_providers_for_service(self, service_type: str) -> Dict[str, Any]:
        """Get all providers that support a specific service type."""
        result = {}
        for provider_id, provider_info in self._providers_info.items():
            for service in provider_info.get('services', []):
                if isinstance(service, dict) and service.get('type') == service_type:
                    result[provider_id] = provider_info
                    break
        return result
    
    def get_provider_class(self, provider_id: str, service_type: str) -> Optional[Type]:
        """Get the provider implementation class for a specific service type."""
        provider_info = self.get_provider_info(provider_id)
        if not provider_info:
            return None
        
        # Find service matching the requested type
        for service in provider_info.get('services', []):
            if isinstance(service, dict) and service.get('type') == service_type:
                implementation_path = service.get('implementation')
                if implementation_path:
                    try:
                        module_path, class_name = implementation_path.rsplit('.', 1)
                        import importlib
                        module = importlib.import_module(module_path)
                        return getattr(module, class_name)
                    except (ImportError, AttributeError) as e:
                        # Fall back to hardcoded imports if dynamic import fails
                        if provider_id == "openai" and service_type == "llm":
                            from .openai import OpenAIProvider
                            return OpenAIProvider
                        elif provider_id == "anthropic" and service_type == "llm":
                            from .anthropic import AnthropicProvider
                            return AnthropicProvider
                        elif provider_id == "google" and service_type == "translation":
                            from .google import GoogleTranslateProvider
                            return GoogleTranslateProvider
                        elif provider_id == "google_rest" and service_type == "translation":
                            from .google_rest import GoogleTranslateRestProvider
                            return GoogleTranslateRestProvider
                        elif provider_id == "google" and service_type == "llm":
                            from .google import GeminiProvider
                            return GeminiProvider
                        elif provider_id == "gemini_rest" and service_type == "llm":
                            from .google_rest import GeminiRestProvider
                            return GeminiRestProvider
                        elif provider_id == "deepl" and service_type == "translation":
                            from .deepl import DeepLProvider
                            return DeepLProvider
                        elif provider_id == "custom" and service_type == "translation":
                            from .custom import ChatGPTTranslationProvider
                            return ChatGPTTranslationProvider
                        return None
        return None
    
    def create_provider(self, 
                      provider_id: str, 
                      service_type: str = None,
                      api_key: Optional[str] = None,
                      config: Optional[Dict[str, Any]] = None) -> Any:
        """Create a provider instance.
        
        Args:
            provider_id: Provider ID
            service_type: Optional service type (inferred from provider if not specified)
            api_key: Optional API key (falls back to environment variables)
            config: Optional additional configuration
            
        Returns:
            Provider instance
        """
        # Infer service type if not provided
        if not service_type:
            provider_info = self.get_provider_info(provider_id)
            if not provider_info:
                raise ValueError(f"Unknown provider: {provider_id}")
            
            # Get the first service type
            services = provider_info.get("services", [])
            if services and isinstance(services[0], dict):
                service_type = services[0].get("type", "llm")
            else:
                service_type = "llm"  # Default to LLM
        
        # Get provider class
        provider_class = self.get_provider_class(provider_id, service_type)
        if not provider_class:
            raise ValueError(f"Provider implementation not found for {provider_id} with service type {service_type}")
        
        # Use unified API key if available
        if not api_key:
            api_key = get_api_key(provider_id)
        
        # Create provider instance
        provider_config = config or {}
        if api_key:
            provider_config["api_key"] = api_key
        
        # Special handling for custom provider that requires an LLM provider
        if provider_id == "custom" and service_type == "translation":
            # For ChatGPTTranslationProvider, we need to create an OpenAI provider first
            from .openai import OpenAIProvider
            llm_provider = OpenAIProvider(api_key=get_api_key("openai"))
            return provider_class(llm_provider=llm_provider)
        
        return provider_class(**provider_config)
    
    def register_provider(self, provider):
        """
        Register a provider instance with the registry.
        
        Args:
            provider: Provider instance to register
            
        Returns:
            True if registration was successful, False otherwise
            
        Raises:
            ValueError: If provider information is invalid
        """
        if not hasattr(provider, 'get_provider_info'):
            raise ValueError(f"Provider {provider} does not implement get_provider_info method")
            
        # Get provider information
        provider_info = provider.get_provider_info()
        if not provider_info or 'id' not in provider_info:
            raise ValueError(f"Invalid provider information: {provider_info}")
            
        provider_id = provider_info['id']
        
        # Determine service type from provider class name
        service_type = "llm"
        if "Translation" in provider.__class__.__name__:
            service_type = "translation"
            
        # Update or create provider info in registry
        if provider_id not in self._providers_info:
            self._providers_info[provider_id] = {
                "name": provider_info.get('name', provider.__class__.__name__),
                "description": provider_info.get('description', f"{provider.__class__.__name__} provider"),
                "services": []
            }
            
        # Check if service already exists
        service_exists = False
        for service in self._providers_info[provider_id].get('services', []):
            if isinstance(service, dict) and service.get('type') == service_type:
                service_exists = True
                # Update service info
                service['models'] = provider_info.get('models', provider.get_supported_models())
                service['default_model'] = provider_info.get('default_model', provider.get_default_model())
                break
                
        # Add new service if it doesn't exist
        if not service_exists:
            self._providers_info[provider_id]['services'].append({
                "type": service_type,
                "implementation": f"{provider.__class__.__module__}.{provider.__class__.__name__}",
                "default_model": provider_info.get('default_model', provider.get_default_model()),
                "models": provider_info.get('models', provider.get_supported_models())
            })
            
        return True
    
    def get_default_model(self, provider_id: str, service_type: str) -> Optional[str]:
        """Get the default model for a provider and service type."""
        provider_info = self.get_provider_info(provider_id)
        if not provider_info:
            return None
        
        for service in provider_info.get('services', []):
            if isinstance(service, dict) and service.get('type') == service_type:
                return service.get('default_model')
        
        return None
    
    def get_models(self, provider_id: str, service_type: str) -> List[str]:
        """Get all models supported by a provider for a specific service type."""
        provider_info = self.get_provider_info(provider_id)
        if not provider_info:
            return []
        
        for service in provider_info.get('services', []):
            if isinstance(service, dict) and service.get('type') == service_type:
                models = service.get('models', [])
                if models and isinstance(models[0], dict):
                    return [model.get("name") for model in models]
                return models
        
        return []
    
    def get_features(self, provider_id: str, service_type: str) -> List[str]:
        """Get all features supported by a provider for a specific service type."""
        provider_info = self.get_provider_info(provider_id)
        if not provider_info:
            return []
        
        for service in provider_info.get('services', []):
            if isinstance(service, dict) and service.get('type') == service_type:
                return service.get('features', [])
        
        return []

# Global provider registry instance
provider_registry = ProviderRegistry() 