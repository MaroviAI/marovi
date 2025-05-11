"""
Marovi API client for unified access to LLM, translation, and custom services.

This module provides the MaroviAPI class, which serves as the main entry point
for interacting with the Marovi API. It wraps the Router class to provide a
more user-friendly interface for accessing services.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union

from .base import ServiceType
from .router import Router, default_router
from ..utils.auth import api_key_manager
from ..providers.provider_registry import provider_registry
from ..config import settings

@dataclass
class ModelInfo:
    """Information about a model."""
    id: str
    name: str
    provider: str
    type: str
    max_tokens: Optional[int] = None
    capabilities: Optional[List[str]] = None
    description: Optional[str] = None

class MaroviAPI:
    """
    Unified API client for Marovi services.
    
    This class provides a user-friendly interface for accessing all Marovi services,
    including LLM, translation, and custom endpoints. It serves as a thin wrapper
    around the Router class, simplifying common operations.
    
    Features:
    - Single API key for all services
    - Automatic service initialization
    - Easy access to all providers and models
    - Provider and model discovery
    - Simple interface for common operations
    """
    
    def __init__(self, api_key: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the API client.
        
        Args:
            api_key: Optional unified API key for all services
            config: Optional additional configuration
        """
        # Initialize configuration
        self.config = config or {}
        
        # Initialize the API key
        self.api_key = api_key
        if api_key:
            api_key_manager.set_unified_api_key(api_key)
        
        # Initialize the router
        self.router = Router(provider_registry)
        
        # Register default providers
        try:
            from ..providers import register_default_providers
            providers_registered = register_default_providers(self.router)
            
            if not providers_registered:
                import logging
                logging.getLogger(__name__).warning(
                    "No providers were registered. The API will have limited functionality. "
                    "Check API keys and dependencies."
                )
        except ImportError as e:
            import logging
            logging.getLogger(__name__).warning(
                f"Could not register default providers: {e}. "
                "The API will have limited functionality."
            )
            
        # Initialize custom endpoints registry
        try:
            # We import here to avoid circular imports, but the import itself
            # triggers the registration of default endpoints due to the initialization
            # code in the custom module's __init__.py
            from ..custom import default_registry
            self.router.set_custom_registry(default_registry)
        except ImportError as e:
            import logging
            logging.getLogger(__name__).warning(
                f"Could not initialize custom endpoints: {e}. "
                "Custom endpoints will not be available."
            )
    
    def get_service(self, 
                  service_type: Union[ServiceType, str], 
                  provider: Optional[str] = None,
                  model: Optional[str] = None) -> Any:
        """
        Get a service by type, provider, and model.
        
        Args:
            service_type: Service type (LLM, TRANSLATION, CUSTOM)
            provider: Optional provider name (uses default if not specified)
            model: Optional model name (uses default if not specified)
            
        Returns:
            Service instance
        """
        # Convert string service type to enum if needed
        if isinstance(service_type, str):
            service_type = ServiceType(service_type.lower())
        
        return self.router.get_service(service_type, provider, model)
    
    def list_providers(self, service_type: Optional[Union[ServiceType, str]] = None) -> Dict[str, Dict[str, Any]]:
        """
        List available providers by service type.
        
        Args:
            service_type: Optional service type to filter by
            
        Returns:
            Dictionary mapping provider IDs to provider information
        """
        # Convert string service type to enum if needed
        if isinstance(service_type, str):
            service_type = ServiceType(service_type.lower())
        
        return self.router.list_available_providers(service_type)
    
    def list_models(self, provider: str, service_type: Optional[Union[ServiceType, str]] = None) -> List[str]:
        """
        List available models for a provider.
        
        Args:
            provider: Provider ID
            service_type: Optional service type (inferred from provider if not specified)
            
        Returns:
            List of model IDs
        """
        # Infer service type if not provided
        if not service_type:
            provider_info = provider_registry.get_provider_info(provider)
            if provider_info:
                services = provider_info.get('services', [])
                if services and isinstance(services[0], dict):
                    service_type = services[0].get('type', 'llm')
        
        # Convert string service type to enum if needed
        if isinstance(service_type, str):
            service_type = ServiceType(service_type.lower())
        
        return self.router.get_provider_models(provider, service_type)
    
    def get_provider_features(self, provider: str, service_type: Optional[Union[ServiceType, str]] = None) -> List[str]:
        """
        Get features supported by a provider.
        
        Args:
            provider: Provider ID
            service_type: Optional service type (inferred from provider if not specified)
            
        Returns:
            List of supported features
        """
        # Infer service type if not provided
        if not service_type:
            provider_info = provider_registry.get_provider_info(provider)
            if provider_info:
                services = provider_info.get('services', [])
                if services and isinstance(services[0], dict):
                    service_type = services[0].get('type', 'llm')
        
        # Convert string service type to enum if needed
        if isinstance(service_type, str):
            service_type = ServiceType(service_type.lower())
        
        return self.router.get_provider_features(provider, service_type)
    
    def get_model_info(self, model_id: str, provider: str, service_type: Optional[Union[ServiceType, str]] = None) -> ModelInfo:
        """
        Get detailed information about a model.
        
        Args:
            model_id: Model ID
            provider: Provider ID
            service_type: Optional service type (inferred from provider if not specified)
            
        Returns:
            ModelInfo instance with model details
        """
        # Infer service type if not provided
        if not service_type:
            provider_info = provider_registry.get_provider_info(provider)
            if provider_info:
                services = provider_info.get('services', [])
                if services and isinstance(services[0], dict):
                    service_type = services[0].get('type', 'llm')
        
        # Get provider information
        provider_info = provider_registry.get_provider_info(provider)
        if not provider_info:
            raise ValueError(f"Unknown provider: {provider}")
        
        # Find model information
        model_info = None
        for service in provider_info.get('services', []):
            if isinstance(service, dict) and service.get('type') == service_type:
                models = service.get('models', [])
                for model in models:
                    if isinstance(model, dict) and model.get('name') == model_id:
                        model_info = model
                        break
        
        if not model_info:
            # Create basic model info if not found
            return ModelInfo(
                id=model_id,
                name=model_id,
                provider=provider,
                type=service_type if isinstance(service_type, str) else service_type.value
            )
        
        # Create model info from model data
        return ModelInfo(
            id=model_id,
            name=model_info.get('name', model_id),
            provider=provider,
            type=service_type if isinstance(service_type, str) else service_type.value,
            max_tokens=model_info.get('max_tokens'),
            capabilities=[
                "streaming" if model_info.get('supports_streaming') else None,
                "json_mode" if model_info.get('supports_json_mode') else None
            ],
            description=model_info.get('description')
        )
    
    @property
    def llm(self) -> Any:
        """Get the default LLM service."""
        return self.router.get_llm()
    
    @property
    def translation(self) -> Any:
        """Get the default translation service."""
        return self.router.get_translation()
    
    @property
    def custom(self) -> Any:
        """Get the custom client proxy that provides access to all custom endpoints via dot notation."""
        return self.router.custom
    
    def get_custom_endpoint(self, name: str, *args, **kwargs) -> Any:
        """
        Get a custom endpoint by name.
        
        Args:
            name: Name of the custom endpoint
            *args: Optional positional arguments to pass to the factory
            **kwargs: Optional keyword arguments to pass to the factory
            
        Returns:
            Custom endpoint instance
        """
        return self.router.get_custom_endpoint(name, **kwargs)
    
    def list_custom_endpoints(self) -> List[str]:
        """
        List all available custom endpoints.
        
        Returns:
            List of custom endpoint names
        """
        return self.router.list_custom_endpoints()
    

# Create a default API client instance for easy access
default_client = MaroviAPI(api_key=settings.MAROVI_API_KEY)

# Global alias for convenient access
api = default_client 