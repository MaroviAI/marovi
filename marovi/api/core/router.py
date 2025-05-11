"""
Router for service discovery and instantiation.

The Router class provides a unified interface for discovering and instantiating
services across different providers, handling both core services and custom endpoints.
"""

import logging
from typing import Dict, Optional, List, Any, Type, Union

from .base import ServiceType
from ..providers.provider_registry import ProviderRegistry, provider_registry
from ..config import get_api_key, get_default_model, settings, ProviderType

logger = logging.getLogger(__name__)

class Router:
    """
    Router for service discovery and instantiation.
    
    The Router provides a unified interface for discovering and creating service 
    instances for different providers and types. It handles API keys, default models,
    and service initialization.
    """
    
    def __init__(self, provider_registry: Optional[ProviderRegistry] = None):
        """
        Initialize the router.
        
        Args:
            provider_registry: Provider registry for service discovery
        """
        self.provider_registry = provider_registry or globals().get('provider_registry')
        self._clients: Dict[str, Dict[str, Any]] = {}
        self._custom_registry = None
        self._custom_client_proxy = None
    
    @classmethod
    def create(cls, api_key: Optional[str] = None, default_providers: bool = False) -> 'Router':
        """
        Create a new Router instance with optional configuration.
        
        Args:
            api_key: Optional API key to use for all services
            default_providers: Whether to initialize with default providers
            
        Returns:
            Configured Router instance
        """
        router = cls(provider_registry)
        
        # Set up default providers if requested
        if default_providers:
            # Import here to avoid circular imports
            from ..providers import register_default_providers
            register_default_providers(router)
            
        return router
    
    def add_provider(self, provider):
        """
        Add a provider to the router.
        
        Args:
            provider: Provider instance to add
            
        Raises:
            ValueError: If the provider cannot be added
        """
        if hasattr(provider, 'get_provider_info'):
            provider_info = provider.get_provider_info()
            provider_id = provider_info.get('id')
            if provider_id:
                # Register the provider with the registry
                self.provider_registry.register_provider(provider)
                
                # Clear any cached clients for this provider
                for client_key in list(self._clients.keys()):
                    if client_key.startswith(f"{provider_id}:"):
                        del self._clients[client_key]
                        
                return True
        
        raise ValueError(f"Failed to add provider: {provider}")
    
    def set_custom_registry(self, registry):
        """
        Set the custom endpoint registry.
        
        Args:
            registry: Custom endpoint registry
        """
        self._custom_registry = registry
        # Reset the custom client proxy to use the new registry
        self._custom_client_proxy = None
    
    @property
    def custom(self):
        """
        Get the custom client proxy for accessing custom endpoints.
        
        Returns:
            CustomClientProxy instance
        """
        if not self._custom_client_proxy:
            # Lazy import to avoid circular dependency
            from ..clients.custom import create_custom_client_proxy
            
            if not self._custom_registry:
                # Lazy import to avoid circular dependency
                from ..custom import default_registry
                self._custom_registry = default_registry
            
            self._custom_client_proxy = create_custom_client_proxy(registry=self._custom_registry)
        
        return self._custom_client_proxy
    
    def get_service(
        self, 
        service_type: ServiceType, 
        provider: Optional[str] = None, 
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs
    ) -> Any:
        """
        Get a service instance.
        
        Args:
            service_type: Type of service to get
            provider: Optional provider to use
            model: Optional model to use
            api_key: Optional API key to use
            **kwargs: Additional options for service initialization
            
        Returns:
            Service instance
        """
        # For custom endpoints, use the custom client proxy
        if service_type == ServiceType.CUSTOM:
            # If provider is specified, it's the name of the custom endpoint
            if provider:
                return self.custom.__getattr__(provider)
            return self.custom
        
        # Determine provider and API key if not provided
        if not provider:
            if service_type == ServiceType.LLM:
                provider = settings.DEFAULT_LLM_PROVIDER
            elif service_type == ServiceType.TRANSLATION:
                provider = settings.DEFAULT_TRANSLATION_PROVIDER
            else:
                provider = "default_provider"
        
        if not api_key:
            api_key = get_api_key(provider)
        
        # Get or create service client
        client_key = f"{provider}:{service_type.value}:{model or 'default'}"
        
        if client_key not in self._clients:
            if service_type == ServiceType.LLM:
                # Import here to avoid circular imports
                from ..clients.llm import LLMClient
                # Use kwargs to pass model and other options
                client_kwargs = {'provider': provider, 'api_key': api_key}
                if model:
                    client_kwargs['model'] = model
                # Add any additional kwargs
                client_kwargs.update(kwargs)
                self._clients[client_key] = LLMClient(**client_kwargs)
            elif service_type == ServiceType.TRANSLATION:
                # Import here to avoid circular imports
                from ..clients.translation import TranslationClient
                self._clients[client_key] = TranslationClient(provider=provider, api_key=api_key, **kwargs)
            else:
                raise ValueError(f"Unsupported service type: {service_type}")
        
        return self._clients[client_key]
    
    def get_llm(self, provider: Optional[str] = None, **kwargs) -> Any:
        """
        Get an LLM client for a provider.
        
        Args:
            provider: Provider to use (default from settings if not specified)
            **kwargs: Additional options for client initialization
            
        Returns:
            LLM client
        """
        return self.get_service(ServiceType.LLM, provider, **kwargs)
    
    def get_translation(self, provider: Optional[str] = None, **kwargs) -> Any:
        """
        Get a translation client for a provider.
        
        Args:
            provider: Provider to use (default from settings if not specified)
            **kwargs: Additional options for client initialization
            
        Returns:
            Translation client
        """
        return self.get_service(ServiceType.TRANSLATION, provider, **kwargs)
    
    def get_custom_endpoint(self, endpoint_name: str, **kwargs) -> Any:
        """
        Get a custom endpoint by name.
        
        Args:
            endpoint_name: Name of the custom endpoint
            **kwargs: Additional options for endpoint initialization
            
        Returns:
            Custom endpoint instance
        """
        return self.get_service(ServiceType.CUSTOM, endpoint_name, **kwargs)
    
    def list_custom_endpoints(self) -> List[str]:
        """
        List all available custom endpoints.
        
        Returns:
            List of custom endpoint names
        """
        return list(dir(self.custom))
    
# Create the default router
default_router = Router(provider_registry)
