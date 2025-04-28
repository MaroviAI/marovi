"""
Base router for handling LLM, translation, and custom endpoints.

This module provides a unified interface for accessing different types of services
through a single router instance.
"""

import logging
from typing import Dict, Optional, Type, Union, Any, List
from enum import Enum

from .llm import LLMClient, create_llm_client
from .translation import TranslationClient, create_translation_client
from .providers.base import LLMProvider, TranslationProvider
from .provider_registry import provider_registry
from .utils.logging import get_logger

# Configure logging
logger = get_logger(__name__)

class ServiceType(Enum):
    """Supported service types."""
    LLM = "llm"
    TRANSLATION = "translation"
    CUSTOM = "custom"

class Router:
    """
    Unified router for accessing different types of services.
    
    Features:
    - Support for multiple service types (LLM, Translation, Custom)
    - Unified interface for all services
    - Automatic client initialization and management
    - Comprehensive logging and monitoring
    - Extensible architecture for custom endpoints
    - Provider discovery through registry
    """
    
    def __init__(self):
        """Initialize the router."""
        self._clients: Dict[ServiceType, Dict[str, Any]] = {
            ServiceType.LLM: {},
            ServiceType.TRANSLATION: {},
            ServiceType.CUSTOM: {}
        }
        self._custom_endpoints: Dict[str, Any] = {}
        logger.info("Initialized Router")
    
    def register_llm_client(self, 
                          provider: str,
                          model: Optional[str] = None,
                          api_key: Optional[str] = None,
                          custom_provider: Optional[LLMProvider] = None,
                          retry_config: Optional[Dict[str, Any]] = None,
                          enable_cache: bool = False,
                          cache_size: int = 100,
                          client_id: Optional[str] = None) -> str:
        """
        Register an LLM client with the router.
        
        Args:
            provider: Provider ID (e.g., "openai", "anthropic")
            model: Optional model name
            api_key: Optional API key
            custom_provider: Optional custom provider implementation
            retry_config: Optional retry configuration
            enable_cache: Whether to enable response caching
            cache_size: Cache size for responses
            client_id: Optional client ID (auto-generated if not provided)
            
        Returns:
            Client ID for the registered client
        """
        # Validate provider
        if not custom_provider and not provider_registry.get_provider_info(provider):
            raise ValueError(f"Unknown provider: {provider}")
        
        # Generate client ID if not provided
        if not client_id:
            client_id = f"{provider}-{len(self._clients[ServiceType.LLM]) + 1}"
        
        # Get default model if not specified
        if not model and not custom_provider:
            model = provider_registry.get_default_model(provider, "llm")
        
        # Create client
        client = create_llm_client(
            provider=provider,
            model=model,
            api_key=api_key,
            custom_provider=custom_provider,
            retry_config=retry_config,
            enable_cache=enable_cache,
            cache_size=cache_size
        )
        
        # Register client
        self._clients[ServiceType.LLM][client_id] = client
        logger.info(f"Registered LLM client with provider={provider}, client_id={client_id}")
        
        return client_id
    
    def register_translation_client(self,
                                  provider: str,
                                  api_key: Optional[str] = None,
                                  custom_provider: Optional[TranslationProvider] = None,
                                  retry_config: Optional[Dict] = None,
                                  client_id: Optional[str] = None) -> str:
        """
        Register a translation client with the router.
        
        Args:
            provider: Provider ID (e.g., "google", "deepl")
            api_key: Optional API key
            custom_provider: Optional custom provider implementation
            retry_config: Optional retry configuration
            client_id: Optional client ID (auto-generated if not provided)
            
        Returns:
            Client ID for the registered client
        """
        # Validate provider
        if not custom_provider and not provider_registry.get_provider_info(provider):
            raise ValueError(f"Unknown provider: {provider}")
        
        # Generate client ID if not provided
        if not client_id:
            client_id = f"{provider}-{len(self._clients[ServiceType.TRANSLATION]) + 1}"
        
        # Create client
        client = create_translation_client(
            provider=provider,
            api_key=api_key,
            custom_provider=custom_provider,
            retry_config=retry_config
        )
        
        # Register client
        self._clients[ServiceType.TRANSLATION][client_id] = client
        logger.info(f"Registered translation client with provider={provider}, client_id={client_id}")
        
        return client_id
    
    def register_custom_endpoint(self, name: str, handler: Any) -> None:
        """Register a custom endpoint with the router."""
        self._custom_endpoints[name] = handler
        logger.info(f"Registered custom endpoint: {name}")
    
    def get_llm_client(self, client_id: Optional[str] = None) -> LLMClient:
        """
        Get a registered LLM client.
        
        Args:
            client_id: Optional client ID (returns first client if not provided)
            
        Returns:
            LLM client
        """
        llm_clients = self._clients[ServiceType.LLM]
        
        if not llm_clients:
            raise ValueError("No LLM clients registered")
        
        if client_id:
            if client_id not in llm_clients:
                raise ValueError(f"LLM client not found: {client_id}")
            return llm_clients[client_id]
        
        # Return first client if ID not provided
        return next(iter(llm_clients.values()))
    
    def get_translation_client(self, client_id: Optional[str] = None) -> TranslationClient:
        """
        Get a registered translation client.
        
        Args:
            client_id: Optional client ID (returns first client if not provided)
            
        Returns:
            Translation client
        """
        translation_clients = self._clients[ServiceType.TRANSLATION]
        
        if not translation_clients:
            raise ValueError("No translation clients registered")
        
        if client_id:
            if client_id not in translation_clients:
                raise ValueError(f"Translation client not found: {client_id}")
            return translation_clients[client_id]
        
        # Return first client if ID not provided
        return next(iter(translation_clients.values()))
    
    def get_custom_endpoint(self, name: str) -> Any:
        """Get a registered custom endpoint."""
        if name not in self._custom_endpoints:
            raise ValueError(f"No custom endpoint registered with name: {name}")
        return self._custom_endpoints[name]
    
    def has_service(self, service_type: ServiceType, client_id: Optional[str] = None) -> bool:
        """
        Check if a service type is registered.
        
        Args:
            service_type: Service type to check
            client_id: Optional client ID to check for
            
        Returns:
            True if service is registered, False otherwise
        """
        if service_type not in self._clients:
            return False
        
        if client_id:
            return client_id in self._clients[service_type]
        
        return bool(self._clients[service_type])
    
    def has_custom_endpoint(self, name: str) -> bool:
        """Check if a custom endpoint is registered."""
        return name in self._custom_endpoints
    
    def list_services(self) -> Dict[ServiceType, List[str]]:
        """
        List all registered services.
        
        Returns:
            Dictionary mapping service types to lists of client IDs
        """
        return {
            service_type: list(clients.keys())
            for service_type, clients in self._clients.items()
            if clients
        }
    
    def list_custom_endpoints(self) -> List[str]:
        """List all registered custom endpoints."""
        return list(self._custom_endpoints.keys())
    
    def list_available_providers(self, service_type: Optional[ServiceType] = None) -> Dict[str, Any]:
        """
        List all available providers.
        
        Args:
            service_type: Optional service type to filter providers
            
        Returns:
            Dictionary of provider information
        """
        if service_type:
            return provider_registry.get_providers_for_service(service_type.value)
        return provider_registry.get_all_providers()
    
    def get_provider_features(self, provider: str, service_type: ServiceType) -> List[str]:
        """
        Get features supported by a provider for a specific service type.
        
        Args:
            provider: Provider ID
            service_type: Service type
            
        Returns:
            List of supported features
        """
        return provider_registry.get_features(provider, service_type.value)
    
    def get_provider_models(self, provider: str, service_type: ServiceType) -> List[Dict[str, Any]]:
        """
        Get models supported by a provider for a specific service type.
        
        Args:
            provider: Provider ID
            service_type: Service type
            
        Returns:
            List of supported models
        """
        return provider_registry.get_models(provider, service_type.value)
