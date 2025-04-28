"""
Provider registry module.

This module handles the registration and discovery of available providers.
"""

import os
import yaml
import importlib
from typing import Dict, List, Any, Optional, Type, Union

from .config import settings
from .utils.logging import get_logger

logger = get_logger(__name__)

class ProviderRegistry:
    """
    Registry for managing API providers.
    
    This class handles loading provider configurations from the registry file
    and providing metadata about available providers and their capabilities.
    """
    
    def __init__(self, registry_path: Optional[str] = None):
        """Initialize the provider registry."""
        self.registry_path = registry_path or os.path.join(
            os.path.dirname(__file__), 
            "providers", 
            "registry.yaml"
        )
        self.providers: Dict[str, Any] = {}
        self.load_registry()
        
    def load_registry(self) -> None:
        """Load provider registry from the YAML file."""
        try:
            with open(self.registry_path, 'r') as f:
                registry_data = yaml.safe_load(f)
                self.providers = registry_data.get('providers', {})
            logger.info(f"Loaded provider registry from {self.registry_path}")
        except Exception as e:
            logger.error(f"Failed to load provider registry: {str(e)}")
            # Initialize with empty registry if loading fails
            self.providers = {}
    
    def get_provider_info(self, provider_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific provider."""
        return self.providers.get(provider_id)
    
    def get_all_providers(self) -> Dict[str, Any]:
        """Get all registered providers."""
        return self.providers
    
    def get_providers_for_service(self, service_type: str) -> Dict[str, Any]:
        """Get all providers that support a specific service type."""
        result = {}
        for provider_id, provider_info in self.providers.items():
            for service in provider_info.get('services', []):
                if service.get('type') == service_type:
                    result[provider_id] = provider_info
                    break
        return result
    
    def get_provider_class(self, provider_id: str, service_type: str) -> Optional[Type]:
        """Get the provider implementation class for a specific service type."""
        provider_info = self.get_provider_info(provider_id)
        if not provider_info:
            return None
        
        for service in provider_info.get('services', []):
            if service.get('type') == service_type:
                implementation_path = service.get('implementation')
                if implementation_path:
                    try:
                        module_path, class_name = implementation_path.rsplit('.', 1)
                        module = importlib.import_module(module_path)
                        return getattr(module, class_name)
                    except (ImportError, AttributeError) as e:
                        logger.error(f"Failed to import provider implementation {implementation_path}: {str(e)}")
                        return None
        return None
    
    def get_default_model(self, provider_id: str, service_type: str) -> Optional[str]:
        """Get the default model for a provider and service type."""
        provider_info = self.get_provider_info(provider_id)
        if not provider_info:
            return None
        
        for service in provider_info.get('services', []):
            if service.get('type') == service_type:
                return service.get('default_model')
        
        return None
    
    def get_models(self, provider_id: str, service_type: str) -> List[Dict[str, Any]]:
        """Get all models supported by a provider for a specific service type."""
        provider_info = self.get_provider_info(provider_id)
        if not provider_info:
            return []
        
        for service in provider_info.get('services', []):
            if service.get('type') == service_type:
                return service.get('models', [])
        
        return []
    
    def get_features(self, provider_id: str, service_type: str) -> List[str]:
        """Get all features supported by a provider for a specific service type."""
        provider_info = self.get_provider_info(provider_id)
        if not provider_info:
            return []
        
        for service in provider_info.get('services', []):
            if service.get('type') == service_type:
                return service.get('features', [])
        
        return []
    
    def get_env_vars(self, provider_id: str) -> List[str]:
        """Get all environment variables required by a provider."""
        provider_info = self.get_provider_info(provider_id)
        if not provider_info:
            return []
        
        return provider_info.get('env_vars', [])
    
    def get_docs_url(self, provider_id: str) -> Optional[str]:
        """Get the documentation URL for a provider."""
        provider_info = self.get_provider_info(provider_id)
        if not provider_info:
            return None
        
        return provider_info.get('docs_url')

# Create a global instance of the registry
provider_registry = ProviderRegistry() 