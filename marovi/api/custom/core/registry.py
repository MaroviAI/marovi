"""
Registry for custom endpoints.

This module provides a registry system for custom endpoints,
allowing them to be discovered and instantiated at runtime.
"""

from typing import Dict, Optional, Any, List, Type, Callable, Set, Union
import logging

from .base import CustomEndpoint
from .errors import CustomEndpointError

logger = logging.getLogger(__name__)

class CustomEndpointRegistry:
    """
    Registry for custom endpoints.
    
    This registry stores information about custom endpoints and their capabilities.
    """
    
    def __init__(self):
        """Initialize the registry."""
        self._endpoints: Dict[str, Any] = {}
        self._factories: Dict[str, Callable] = {}
        self._capabilities: Dict[str, Set[str]] = {}
        self.logger = logging.getLogger(__name__)
    
    def register_endpoint(self, name: str, endpoint_class: Union[Type[CustomEndpoint], CustomEndpoint], **kwargs) -> bool:
        """
        Register a custom endpoint with the registry.
        
        Args:
            name: Name of the endpoint
            endpoint_class: Endpoint class or instance
            **kwargs: Optional parameters to pass when instantiating the endpoint
            
        Returns:
            True if registration was successful, False otherwise
            
        Raises:
            ValueError: If the endpoint is not valid or already registered
        """
        # Check if the endpoint is already registered
        if name in self._endpoints:
            raise ValueError(f"Endpoint '{name}' is already registered")
            
        # If it's already an instance, register it directly
        if isinstance(endpoint_class, CustomEndpoint):
            endpoint_instance = endpoint_class
            self._endpoints[name] = endpoint_instance
            self.logger.debug(f"Registered endpoint '{name}' (instance)")
        else:
            # Otherwise, instantiate the endpoint class and register the instance
            try:
                endpoint_instance = endpoint_class(**kwargs)
                self._endpoints[name] = endpoint_instance
                self.logger.debug(f"Registered endpoint '{name}' (class instantiated)")
            except Exception as e:
                self.logger.error(f"Failed to register endpoint '{name}': {str(e)}")
                raise ValueError(f"Failed to register endpoint '{name}': {str(e)}")
        
        # Register capabilities if the endpoint has a get_capabilities method
        try:
            capabilities = endpoint_instance.get_capabilities()
            for capability in capabilities:
                if capability not in self._capabilities:
                    self._capabilities[capability] = set()
                self._capabilities[capability].add(name)
            self.logger.debug(f"Registered capabilities for '{name}': {capabilities}")
        except Exception as e:
            self.logger.warning(f"Could not register capabilities for '{name}': {str(e)}")
            
        return True
    
    def register_factory(self, name: str, factory: Callable) -> None:
        """
        Register a factory function for creating endpoints.
        
        Args:
            name: Name of the factory
            factory: Factory function
        """
        if name in self._factories:
            logger.warning(f"Overwriting existing factory: {name}")
        
        self._factories[name] = factory
    
    def get_endpoint_class(self, name: str) -> Optional[Type[CustomEndpoint]]:
        """
        Get a custom endpoint class or instance by name.
        
        Args:
            name: Name of the endpoint
            
        Returns:
            The endpoint class or instance if found, None otherwise
        """
        return self._endpoints.get(name)
    
    def get_factory(self, name: str) -> Optional[Callable]:
        """
        Get a factory function by name.
        
        Args:
            name: Name of the factory
            
        Returns:
            The factory function if found, None otherwise
        """
        return self._factories.get(name)
    
    def get_endpoints_by_capability(self, capability: str) -> List[str]:
        """
        Get all endpoints that provide a specific capability.
        
        Args:
            capability: The capability to search for
            
        Returns:
            List of endpoint names with the capability
        """
        return list(self._capabilities.get(capability, set()))
    
    def list_endpoints(self) -> List[str]:
        """
        List all registered endpoints.
        
        Returns:
            List of endpoint names
        """
        return list(self._endpoints.keys())
    
    def list_factories(self) -> List[str]:
        """
        List all registered factories.
        
        Returns:
            List of factory names
        """
        return list(self._factories.keys())
    
    def list_capabilities(self) -> List[str]:
        """
        List all registered capabilities.
        
        Returns:
            List of capability names
        """
        return list(self._capabilities.keys())

# Create the default registry
default_registry = CustomEndpointRegistry()

def register_endpoint(name: str, endpoint_class: Union[Type[CustomEndpoint], CustomEndpoint], **kwargs) -> bool:
    """
    Register a custom endpoint with the default registry.
    
    This is a convenience function that delegates to the default registry's register_endpoint method.
    
    Args:
        name: Name of the endpoint
        endpoint_class: Endpoint class or instance
        **kwargs: Optional parameters to pass when instantiating the endpoint
        
    Returns:
        True if registration was successful, False otherwise
        
    Raises:
        ValueError: If the endpoint is not valid or already registered
    """
    return default_registry.register_endpoint(name, endpoint_class, **kwargs)

def register_default_endpoints(registry=None):
    """
    Register default endpoints with the registry.
    
    Args:
        registry: Registry to register with (defaults to default_registry)
    """
    if registry is None:
        registry = default_registry
    
    logger.debug("Registering default endpoints")
    
    # Register LLMTranslate endpoint
    try:
        from ..endpoints.llm_translate import LLMTranslate
        llm_translate = LLMTranslate()
        registry.register_endpoint("llm_translate", llm_translate)
        logger.debug("Registered LLMTranslate endpoint")
    except (ImportError, Exception) as e:
        logger.warning(f"Failed to register LLMTranslate endpoint: {str(e)}")
    
    # Register FormatConverter endpoint
    try:
        from ..endpoints.convert_format import FormatConverter
        format_converter = FormatConverter()
        registry.register_endpoint("convert_format", format_converter)
        logger.debug("Registered FormatConverter endpoint")
    except (ImportError, Exception) as e:
        logger.warning(f"Failed to register FormatConverter endpoint: {str(e)}")
    
    # Register Summarizer endpoint
    try:
        from ..endpoints.summarize import Summarizer
        summarizer = Summarizer()
        registry.register_endpoint("summarize", summarizer)
        logger.debug("Registered Summarizer endpoint")
    except (ImportError, Exception) as e:
        logger.warning(f"Failed to register Summarizer endpoint: {str(e)}")
