"""
Service container module for dependency injection.

This module provides a service container that manages all application dependencies
in the application to promote loose coupling and better testability.
"""

from typing import Dict, Optional, Any, Type, TypeVar, cast
import os
import logging

from ..config import APISettings
from ..utils.logging import setup_logging, get_logger

T = TypeVar('T')

class ServiceContainer:
    """
    Service container for managing application dependencies.
    
    This class provides a centralized container for registering and retrieving
    service instances, allowing for better dependency management and testability.
    """
    
    def __init__(self, settings: Optional[APISettings] = None):
        """
        Initialize the service container.
        
        Args:
            settings: Optional APISettings instance (will be created if not provided)
        """
        self._services: Dict[str, Any] = {}
        self._factories: Dict[str, Any] = {}
        self._initialized = False
        
        # Register core services
        self._register('settings', settings or self._create_settings())
        self._register('logger', get_logger(__name__))
        
    def _create_settings(self) -> APISettings:
        """Create settings instance with environment variables."""
        env_file = os.environ.get('MAROVI_ENV_FILE', '.env')
        return APISettings(_env_file=env_file)
    
    def initialize(self) -> None:
        """Initialize the service container and all registered services."""
        if self._initialized:
            return
            
        # Set up logging based on settings
        setup_logging(self.get('settings').LOG_LEVEL)
        self._initialized = True
        
        logger = self.get('logger')
        logger.info("Service container initialized")
    
    def _register(self, name: str, instance: Any) -> None:
        """
        Register a service instance with the container.
        
        Args:
            name: Service name
            instance: Service instance
        """
        self._services[name] = instance
    
    def register(self, name: str, instance: Any) -> None:
        """
        Register a service instance with the container.
        
        Args:
            name: Service name
            instance: Service instance
        """
        self._register(name, instance)
        
        logger = self.get('logger')
        logger.debug(f"Registered service: {name}")
    
    def register_factory(self, name: str, factory: Any) -> None:
        """
        Register a factory function for lazy service instantiation.
        
        Args:
            name: Service name
            factory: Factory function that creates the service
        """
        self._factories[name] = factory
        
        logger = self.get('logger')
        logger.debug(f"Registered service factory: {name}")
    
    def get(self, name: str, default: Optional[Any] = None) -> Any:
        """
        Get a service instance by name.
        
        Args:
            name: Service name
            default: Default value if service not found
            
        Returns:
            Service instance
        """
        # Return existing instance if already created
        if name in self._services:
            return self._services[name]
        
        # Create instance from factory if available
        if name in self._factories:
            instance = self._factories[name](self)
            self._services[name] = instance
            return instance
        
        return default
    
    def get_typed(self, service_type: Type[T], name: Optional[str] = None) -> T:
        """
        Get a service instance by type with optional name.
        
        Args:
            service_type: Type of service to retrieve
            name: Optional service name (if multiple services of same type)
            
        Returns:
            Service instance of the specified type
            
        Raises:
            KeyError: If service not found
        """
        if name:
            instance = self.get(name)
            if not instance:
                raise KeyError(f"Service not found: {name}")
            return cast(service_type, instance)
        
        # Search for service by type
        for instance in self._services.values():
            if isinstance(instance, service_type):
                return cast(service_type, instance)
        
        # Search for factory by type
        for factory_name, factory in self._factories.items():
            # Create instance to check type
            instance = factory(self)
            if isinstance(instance, service_type):
                self._services[factory_name] = instance
                return cast(service_type, instance)
        
        raise KeyError(f"No service of type {service_type.__name__} found")
    
    def has(self, name: str) -> bool:
        """
        Check if a service is registered.
        
        Args:
            name: Service name
            
        Returns:
            True if service is registered, False otherwise
        """
        return name in self._services or name in self._factories
    
    def remove(self, name: str) -> None:
        """
        Remove a service by name.
        
        Args:
            name: Service name
        """
        if name in self._services:
            del self._services[name]
        
        if name in self._factories:
            del self._factories[name]
    
    def clear(self) -> None:
        """Clear all registered services."""
        self._services.clear()
        self._factories.clear()
        self._initialized = False
        
        # Re-register core services
        self._register('settings', self._create_settings())
        self._register('logger', get_logger(__name__))

# Create global container instance
container = ServiceContainer() 