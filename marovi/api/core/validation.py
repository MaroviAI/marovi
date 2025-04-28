"""
Configuration validation utilities.

This module provides validation utilities to ensure that
configuration data references valid implementations.
"""

import importlib
import inspect
from typing import Dict, Any, List, Type, Set, Optional, Callable, Union, Tuple
import yaml

from .exceptions import ConfigurationError


def validate_class_exists(module_path: str, class_name: str) -> Type:
    """
    Validate that a class exists in a module.
    
    Args:
        module_path: Dot notation path to module
        class_name: Name of the class
        
    Returns:
        The class object if it exists
        
    Raises:
        ConfigurationError: If the module or class doesn't exist
    """
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ConfigurationError(
            f"Module '{module_path}' not found: {str(e)}"
        ) from e
    
    if not hasattr(module, class_name):
        raise ConfigurationError(
            f"Class '{class_name}' not found in module '{module_path}'"
        )
    
    cls = getattr(module, class_name)
    if not inspect.isclass(cls):
        raise ConfigurationError(
            f"'{class_name}' in module '{module_path}' is not a class"
        )
    
    return cls


def validate_function_exists(module_path: str, function_name: str) -> Callable:
    """
    Validate that a function exists in a module.
    
    Args:
        module_path: Dot notation path to module
        function_name: Name of the function
        
    Returns:
        The function object if it exists
        
    Raises:
        ConfigurationError: If the module or function doesn't exist
    """
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ConfigurationError(
            f"Module '{module_path}' not found: {str(e)}"
        ) from e
    
    if not hasattr(module, function_name):
        raise ConfigurationError(
            f"Function '{function_name}' not found in module '{module_path}'"
        )
    
    func = getattr(module, function_name)
    if not callable(func):
        raise ConfigurationError(
            f"'{function_name}' in module '{module_path}' is not callable"
        )
    
    return func


def split_implementation_path(implementation: str) -> Tuple[str, str]:
    """
    Split an implementation path into module path and object name.
    
    Args:
        implementation: Implementation path (e.g., 'marovi.api.providers.google.GoogleProvider')
        
    Returns:
        Tuple of (module_path, object_name)
        
    Raises:
        ConfigurationError: If the implementation path is invalid
    """
    if not implementation or not isinstance(implementation, str):
        raise ConfigurationError(f"Invalid implementation path: {implementation}")
    
    parts = implementation.split('.')
    if len(parts) < 2:
        raise ConfigurationError(
            f"Implementation path must be in format 'module.path.ObjectName': {implementation}"
        )
    
    object_name = parts[-1]
    module_path = '.'.join(parts[:-1])
    
    return module_path, object_name


def validate_implementation(implementation: str, base_class: Optional[Type] = None) -> Type:
    """
    Validate that an implementation exists and optionally inherits from a base class.
    
    Args:
        implementation: Implementation path (e.g., 'marovi.api.providers.google.GoogleProvider')
        base_class: Optional base class that the implementation must inherit from
        
    Returns:
        The implementation class if it exists and passes validation
        
    Raises:
        ConfigurationError: If the implementation doesn't exist or doesn't inherit from base_class
    """
    module_path, class_name = split_implementation_path(implementation)
    
    cls = validate_class_exists(module_path, class_name)
    
    if base_class and not issubclass(cls, base_class):
        raise ConfigurationError(
            f"Class '{implementation}' must inherit from '{base_class.__name__}'"
        )
    
    return cls


def validate_provider_registry(registry_data: Dict[str, Any], base_provider_class: Type) -> List[str]:
    """
    Validate a provider registry configuration.
    
    Args:
        registry_data: Provider registry data
        base_provider_class: Base provider class that all implementations must inherit from
        
    Returns:
        List of validation warnings (non-critical issues)
        
    Raises:
        ConfigurationError: If any validation errors are found
    """
    warnings = []
    
    if not isinstance(registry_data, dict):
        raise ConfigurationError("Provider registry must be a dictionary")
    
    # Validate providers section
    if 'providers' not in registry_data:
        raise ConfigurationError("Provider registry must contain a 'providers' section")
    
    providers = registry_data['providers']
    if not isinstance(providers, dict):
        raise ConfigurationError("Providers section must be a dictionary")
    
    # Validate each provider
    for provider_id, provider_data in providers.items():
        if not isinstance(provider_data, dict):
            raise ConfigurationError(f"Provider '{provider_id}' data must be a dictionary")
        
        # Check required fields
        for field in ['name', 'implementation']:
            if field not in provider_data:
                raise ConfigurationError(f"Provider '{provider_id}' must contain a '{field}' field")
        
        # Validate implementation
        implementation = provider_data['implementation']
        try:
            validate_implementation(implementation, base_provider_class)
        except ConfigurationError as e:
            raise ConfigurationError(f"Provider '{provider_id}': {str(e)}") from e
        
        # Check services
        if 'services' not in provider_data:
            warnings.append(f"Provider '{provider_id}' does not specify any services")
            continue
        
        services = provider_data['services']
        if not isinstance(services, dict):
            raise ConfigurationError(f"Provider '{provider_id}': services must be a dictionary")
        
        # Validate each service
        for service_id, service_data in services.items():
            if not isinstance(service_data, dict):
                raise ConfigurationError(
                    f"Provider '{provider_id}', service '{service_id}': data must be a dictionary"
                )
    
    return warnings


def validate_yaml_file(file_path: str) -> Dict[str, Any]:
    """
    Validate that a file is a valid YAML file.
    
    Args:
        file_path: Path to the YAML file
        
    Returns:
        The parsed YAML data
        
    Raises:
        ConfigurationError: If the file doesn't exist or is not valid YAML
    """
    try:
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
            if data is None:
                return {}
            return data
    except FileNotFoundError as e:
        raise ConfigurationError(f"File not found: {file_path}") from e
    except yaml.YAMLError as e:
        raise ConfigurationError(f"Invalid YAML file '{file_path}': {str(e)}") from e
    except Exception as e:
        raise ConfigurationError(f"Error reading file '{file_path}': {str(e)}") from e 