"""
Custom client for interacting with custom endpoints.

This module provides a unified interface for making requests to different custom endpoints
which represent complex workflows that may combine multiple services rather than simply
wrapping external API calls.
"""

import time
import logging
import importlib
import asyncio
from typing import Dict, Any, Optional, List, Type, Union, TypeVar, Generic, Callable

from pydantic import BaseModel

# Use lazy imports to avoid circular imports
# from ..custom.core.registry import CustomEndpointRegistry
# from ..custom.core.base import CustomEndpoint

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Type variables for request and response
RequestType = TypeVar('RequestType', bound=BaseModel)
ResponseType = TypeVar('ResponseType', bound=BaseModel)

# Default retry configuration
DEFAULT_RETRY_CONFIG = {
    "max_retries": 3,
    "initial_backoff": 1.0,
    "max_backoff": 10.0,
    "backoff_factor": 2.0,
    "retryable_errors": [
        "timeout",
        "connection_error", 
        "server_error"
    ]
}

class CustomEndpointError(Exception):
    """Exception raised for custom endpoint errors."""
    pass

# Global client cache and registry reference
_client_cache: Dict[str, 'CustomClient'] = {}
_default_registry = None

def get_registry():
    """Get the default endpoint registry with lazy loading."""
    global _default_registry
    if _default_registry is None:
        # Lazy import to avoid circular imports
        from ..custom import default_registry
        _default_registry = default_registry
    return _default_registry

class CustomClient(Generic[RequestType, ResponseType]):
    """
    A client for interacting with custom endpoints.
    
    This client provides methods for:
    - Instantiating custom endpoints by name
    - Calling endpoints with automatic validation
    - Error handling and retries
    - Metadata access
    """
    
    def __init__(self, 
                endpoint_name: str,
                request_model: Type[RequestType] = None,
                response_model: Type[ResponseType] = None,
                registry = None,  # Type: Optional[CustomEndpointRegistry]
                retry_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the custom client.
        
        Args:
            endpoint_name: Name of the custom endpoint to use
            request_model: Optional request model class (will try to autodiscover if not provided)
            response_model: Optional response model class (will try to autodiscover if not provided)
            registry: Optional endpoint registry (will create one if not provided)
            retry_config: Configuration for retry logic
        """
        self.endpoint_name = endpoint_name
        self.registry = registry or self._get_default_registry()
        
        # Load the endpoint
        self.endpoint = self._load_endpoint(endpoint_name)
        
        # Set the request and response models
        self.request_model = request_model or getattr(self.endpoint, 'request_model', None)
        self.response_model = response_model or getattr(self.endpoint, 'response_model', None)
        
        # Validate models
        if not self.request_model or not self.response_model:
            raise ValueError(f"Could not determine request and response models for endpoint: {endpoint_name}")
            
        # Set retry configuration
        self.retry_config = retry_config or DEFAULT_RETRY_CONFIG
        
        # Initialize the endpoint with services if needed
        self._initialize_endpoint_with_services()
        
        logger.info(f"Initialized CustomClient for endpoint={endpoint_name}")
    
    def _get_default_registry(self):
        """Get or create the default endpoint registry."""
        return get_registry()
    
    def _load_endpoint(self, endpoint_name: str):
        """Load a custom endpoint by name."""
        # Lazy import to avoid circular imports
        from ..custom.core.base import CustomEndpoint
        
        # Try to get the endpoint class from the registry
        endpoint_class_or_instance = self.registry.get_endpoint_class(endpoint_name)
        
        # If not found in registry, try to import it directly
        if not endpoint_class_or_instance:
            try:
                # Convention: endpoints are in marovi.api.custom.endpoints.<n>
                module_name = f"marovi.api.custom.endpoints.{endpoint_name}"
                module = importlib.import_module(module_name)
                
                # Convention: endpoint class has same name but in CamelCase
                class_name = ''.join(word.capitalize() for word in endpoint_name.split('_'))
                endpoint_class_or_instance = getattr(module, class_name)
                
            except (ImportError, AttributeError) as e:
                raise ValueError(f"Could not load endpoint {endpoint_name}: {str(e)}")
        
        # If it's already an instance, use it directly
        if isinstance(endpoint_class_or_instance, CustomEndpoint):
            logger.debug(f"Using pre-instantiated endpoint {endpoint_name}")
            return endpoint_class_or_instance
            
        # Otherwise, instantiate the endpoint
        try:
            if callable(endpoint_class_or_instance) and not isinstance(endpoint_class_or_instance, type):
                # It's a factory function
                return endpoint_class_or_instance()
            else:
                # It's a class, instantiate it
                return endpoint_class_or_instance()
        except Exception as e:
            raise ValueError(f"Could not instantiate endpoint {endpoint_name}: {str(e)}")
    
    def _initialize_endpoint_with_services(self):
        """Initialize the endpoint with services from the router if needed."""
        # Check if the endpoint has methods that might need service clients
        endpoint_needs_llm = hasattr(self.endpoint, '_get_llm_client')
        endpoint_needs_translation = hasattr(self.endpoint, '_get_translation_client')
        
        if endpoint_needs_llm or endpoint_needs_translation:
            try:
                # Lazy import to avoid circular imports
                from ..core.router import default_router
                from ..core.base import ServiceType
                
                # Set services on the endpoint if they're needed and not already set
                if endpoint_needs_llm and not getattr(self.endpoint, 'llm_client', None):
                    try:
                        self.endpoint.llm_client = default_router.get_service(ServiceType.LLM)
                        logger.debug(f"Initialized endpoint {self.endpoint_name} with LLM client from router")
                    except Exception as e:
                        logger.warning(f"Could not initialize endpoint {self.endpoint_name} with LLM client: {str(e)}")
                
                if endpoint_needs_translation and not getattr(self.endpoint, 'translation_client', None):
                    try:
                        self.endpoint.translation_client = default_router.get_service(ServiceType.TRANSLATION)
                        logger.debug(f"Initialized endpoint {self.endpoint_name} with translation client from router")
                    except Exception as e:
                        logger.warning(f"Could not initialize endpoint {self.endpoint_name} with translation client: {str(e)}")
                        
            except ImportError as e:
                logger.warning(f"Could not import router for endpoint service initialization: {str(e)}")
    
    def _should_retry(self, error: Exception, attempt: int) -> bool:
        """
        Determine if a request should be retried based on the error and attempt number.
        
        Args:
            error: The exception that occurred
            attempt: The current attempt number (starting from 1)
            
        Returns:
            True if the request should be retried, False otherwise
        """
        if attempt >= self.retry_config["max_retries"]:
            return False
        
        error_type = type(error).__name__.lower()
        error_msg = str(error).lower()
        
        # Check if error matches any retryable error patterns
        for retryable_error in self.retry_config["retryable_errors"]:
            if retryable_error in error_type or retryable_error in error_msg:
                return True
        
        return False
    
    def _sync_backoff(self, attempt: int) -> None:
        """Implement synchronous exponential backoff for retries."""
        backoff_time = min(
            self.retry_config["initial_backoff"] * (self.retry_config["backoff_factor"] ** attempt),
            self.retry_config["max_backoff"]
        )
        time.sleep(backoff_time)
    
    async def _backoff(self, attempt: int) -> None:
        """Implement asynchronous exponential backoff for retries."""
        backoff_time = min(
            self.retry_config["initial_backoff"] * (self.retry_config["backoff_factor"] ** attempt),
            self.retry_config["max_backoff"]
        )
        await asyncio.sleep(backoff_time)
    
    def process(self, request_data: Union[Dict[str, Any], RequestType], **kwargs) -> ResponseType:
        """
        Process a request and return a response.
        
        Args:
            request_data: Request data (either a dict or an instance of the request model)
            **kwargs: Additional arguments to pass to the endpoint
            
        Returns:
            Response model instance
        """
        # Ensure we have a request model instance
        if isinstance(request_data, dict):
            request = self.request_model(**request_data)
        elif isinstance(request_data, BaseModel):
            if not isinstance(request_data, self.request_model):
                # Convert from one model to another if they're different
                request = self.request_model(**request_data.dict())
            else:
                request = request_data
        else:
            raise TypeError(f"Expected dict or {self.request_model.__name__}, got {type(request_data).__name__}")
            
        step_name = kwargs.get('step_name')
        start_time = time.time()
        
        # Track attempts for retry logic
        attempt = 0
        
        while True:
            try:
                # Process the request
                response = self.endpoint.process(request, **kwargs)
                
                # Convert to response model if needed
                if not isinstance(response, self.response_model):
                    if isinstance(response, BaseModel):
                        response = self.response_model(**response.dict())
                    elif isinstance(response, dict):
                        response = self.response_model(**response)
                    else:
                        raise TypeError(f"Endpoint returned {type(response).__name__}, expected {self.response_model.__name__}")
                
                # Log success if step_name provided
                if step_name:
                    latency = time.time() - start_time
                    logger.info(f"Custom endpoint {self.endpoint_name} completed successfully: "
                               f"latency={latency:.2f}s, attempts={attempt+1}")
                
                return response
                
            except Exception as e:
                attempt += 1
                
                # Check if we should retry
                if self._should_retry(e, attempt):
                    logger.warning(f"Retrying custom endpoint {self.endpoint_name} (attempt {attempt}/{self.retry_config['max_retries']}): {str(e)}")
                    self._sync_backoff(attempt - 1)
                    continue
                
                # If we shouldn't retry, log the error and raise
                if step_name:
                    logger.error(f"Custom endpoint {self.endpoint_name} failed after {attempt} attempts: {str(e)}")
                
                raise CustomEndpointError(f"Error in endpoint {self.endpoint_name}: {str(e)}") from e
    
    async def aprocess(self, request_data: Union[Dict[str, Any], RequestType], **kwargs) -> ResponseType:
        """
        Process a request asynchronously and return a response.
        
        Args:
            request_data: Request data (either a dict or an instance of the request model)
            **kwargs: Additional arguments to pass to the endpoint
            
        Returns:
            Response model instance
        """
        # Ensure we have a request model instance
        if isinstance(request_data, dict):
            request = self.request_model(**request_data)
        elif isinstance(request_data, BaseModel):
            if not isinstance(request_data, self.request_model):
                # Convert from one model to another if they're different
                request = self.request_model(**request_data.dict())
            else:
                request = request_data
        else:
            raise TypeError(f"Expected dict or {self.request_model.__name__}, got {type(request_data).__name__}")
            
        step_name = kwargs.get('step_name')
        start_time = time.time()
        
        # Track attempts for retry logic
        attempt = 0
        
        while True:
            try:
                # Process the request asynchronously if the method exists
                if hasattr(self.endpoint, 'aprocess'):
                    response = await self.endpoint.aprocess(request, **kwargs)
                else:
                    # Fall back to sync processing in a separate thread
                    response = await asyncio.to_thread(self.endpoint.process, request, **kwargs)
                
                # Convert to response model if needed
                if not isinstance(response, self.response_model):
                    if isinstance(response, BaseModel):
                        response = self.response_model(**response.dict())
                    elif isinstance(response, dict):
                        response = self.response_model(**response)
                    else:
                        raise TypeError(f"Endpoint returned {type(response).__name__}, expected {self.response_model.__name__}")
                
                # Log success if step_name provided
                if step_name:
                    latency = time.time() - start_time
                    logger.info(f"Async custom endpoint {self.endpoint_name} completed successfully: "
                               f"latency={latency:.2f}s, attempts={attempt+1}")
                
                return response
                
            except Exception as e:
                attempt += 1
                
                # Check if we should retry
                if self._should_retry(e, attempt):
                    logger.warning(f"Retrying async custom endpoint {self.endpoint_name} (attempt {attempt}/{self.retry_config['max_retries']}): {str(e)}")
                    await self._backoff(attempt - 1)
                    continue
                
                # If we shouldn't retry, log the error and raise
                if step_name:
                    logger.error(f"Async custom endpoint {self.endpoint_name} failed after {attempt} attempts: {str(e)}")
                
                raise CustomEndpointError(f"Error in endpoint {self.endpoint_name}: {str(e)}") from e
    
    def get_capabilities(self) -> List[str]:
        """Get the capabilities supported by this endpoint."""
        return self.endpoint.get_capabilities()
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about this endpoint."""
        return self.endpoint.get_metadata()


class CallableEndpoint:
    """
    Wrapper for callable endpoints that properly forwards arguments.
    
    This class wraps an endpoint that has a __call__ method to ensure that
    arguments are correctly forwarded when the endpoint is called.
    """
    
    def __init__(self, endpoint):
        """
        Initialize with the endpoint to wrap.
        
        Args:
            endpoint: The endpoint to wrap
        """
        self.endpoint = endpoint
        
        # Copy attributes from the endpoint for better discoverability
        if hasattr(endpoint, '__doc__'):
            self.__doc__ = endpoint.__doc__
        if hasattr(endpoint, '__name__'):
            self.__name__ = endpoint.__name__
        
    def __call__(self, *args, **kwargs):
        """
        Forward the call to the endpoint's __call__ method.
        
        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Result from the endpoint's __call__ method
        """
        return self.endpoint(*args, **kwargs)
    
    def __getattr__(self, name):
        """
        Forward attribute access to the endpoint.
        
        Args:
            name: Name of the attribute
            
        Returns:
            Attribute from the endpoint
            
        Raises:
            AttributeError: If the attribute does not exist
        """
        return getattr(self.endpoint, name)


class CustomClientProxy:
    """
    A proxy for accessing custom endpoints as attributes.
    
    This class provides dynamic attribute access to custom endpoints,
    allowing them to be accessed using dot notation.
    """
    
    def __init__(self, registry = None, retry_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the custom client proxy.
        
        Args:
            registry: Optional endpoint registry
            retry_config: Configuration for retry logic
        """
        self.registry = registry or self._get_default_registry()
        self.retry_config = retry_config or DEFAULT_RETRY_CONFIG
        self._clients = {}
        self._callable_endpoints = {}
    
    def _get_default_registry(self):
        """Get or create the default endpoint registry."""
        return get_registry()
    
    def __getattr__(self, name: str) -> Any:
        """
        Get a custom endpoint by name.
        
        This method allows endpoints to be accessed using dot notation:
        
        ```python
        client = CustomClientProxy()
        translator = client.translator  # Returns a client for the 'translator' endpoint
        ```
        
        Args:
            name: Name of the endpoint
            
        Returns:
            Client for the endpoint or a wrapper function if the endpoint is callable
            
        Raises:
            AttributeError: If the endpoint does not exist
        """
        if name.startswith('_'):
            raise AttributeError(f"No attribute named '{name}'")
        
        # Check if we already have a callable endpoint for this name
        if name in self._callable_endpoints:
            return self._callable_endpoints[name]
        
        # Check if we already have a client for this endpoint
        if name in self._clients:
            # If the endpoint is directly callable, create a wrapper
            endpoint = self._clients[name].endpoint
            if hasattr(endpoint, '__call__') and callable(endpoint.__call__):
                wrapper = CallableEndpoint(endpoint)
                self._callable_endpoints[name] = wrapper
                return wrapper
            return self._clients[name]
        
        # Check if the endpoint exists in the registry
        registry_endpoints = self.registry.list_endpoints()
        
        if name not in registry_endpoints:
            # Try to find a close match
            close_matches = []
            for endpoint in registry_endpoints:
                if endpoint.startswith(name) or name in endpoint:
                    close_matches.append(endpoint)
            
            if close_matches:
                match_str = ", ".join(close_matches)
                raise AttributeError(f"No endpoint named '{name}'. Did you mean one of: {match_str}?")
            else:
                raise AttributeError(f"No endpoint named '{name}'. Available endpoints: {', '.join(registry_endpoints)}")
        
        # Create a client for this endpoint
        client = create_custom_client(name, registry=self.registry, retry_config=self.retry_config)
        self._clients[name] = client
        
        # If the endpoint is directly callable, create a wrapper
        endpoint = client.endpoint
        if hasattr(endpoint, '__call__') and callable(endpoint.__call__):
            wrapper = CallableEndpoint(endpoint)
            self._callable_endpoints[name] = wrapper
            return wrapper
        
        return client
    
    def __dir__(self) -> List[str]:
        """
        Get a list of available endpoints.
        
        This method is used by autocomplete in interactive environments.
        
        Returns:
            List of endpoint names
        """
        # Start with the standard attributes
        attrs = set(super().__dir__())
        
        # Add all the endpoints from the registry
        try:
            # The registry might not be fully initialized yet
            endpoints = self.registry.list_endpoints()
            attrs.update(endpoints)
        except Exception as e:
            logger.debug(f"Could not list endpoints: {str(e)}")
        
        return sorted(attrs)
    
    def get_client(self, endpoint_name: str) -> 'CustomClient':
        """
        Get a client for a custom endpoint.
        
        This is an alternative to using dot notation.
        
        Args:
            endpoint_name: Name of the endpoint
            
        Returns:
            Client for the endpoint
        """
        return self.__getattr__(endpoint_name)


def get_client(endpoint_name: str, 
              request_model = None,
              response_model = None,
              registry = None,
              retry_config: Optional[Dict[str, Any]] = None) -> 'CustomClient':
    """
    Get a client for a custom endpoint.
    
    Args:
        endpoint_name: Name of the custom endpoint
        request_model: Optional request model class
        response_model: Optional response model class
        registry: Optional registry (uses default if None)
        retry_config: Optional retry configuration
        
    Returns:
        CustomClient instance for the endpoint
    """
    global _client_cache
    
    # Create a cache key
    cache_key = f"{endpoint_name}"
    
    # For clients with custom models or config, don't use cache
    use_cache = (request_model is None and 
                response_model is None and 
                retry_config is None)
    
    if use_cache and cache_key in _client_cache:
        return _client_cache[cache_key]
    
    # Get the registry
    registry = registry or get_registry()
    
    # Create the client
    client = CustomClient(
        endpoint_name=endpoint_name,
        request_model=request_model,
        response_model=response_model,
        registry=registry,
        retry_config=retry_config
    )
    
    # Cache the client if using standard configuration
    if use_cache:
        _client_cache[cache_key] = client
    
    return client

def create_custom_client(endpoint_name: str,
                        request_model = None,
                        response_model = None,
                        registry = None,
                        retry_config: Optional[Dict[str, Any]] = None) -> 'CustomClient':
    """
    Create a custom client for an endpoint.
    
    Args:
        endpoint_name: Name of the endpoint
        request_model: Optional request model class
        response_model: Optional response model class
        registry: Optional registry
        retry_config: Configuration for retry logic
        
    Returns:
        Custom client for the endpoint
    """
    # Use the get_client function to leverage caching
    return get_client(
        endpoint_name=endpoint_name,
        request_model=request_model,
        response_model=response_model,
        registry=registry,
        retry_config=retry_config
    )

def create_custom_client_proxy(registry = None,
                             retry_config: Optional[Dict[str, Any]] = None) -> 'CustomClientProxy':
    """
    Create a custom client proxy.
    
    Args:
        registry: Optional registry
        retry_config: Configuration for retry logic
        
    Returns:
        Custom client proxy
    """
    # Use the shared registry if none is specified
    registry = registry or get_registry()
    
    return CustomClientProxy(
        registry=registry,
        retry_config=retry_config
    )
