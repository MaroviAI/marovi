"""
Exception classes for the Marovi API.

This module provides a standardized exception hierarchy for consistent error handling.
"""

from typing import Dict, Any, Optional, List


class MaroviError(Exception):
    """Base exception class for all Marovi API errors."""
    
    def __init__(
        self, 
        message: str, 
        code: str = "unknown_error", 
        status_code: int = 500, 
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a MaroviError.
        
        Args:
            message: Human-readable error message
            code: Machine-readable error code
            status_code: HTTP status code
            details: Additional error details
        """
        self.message = message
        self.code = code
        self.status_code = status_code
        self.details = details or {}
        super().__init__(message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the exception to a dictionary representation."""
        return {
            "error": {
                "message": self.message,
                "code": self.code,
                "status_code": self.status_code,
                "details": self.details
            }
        }


class ValidationError(MaroviError):
    """Exception for validation errors."""
    
    def __init__(
        self, 
        message: str, 
        errors: List[Dict[str, Any]], 
        status_code: int = 400,
        code: str = "validation_error"
    ):
        """
        Initialize a ValidationError.
        
        Args:
            message: Human-readable error message
            errors: List of validation errors
            status_code: HTTP status code
            code: Machine-readable error code
        """
        details = {"errors": errors}
        super().__init__(message, code, status_code, details)


class AuthenticationError(MaroviError):
    """Exception for authentication errors."""
    
    def __init__(
        self, 
        message: str = "Authentication failed", 
        details: Optional[Dict[str, Any]] = None, 
        status_code: int = 401,
        code: str = "authentication_error"
    ):
        """
        Initialize an AuthenticationError.
        
        Args:
            message: Human-readable error message
            details: Additional error details
            status_code: HTTP status code
            code: Machine-readable error code
        """
        super().__init__(message, code, status_code, details)


class AuthorizationError(MaroviError):
    """Exception for authorization errors."""
    
    def __init__(
        self, 
        message: str = "Authorization failed", 
        details: Optional[Dict[str, Any]] = None, 
        status_code: int = 403,
        code: str = "authorization_error"
    ):
        """
        Initialize an AuthorizationError.
        
        Args:
            message: Human-readable error message
            details: Additional error details
            status_code: HTTP status code
            code: Machine-readable error code
        """
        super().__init__(message, code, status_code, details)


class NotFoundError(MaroviError):
    """Exception for resource not found errors."""
    
    def __init__(
        self, 
        message: str = "Resource not found", 
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        status_code: int = 404,
        code: str = "not_found"
    ):
        """
        Initialize a NotFoundError.
        
        Args:
            message: Human-readable error message
            resource_type: Type of resource that was not found
            resource_id: ID of resource that was not found
            status_code: HTTP status code
            code: Machine-readable error code
        """
        details = {}
        if resource_type:
            details["resource_type"] = resource_type
        if resource_id:
            details["resource_id"] = resource_id
        
        super().__init__(message, code, status_code, details)


class RateLimitError(MaroviError):
    """Exception for rate limit errors."""
    
    def __init__(
        self, 
        message: str = "Rate limit exceeded", 
        retry_after: Optional[int] = None,
        status_code: int = 429,
        code: str = "rate_limit_exceeded"
    ):
        """
        Initialize a RateLimitError.
        
        Args:
            message: Human-readable error message
            retry_after: Seconds to wait before retrying
            status_code: HTTP status code
            code: Machine-readable error code
        """
        details = {}
        if retry_after is not None:
            details["retry_after"] = retry_after
        
        super().__init__(message, code, status_code, details)


class ServerError(MaroviError):
    """Exception for server errors."""
    
    def __init__(
        self, 
        message: str = "Internal server error", 
        details: Optional[Dict[str, Any]] = None, 
        status_code: int = 500,
        code: str = "server_error"
    ):
        """
        Initialize a ServerError.
        
        Args:
            message: Human-readable error message
            details: Additional error details
            status_code: HTTP status code
            code: Machine-readable error code
        """
        super().__init__(message, code, status_code, details)


class ProviderError(MaroviError):
    """Exception for provider-specific errors."""
    
    def __init__(
        self, 
        message: str, 
        provider: str,
        provider_error: Optional[Exception] = None,
        status_code: int = 502,
        code: str = "provider_error"
    ):
        """
        Initialize a ProviderError.
        
        Args:
            message: Human-readable error message
            provider: Provider name
            provider_error: Original provider exception
            status_code: HTTP status code
            code: Machine-readable error code
        """
        details = {"provider": provider}
        if provider_error:
            details["provider_error"] = str(provider_error)
            details["provider_error_type"] = type(provider_error).__name__
        
        super().__init__(message, code, status_code, details)


class ConfigurationError(MaroviError):
    """Exception for configuration errors."""
    
    def __init__(
        self, 
        message: str, 
        details: Optional[Dict[str, Any]] = None, 
        status_code: int = 500,
        code: str = "configuration_error"
    ):
        """
        Initialize a ConfigurationError.
        
        Args:
            message: Human-readable error message
            details: Additional error details
            status_code: HTTP status code
            code: Machine-readable error code
        """
        super().__init__(message, code, status_code, details)


class TimeoutError(MaroviError):
    """Exception for timeout errors."""
    
    def __init__(
        self, 
        message: str = "Request timed out", 
        timeout: Optional[float] = None,
        status_code: int = 504,
        code: str = "timeout"
    ):
        """
        Initialize a TimeoutError.
        
        Args:
            message: Human-readable error message
            timeout: Timeout in seconds
            status_code: HTTP status code
            code: Machine-readable error code
        """
        details = {}
        if timeout is not None:
            details["timeout"] = timeout
        
        super().__init__(message, code, status_code, details)


class ResourceError(MaroviError):
    """
    Exception raised for errors related to cloud resources.
    
    This includes resource not found, access denied, etc.
    """
    pass


class OperationError(MaroviError):
    """
    Exception raised for errors in operations.
    
    This includes operation failures, timeouts, etc.
    """
    pass


class PermissionError(MaroviError):
    """
    Exception raised for permission errors.
    
    This includes insufficient permissions to perform an operation.
    """
    pass 