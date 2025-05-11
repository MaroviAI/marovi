"""
Custom error classes for the Marovi API custom endpoints.

This module provides error classes for custom endpoints to use
for specific error conditions.
"""

class CustomEndpointError(Exception):
    """Base exception class for errors in custom endpoints."""
    pass

class ValidationError(CustomEndpointError):
    """Exception raised when input validation fails."""
    pass

class ProcessingError(CustomEndpointError):
    """Exception raised when processing fails."""
    pass

class ServiceUnavailableError(CustomEndpointError):
    """Exception raised when a required service is unavailable."""
    pass