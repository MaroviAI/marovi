"""
Logging utilities for the API.

This module provides consistent logging functionality across the API.
"""

import logging
import sys
from typing import Optional, Dict, Any
from datetime import datetime

from ..config import settings

def setup_logging() -> None:
    """Set up logging configuration."""
    log_level = getattr(logging, settings.LOG_LEVEL.upper())
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear existing handlers
    root_logger.handlers = []
    
    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Add file handler if debug mode is enabled
    if settings.DEBUG:
        file_handler = logging.FileHandler('api.log')
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name."""
    return logging.getLogger(name)

class RequestLogger:
    """Logger for API requests and responses."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize the request logger."""
        self.logger = logger or get_logger(__name__)
    
    def log_request(self, 
                   service: str,
                   request: Dict[str, Any],
                   metadata: Optional[Dict[str, Any]] = None) -> None:
        """Log an API request."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "service": service,
            "request": request,
            "metadata": metadata or {}
        }
        self.logger.info(f"API Request: {log_data}")
    
    def log_response(self,
                    service: str,
                    response: Dict[str, Any],
                    latency: float,
                    metadata: Optional[Dict[str, Any]] = None) -> None:
        """Log an API response."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "service": service,
            "response": response,
            "latency": latency,
            "metadata": metadata or {}
        }
        self.logger.info(f"API Response: {log_data}")
    
    def log_error(self,
                 service: str,
                 error: Exception,
                 request: Optional[Dict[str, Any]] = None,
                 metadata: Optional[Dict[str, Any]] = None) -> None:
        """Log an API error."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "service": service,
            "error": str(error),
            "error_type": type(error).__name__,
            "request": request,
            "metadata": metadata or {}
        }
        self.logger.error(f"API Error: {log_data}")
    
    def log_batch_request(self,
                         service: str,
                         request: Dict[str, Any],
                         batch_size: int,
                         metadata: Optional[Dict[str, Any]] = None) -> None:
        """Log a batch API request."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "service": service,
            "request": request,
            "batch_size": batch_size,
            "metadata": metadata or {}
        }
        self.logger.info(f"API Batch Request: {log_data}")
    
    def log_batch_response(self,
                          service: str,
                          response: Dict[str, Any],
                          total_latency: float,
                          avg_latency: float,
                          batch_size: int,
                          metadata: Optional[Dict[str, Any]] = None) -> None:
        """Log a batch API response."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "service": service,
            "response": response,
            "total_latency": total_latency,
            "avg_latency": avg_latency,
            "batch_size": batch_size,
            "metadata": metadata or {}
        }
        self.logger.info(f"API Batch Response: {log_data}")

# Initialize logging
setup_logging()

# Create default request logger
request_logger = RequestLogger()
