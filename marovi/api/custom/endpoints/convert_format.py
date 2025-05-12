"""
Format conversion endpoint for the Marovi API.

This module provides a custom endpoint for converting text between different formats
such as HTML, Markdown, Wiki, LaTeX, etc. using LLM capabilities.
"""

import time
import logging
import os
from typing import Dict, Any, List, Optional, Union

from ..core.base import CustomEndpoint
from ...core.base import ServiceType
from ..schemas import FormatConversionRequest, FormatConversionResponse, SupportedFormat

# Configure logging
logger = logging.getLogger(__name__)

# Define a format conversion exception
class FormatConversionError(Exception):
    """Exception raised when format conversion fails."""
    pass

# Define supported formats for easy reference
SUPPORTED_FORMATS = [format.value for format in SupportedFormat]

class FormatConverter(CustomEndpoint):
    """
    Format conversion service that uses LLMs.
    
    This endpoint provides format conversion services using LLM models to convert
    text between different formats such as HTML, Markdown, Wiki, LaTeX, etc.
    """
    
    def __init__(self):
        """Initialize the format converter."""
        self.request_model = FormatConversionRequest
        self.response_model = FormatConversionResponse
        self.llm_client = None
        
        # Load the template
        self.template_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "prompts",
            "convert_format.jinja"
        )
        self.template = None
        if os.path.exists(self.template_path):
            with open(self.template_path, 'r') as f:
                self.template = f.read()
        
    def __call__(self, request) -> FormatConversionResponse:
        """
        Convert text from one format to another.
        
        Args:
            request: FormatConversionRequest instance
        Returns:
            FormatConversionResponse with converted text and metadata
        """
        if isinstance(request, dict):
            request = FormatConversionRequest(**request)
        return self.process(request)
    
    def _get_llm_client(self):
        """
        Get the LLM client, either the one provided in the constructor or from the router.
        
        Returns:
            LLM client instance
        """
        if self.llm_client:
            return self.llm_client
            
        # Get the default LLM client from the router
        try:
            # Lazy import to avoid circular dependencies
            from ...core.router import default_router
            from ...core.base import ServiceType
            
            # Get the default LLM client
            llm_client = default_router.get_service(ServiceType.LLM)
            logger.info("Using default LLM client from router")
            return llm_client
        except Exception as e:
            logger.error(f"Failed to get default LLM client: {str(e)}")
            raise ValueError("No LLM client provided and could not get default client from router")
    
    def process(self, request: FormatConversionRequest) -> FormatConversionResponse:
        """
        Process a format conversion request.
        
        Args:
            request: Format conversion request
            
        Returns:
            Format conversion response
        """
        start_time = time.time()
        
        try:
            # Get the LLM client
            llm_client = self._get_llm_client()
            
            # Extract parameters from request
            text = request.text
            source_format = request.source_format
            target_format = request.target_format
            preserve_structure = request.preserve_structure
            preserve_links = request.preserve_links
            preserve_images = request.preserve_images
            options = request.options or {}
            
            # If the source and target formats are the same, just return the original text
            if source_format == target_format:
                return FormatConversionResponse(
                    text=text,
                    converted_text=text,
                    source_format=source_format,
                    target_format=target_format,
                    success=True,
                    metadata={"note": "Source and target formats are the same, no conversion needed"}
                )
            
            # Prepare the prompt
            # If we have a Jinja template loaded, we'll use that for a more structured prompt
            # Otherwise, we'll create a simple prompt string
            if self.template:
                try:
                    from jinja2 import Template
                    prompt_template = Template(self.template)
                    prompt = prompt_template.render(
                        text=text,
                        source_format=source_format,
                        target_format=target_format,
                        preserve_structure=preserve_structure,
                        preserve_links=preserve_links,
                        preserve_images=preserve_images,
                        **options
                    )
                except Exception as e:
                    logger.warning(f"Failed to render template: {str(e)}. Using fallback prompt.")
                    prompt = self._create_fallback_prompt(
                        text, source_format, target_format, 
                        preserve_structure, preserve_links, preserve_images
                    )
            else:
                prompt = self._create_fallback_prompt(
                    text, source_format, target_format, 
                    preserve_structure, preserve_links, preserve_images
                )
            
            # Call the LLM
            converted_text = llm_client.complete(
                prompt,
                temperature=options.get('temperature', 0.1),
                model=options.get('model'),
                max_tokens=options.get('max_tokens', 8000)
            )
            
            # Handle both string responses and LLMResponse objects
            if hasattr(converted_text, 'content'):
                converted_text = converted_text.content
            else:
                converted_text = str(converted_text)
            
            # Create and return response
            return FormatConversionResponse(
                text=text,
                converted_text=converted_text,
                source_format=source_format,
                target_format=target_format,
                success=True,
                metadata={
                    "latency": time.time() - start_time,
                    "model": getattr(llm_client, 'model', 'default'),
                    "temperature": options.get('temperature', 0.1),
                }
            )
        except Exception as e:
            latency = time.time() - start_time
            logger.error(f"Format conversion failed: {str(e)}")
            return FormatConversionResponse(
                text=text,
                converted_text="",
                source_format=request.source_format, 
                target_format=request.target_format,
                success=False,
                error=str(e),
                metadata={"latency": latency}
            )
    
    def _create_fallback_prompt(self, text, source_format, target_format, 
                               preserve_structure, preserve_links, preserve_images):
        """Create a fallback prompt for format conversion if the template can't be loaded."""
        prompt = f"Convert the following text from {source_format} format to {target_format} format:\n\n{text}\n\n"
        
        if preserve_structure:
            prompt += "Please preserve the document structure, including headings, sections, lists, and tables.\n"
        if preserve_links:
            prompt += "Please preserve all links. Make sure URLs are properly formatted for the target format.\n"
        if preserve_images:
            prompt += "Please preserve all image references. Convert image tags to the appropriate target format syntax.\n"
            
        prompt += f"\nProvide only the converted {target_format} text without any explanations or comments."
        
        return prompt
    
    def convert(self, text: str, source_format: str, target_format: str, 
               provider: Optional[str] = None, **kwargs) -> FormatConversionResponse:
        """
        Convert text from one format to another.
        
        Args:
            text: Text to convert
            source_format: Source format code (html, md, wiki, etc.)
            target_format: Target format code
            provider: Optional LLM provider to use (defaults to the router's default)
            **kwargs: Additional parameters like model, temperature
            
        Returns:
            FormatConversionResponse with converted text and metadata
        """
        start_time = time.time()
        
        # Store the original LLM client
        original_llm_client = self.llm_client
        
        try:
            # If provider is specified, get an LLM client for that provider
            if provider:
                try:
                    # Lazy import to avoid circular dependencies
                    from ...core.router import default_router
                    from ...core.base import ServiceType
                    
                    # Get the LLM client for the specified provider
                    self.llm_client = default_router.get_service(ServiceType.LLM, provider=provider)
                    logger.info(f"Using specified LLM provider: {provider}")
                except Exception as e:
                    logger.warning(f"Could not use specified provider {provider}: {str(e)}. Using default.")
                    # Ensure we have a default client
                    if not self.llm_client:
                        self.llm_client = self._get_llm_client()
            elif not self.llm_client:
                # Get the default LLM client if not already set
                self.llm_client = self._get_llm_client()
            
            # Create a request
            request = FormatConversionRequest(
                text=text,
                source_format=source_format,
                target_format=target_format,
                preserve_structure=kwargs.get('preserve_structure', True),
                preserve_links=kwargs.get('preserve_links', True),
                preserve_images=kwargs.get('preserve_images', True),
                options=kwargs
            )
            
            # Process the request
            response = self.process(request)
            
            # Add provider information to metadata
            if response.metadata is None:
                response.metadata = {}
            response.metadata["provider"] = provider or getattr(self.llm_client, 'provider_type_str', 'llm')
            
            return response
            
        except Exception as e:
            # Log the error
            logger.error(f"Format conversion failed: {str(e)}")
            
            # Create error response
            return FormatConversionResponse(
                text=text,
                converted_text="",
                source_format=source_format,
                target_format=target_format,
                success=False,
                error=str(e),
                metadata={
                    "latency": time.time() - start_time,
                    "provider": provider or getattr(self.llm_client, 'provider_type_str', 'llm') if self.llm_client else "unknown"
                }
            )
        finally:
            # Restore the original LLM client
            self.llm_client = original_llm_client
    
    def get_capabilities(self) -> List[str]:
        """Get the capabilities of this converter."""
        return ["format_conversion"]
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about this converter."""
        return {
            "supported_formats": SUPPORTED_FORMATS,
            "type": "format_converter",
            "provider": "llm",
            "uses_llm": True
        }
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported formats."""
        return SUPPORTED_FORMATS
