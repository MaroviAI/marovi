"""
Text cleaning endpoint for the Marovi API.

This module provides a custom endpoint for cleaning text from artifacts, leftover tags, 
and other unwanted elements that may remain after conversion between formats.
"""

import time
import logging
import os
import re
from typing import Dict, Any, List, Optional, Union

from marovi.api.core.base import CustomEndpoint
from marovi.api.core.base import ServiceType
from marovi.api.custom.schemas import CleanTextRequest, CleanTextResponse, SupportedFormat

# Configure logging
logger = logging.getLogger(__name__)

# Define supported formats for easy reference
SUPPORTED_FORMATS = [format.value for format in SupportedFormat]

class CleanText(CustomEndpoint):
    """
    Text cleaning service that uses LLMs.
    
    This endpoint provides text cleaning services using LLM models to clean up text artifacts
    and leftover elements from conversion between formats.
    """
    
    def __init__(self):
        """Initialize the text cleaner."""
        self.request_model = CleanTextRequest
        self.response_model = CleanTextResponse
        self.llm_client = None
        
        # Load the template
        self.template_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "prompts",
            "clean_text.jinja"
        )
        self.template = None
        if os.path.exists(self.template_path):
            with open(self.template_path, 'r') as f:
                self.template = f.read()
        
    def __call__(self, request) -> CleanTextResponse:
        """
        Clean text from artifacts and unwanted elements.
        
        Args:
            request: CleanTextRequest instance
        Returns:
            CleanTextResponse with cleaned text and metadata
        """
        if isinstance(request, dict):
            request = CleanTextRequest(**request)
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
            from marovi.api.core.router import default_router
            
            # Get the default LLM client
            llm_client = default_router.get_service(ServiceType.LLM)
            logger.info("Using default LLM client from router")
            return llm_client
        except Exception as e:
            logger.error(f"Failed to get default LLM client: {str(e)}")
            raise ValueError("No LLM client provided and could not get default client from router")
    
    def process(self, request: CleanTextRequest) -> CleanTextResponse:
        """
        Process a text cleaning request.
        
        Args:
            request: Text cleaning request
            
        Returns:
            Text cleaning response
        """
        start_time = time.time()
        
        try:
            # Get the LLM client
            llm_client = self._get_llm_client()
            
            # Extract parameters from request
            text = request.text
            format_value = request.format
            preserve_structure = request.preserve_structure
            preserve_links = request.preserve_links
            preserve_images = request.preserve_images
            remove_html_artifacts = request.remove_html_artifacts
            options = request.options or {}
            
            # Render the template
            from jinja2 import Template
            prompt_template = Template(self.template)
            prompt = prompt_template.render(
                text=text,
                format=format_value,
                preserve_structure=preserve_structure,
                preserve_links=preserve_links,
                preserve_images=preserve_images,
                remove_html_artifacts=remove_html_artifacts,
                **options
            )
            
            # Call the LLM
            llm_response = llm_client.complete(
                prompt,
                temperature=options.get('temperature', 0.1),
                model=options.get('model'),
                max_tokens=options.get('max_tokens', 8000),
                response_format={"type": "json_object"} if options.get('structured_response', True) else None
            )
            
            # Handle both string responses and LLMResponse objects
            if hasattr(llm_response, 'content'):
                llm_response = llm_response.content
            else:
                llm_response = str(llm_response)
            
            # Try to parse as JSON
            try:
                import json
                response_data = json.loads(llm_response)
                cleaned_text = response_data.get('cleaned_text', '')
                metadata = response_data.get('metadata', {})
                success = response_data.get('success', True)
            except (json.JSONDecodeError, TypeError, AttributeError) as e:
                logger.warning(f"Failed to parse LLM response as JSON: {str(e)}. Using raw response.")
                cleaned_text = llm_response
                metadata = {}
                success = True
                
            # Add additional metadata
            metadata.update({
                "latency": time.time() - start_time,
                "model": getattr(llm_client, 'model', 'default'),
                "temperature": options.get('temperature', 0.1),
            })
            
            # Create and return response
            return CleanTextResponse(
                text=text,
                cleaned_text=cleaned_text,
                format=format_value,
                success=success,
                metadata=metadata
            )
        except Exception as e:
            latency = time.time() - start_time
            logger.error(f"Text cleaning failed: {str(e)}")
            return CleanTextResponse(
                text=text,
                cleaned_text="",
                format=request.format,
                success=False,
                error=str(e),
                metadata={"latency": latency}
            )
    
    def clean(self, text: str, format_value: str, 
             provider: Optional[str] = None, **kwargs) -> CleanTextResponse:
        """
        Clean text from artifacts and unwanted elements.
        
        Args:
            text: Text to clean
            format_value: Format of the text (html, md, wiki, etc.)
            provider: Optional LLM provider to use (defaults to the router's default)
            **kwargs: Additional parameters like model, temperature
            
        Returns:
            CleanTextResponse with cleaned text and metadata
        """
        start_time = time.time()
        
        # Store the original LLM client
        original_llm_client = self.llm_client
        
        try:
            # If provider is specified, get an LLM client for that provider
            if provider:
                try:
                    # Lazy import to avoid circular dependencies
                    from marovi.api.core.router import default_router
                    
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
            
            # Ensure structured_response is set in options
            if 'options' not in kwargs:
                kwargs['options'] = {}
            if 'structured_response' not in kwargs and 'structured_response' not in kwargs.get('options', {}):
                if kwargs.get('options'):
                    kwargs['options']['structured_response'] = True
                else:
                    kwargs['options'] = {'structured_response': True}
                    
            # Create a request
            request = CleanTextRequest(
                text=text,
                format=format_value,
                preserve_structure=kwargs.get('preserve_structure', True),
                preserve_links=kwargs.get('preserve_links', True),
                preserve_images=kwargs.get('preserve_images', True),
                remove_html_artifacts=kwargs.get('remove_html_artifacts', True),
                options=kwargs.get('options')
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
            logger.error(f"Text cleaning failed: {str(e)}")
            
            # Create error response
            return CleanTextResponse(
                text=text,
                cleaned_text="",
                format=format_value,
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
    
    def clean_wiki_output(self, wiki_text: str) -> str:
        """
        Clean Wiki markup text specifically from common conversion artifacts.
        This is a light weight cleaning that doesn't use LLM.
        
        Args:
            wiki_text: Wiki markup text to clean
            
        Returns:
            Cleaned Wiki markup text
        """
        # Remove HTML comments
        cleaned = re.sub(r'<!--.*?-->', '', wiki_text, flags=re.DOTALL)
        
        # Fix double brackets in links
        cleaned = re.sub(r'\[\[\[\[(.*?)\]\]\]\]', r'[[\1]]', cleaned)
        
        # Fix spacing in headings
        cleaned = re.sub(r'(?m)^(=+)(\S)', r'\1 \2', cleaned)  # Start of heading
        cleaned = re.sub(r'(\S)(=+)$', r'\1 \2', cleaned)  # End of heading
        
        # Fix HTML entities
        entities = {
            '&nbsp;': ' ',
            '&lt;': '<',
            '&gt;': '>',
            '&amp;': '&',
            '&quot;': '"',
            '&apos;': "'",
        }
        for entity, replacement in entities.items():
            cleaned = cleaned.replace(entity, replacement)
        
        # Remove any remaining HTML tags
        cleaned = re.sub(r'<(?!math|\/math)[^>]+>', '', cleaned)
        
        return cleaned
    
    def get_capabilities(self) -> List[str]:
        """Get the capabilities of this text cleaner."""
        return ["text_cleaning"]
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about this text cleaner."""
        return {
            "supported_formats": SUPPORTED_FORMATS,
            "type": "text_cleaner",
            "provider": "llm",
            "uses_llm": True
        }
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported formats."""
        return SUPPORTED_FORMATS