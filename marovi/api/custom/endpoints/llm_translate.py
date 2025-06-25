"""
LLM-based translation service.

This module provides a translation service that uses LLMs to translate text from one language to another.

Custom endpoints can access service clients (LLM, Translation) from the router in two ways:
1. By implementing a _get_llm_client() or _get_translation_client() method that tries to get
   the client from the router when not explicitly provided.
2. Through the CustomClient._initialize_endpoint_with_services() method, which will set
   the llm_client and translation_client attributes on the endpoint instance if they implement
   the corresponding getter methods.

This design allows custom endpoints to be flexible - they can work with explicitly provided
clients or automatically get the default clients from the router.
"""

from typing import Dict, Any, Optional, List, Union
from ..core.base import CustomEndpoint
from pydantic import BaseModel, Field
import logging
import time

# Configure logging
logger = logging.getLogger(__name__)

# Define a TranslationError exception
class TranslationError(Exception):
    """Exception raised when translation fails."""
    pass

class TranslationRequest(BaseModel):
    """Request model for translation."""
    text: str
    source_lang: str
    target_lang: str
    options: Optional[Dict[str, Any]] = None

class TranslationResponse(BaseModel):
    """Response model for translation."""
    content: str
    source_lang: str
    target_lang: str
    success: bool = True
    error: Optional[str] = None
    latency: float = Field(default=0.0)
    format: str = Field(default="text")

class LLMTranslate(CustomEndpoint):
    """
    Translation service that uses LLMs for translation.
    
    This endpoint provides translation services using LLM models instead of
    traditional translation APIs. It follows the same interface as the
    standard translation endpoints for consistency.
    """
    
    def __init__(self, llm_client=None):
        """Initialize the LLM translator."""
        self.request_model = TranslationRequest
        self.response_model = TranslationResponse
        self.llm_client = llm_client
    
    def __call__(self, text: str, source_lang: str, target_lang: str, **kwargs) -> Dict[str, Any]:
        """
        Translate text using an LLM.
        
        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code
            **kwargs: Additional parameters like model, instruction, temperature
            
        Returns:
            JSON response with translated text and metadata
        """
        result = self.translate(text, source_lang, target_lang, **kwargs)
        return result
    
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
    
    def process(self, request: TranslationRequest) -> TranslationResponse:
        """
        Process a translation request.
        
        Args:
            request: Translation request
            
        Returns:
            Translation response
        """
        start_time = time.time()
        
        try:
            # Get the LLM client
            llm_client = self._get_llm_client()
            
            # Extract parameters from request
            text = request.text
            source_lang = request.source_lang
            target_lang = request.target_lang
            options = request.options or {}
            
            # Create prompt for the LLM
            prompt = f"""
            Translate the following text from {source_lang} to {target_lang}:
            
            {text}
            
            Only return the translated text, without any explanations or notes.
            """
            
            # Call the LLM
            translated_text = llm_client.complete(prompt).strip()
            
            # Calculate latency
            latency = time.time() - start_time
            
            # Create and return response
            return TranslationResponse(
                content=translated_text,
                source_lang=source_lang,
                target_lang=target_lang,
                success=True,
                latency=latency,
                format="text"
            )
        except Exception as e:
            latency = time.time() - start_time
            logger.error(f"Translation failed: {str(e)}")
            return TranslationResponse(
                content="",
                source_lang=request.source_lang, 
                target_lang=request.target_lang,
                success=False,
                error=str(e),
                latency=latency,
                format="text"
            )
    
    def translate(self, text: str, source_lang: str, target_lang: str, 
                provider: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Translate text using an LLM.
        
        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code
            provider: Optional LLM provider to use (defaults to the router's default)
            **kwargs: Additional parameters like model, instruction, temperature
            
        Returns:
            JSON response with translated text and metadata
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
                
            # Create a translation instruction
            instruction = kwargs.get('instruction') or self._create_translation_instruction(source_lang, target_lang)
            
            # Ensure text is a string (handle list input if needed)
            if isinstance(text, list):
                text = "\n".join([str(item) for item in text])
            elif not isinstance(text, str):
                text = str(text)
            
            # Use the provided LLM client to translate
            response = self.llm_client.complete(
                instruction + "\n\n" + text,
                temperature=kwargs.get('temperature', 0.1),
                model=kwargs.get('model'),
                max_tokens=kwargs.get('max_tokens', 4000)
            )
            
            # Handle both string responses and LLMResponse objects
            if hasattr(response, 'content'):
                translated_text = response.content.strip()
            else:
                # Assume it's a string
                translated_text = str(response).strip()
            
            # Create a standardized response similar to other translation endpoints
            result = {
                "provider": provider or getattr(self.llm_client, 'provider_type_str', 'llm'),
                "text": text,
                "translated_text": translated_text,
                "source_lang": source_lang,
                "target_lang": target_lang,
                "metadata": {
                    "latency": time.time() - start_time,
                    "model": kwargs.get('model', getattr(self.llm_client, 'model', 'default')),
                    "temperature": kwargs.get('temperature', 0.1),
                }
            }
            
            return result
            
        except Exception as e:
            # Log the error
            logger.error(f"LLM translation failed: {str(e)}")
            
            # Raise a specific exception
            raise TranslationError(f"LLM translation failed: {str(e)}")
        finally:
            # Restore the original LLM client
            self.llm_client = original_llm_client
    
    def _create_translation_instruction(self, source_lang: str, target_lang: str) -> str:
        """Create a translation instruction for the LLM."""
        if source_lang == 'auto':
            source_lang = 'the source language'
        else:
            source_lang = self._get_full_language_name(source_lang)
            
        target_lang = self._get_full_language_name(target_lang)
            
        return f"Translate the following text from {source_lang} to {target_lang}. Output only the translated text, with no explanations or notes."
    
    def get_capabilities(self) -> List[str]:
        """Get the capabilities of this translator."""
        return ["translation", "llm_translate"]
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about this translator."""
        return {
            "type": "translator",
            "provider": "llm",
            "supports_batch": False,
            "uses_llm": True
        }
    
    def _get_full_language_name(self, lang_code: str) -> str:
        """
        Convert a language code to a full language name.
        
        Args:
            lang_code: ISO language code (e.g., 'en', 'fr')
            
        Returns:
            Full language name (e.g., 'English', 'French')
        """
        # Common language codes to names mapping
        language_map = {
            'en': 'English',
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German',
            'it': 'Italian',
            'pt': 'Portuguese',
            'ru': 'Russian',
            'zh': 'Chinese',
            'ja': 'Japanese',
            'ko': 'Korean',
            'ar': 'Arabic',
            'hi': 'Hindi',
            'bn': 'Bengali',
            'nl': 'Dutch',
            'pl': 'Polish',
            'sv': 'Swedish',
            'fi': 'Finnish',
            'no': 'Norwegian',
            'da': 'Danish',
            'tr': 'Turkish',
            'el': 'Greek',
            'he': 'Hebrew',
            'th': 'Thai',
            'vi': 'Vietnamese',
            'cs': 'Czech',
            'hu': 'Hungarian',
            'ro': 'Romanian',
            'uk': 'Ukrainian',
            'auto': 'Auto-detected language'
        }
        
        # Return the full name if found, otherwise use the code
        return language_map.get(lang_code.lower(), lang_code) 

def create_llm_translate(llm_client=None, **kwargs):
    """
    Factory function to create an LLMTranslate instance.
    
    Args:
        llm_client: Optional LlmClient instance to use
        **kwargs: Additional keyword arguments to pass to LLMTranslate constructor
        
    Returns:
        LLMTranslate instance
    """
    return LLMTranslate(llm_client=llm_client, **kwargs)