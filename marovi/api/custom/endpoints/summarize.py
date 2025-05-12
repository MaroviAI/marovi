"""
Summarizer endpoint for text summarization.

This module provides a custom endpoint for summarizing text using LLM capabilities.
"""

import logging
import time
from typing import Dict, Any, List, Optional, Union, Tuple

from ..core.base import CustomEndpoint
from ..core.errors import CustomEndpointError
from ..schemas import SummarizationRequest, SummarizationResponse, SummaryStyle

logger = logging.getLogger(__name__)

class SummarizationError(CustomEndpointError):
    """Exception raised when text summarization fails."""
    pass

class Summarizer(CustomEndpoint):
    """
    Custom endpoint for text summarization.
    
    This class provides methods for summarizing text using LLM capabilities with
    various style options and parameters.
    """
    
    def __init__(self, llm_client=None):
        """
        Initialize the Summarizer endpoint.
        
        Args:
            llm_client: LLM client instance for interacting with language models
        """
        self.llm_client = llm_client
        self.request_model = SummarizationRequest
        self.response_model = SummarizationResponse
        logger.debug("Initialized Summarizer endpoint")
    
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
            from marovi.api.core.base import ServiceType
            
            # Get the default LLM client
            llm_client = default_router.get_service(ServiceType.LLM)
            logger.info("Using default LLM client from router")
            return llm_client
        except Exception as e:
            logger.error(f"Failed to get default LLM client: {str(e)}")
            raise ValueError("No LLM client provided and could not get default client from router")
    
    def __call__(self, request):
        """
        Process a summarization request.
        
        Args:
            request: SummarizationRequest instance
            
        Returns:
            SummarizationResponse instance
        """
        # This method is required by the CustomEndpoint base class
        if isinstance(request, dict):
            request = SummarizationRequest(**request)
        return self.summarize(request)
    
    def _count_words(self, text):
        """Count the number of words in text."""
        return len(text.split())
    
    def process(self, request):
        """
        Process a summarization request.
        
        Args:
            request: SummarizationRequest instance
            
        Returns:
            SummarizationResponse instance
        """
        # This method is required by the CustomEndpoint base class
        if isinstance(request, dict):
            request = SummarizationRequest(**request)
        return self.summarize(request)
    
    def summarize(self, request_data: Union[Dict[str, Any], SummarizationRequest]) -> SummarizationResponse:
        """
        Summarize text based on the provided request parameters.
        
        Args:
            request_data: SummarizationRequest or dict with summarization parameters
            
        Returns:
            SummarizationResponse with the summarization results
        
        Raises:
            SummarizationError: If summarization fails
        """
        start_time = time.time()
        
        # Convert dict to SummarizationRequest if necessary
        if isinstance(request_data, dict):
            try:
                request = SummarizationRequest(**request_data)
            except Exception as e:
                logger.error(f"Invalid summarization request: {e}")
                raise SummarizationError(f"Invalid summarization request: {e}")
        else:
            request = request_data
        
        logger.info(f"Summarizing text with style: {request.style}")
        
        # Count words in original text for metadata
        word_count_original = self._count_words(request.text)
        
        try:
            # Get the LLM client
            llm_client = self._get_llm_client()
            
            # Create prompt for the LLM
            prompt = self._create_summarization_prompt(
                text=request.text,
                style=request.style,
                max_length=request.max_length,
                include_keywords=request.include_keywords,
                preserve_quotes=request.preserve_quotes,
                language=request.language,
                options=request.options or {}
            )
            
            # Call the LLM
            summary = llm_client.complete(prompt)
            
            # Handle string or LLMResponse
            if hasattr(summary, 'content'):
                summary = summary.content
                
            # Count words in summary
            word_count_summary = self._count_words(summary)
            
            # Extract keywords if present
            keywords = None
            if request.include_keywords and "Keywords:" in summary:
                # Look for keywords section at the end of the summary
                summary_parts = summary.split("Keywords:", 1)
                if len(summary_parts) > 1:
                    summary = summary_parts[0].strip()
                    keywords_text = summary_parts[1].strip()
                    keywords = [k.strip() for k in keywords_text.split(",")]
            
            # Create response
            response = SummarizationResponse(
                text=request.text,
                summary=summary,
                style=request.style,
                word_count_original=word_count_original,
                word_count_summary=word_count_summary,
                keywords=keywords,
                success=True,
                metadata={
                    "processing_time": time.time() - start_time,
                    "model": getattr(llm_client, 'model', 'default')
                }
            )
            
            logger.info(f"Successfully summarized text: {word_count_original} words → {word_count_summary} words")
            return response
            
        except Exception as e:
            logger.error(f"Summarization failed: {str(e)}")
            return SummarizationResponse(
                text=request.text,
                summary="",
                style=request.style,
                word_count_original=word_count_original,
                word_count_summary=0,
                success=False,
                error=str(e),
                metadata={"processing_time": time.time() - start_time}
            )
    
    def _create_summarization_prompt(
        self, 
        text: str,
        style: str,
        max_length: Optional[int] = None,
        include_keywords: bool = False,
        preserve_quotes: bool = False,
        language: str = "en",
        options: Dict[str, Any] = {}
    ) -> str:
        """
        Create a prompt for summarization.
        
        Args:
            text: Text to summarize
            style: Summary style (bullet, paragraph, structured, concise)
            max_length: Maximum length of summary in words
            include_keywords: Whether to include keywords
            preserve_quotes: Whether to preserve important quotes
            language: Language code for the summary
            options: Additional options for the summarization
            
        Returns:
            Prompt for the LLM
        """
        # Basic prompt
        prompt = f"I need to create a summary of the following text:\n\n{text}\n\n"
        
        # Style-specific instructions
        if style == "bullet":
            prompt += "Please provide a bullet-point summary with the main ideas and key points.\n"
        elif style == "paragraph":
            prompt += "Please provide a concise paragraph summary that captures the essence of the text.\n"
        elif style == "structured":
            prompt += "Please provide a structured summary with sections for main topics, key points, and conclusions.\n"
        else:  # concise
            prompt += "Please provide a concise summary that captures the essence of the text.\n"
        
        # Additional options
        if max_length:
            prompt += f"The summary should be approximately {max_length} words long.\n"
        
        if include_keywords:
            prompt += "Include a list of key terms or keywords at the end of the summary.\n"
        
        if preserve_quotes:
            prompt += "Preserve important quotes from the original text in the summary.\n"
        
        if language and language != "en":
            prompt += f"Provide the summary in {language} language.\n"
        
        # General guidelines
        prompt += """
        Ensure the summary is:
        - Accurate and faithful to the original text
        - Concise but comprehensive
        - Well-organized
        - Focused on the most important information
        """
        
        if style == "structured":
            prompt += "- With appropriate headings and structure\n"
        
        if style == "bullet":
            prompt += "- Formatted as bullet points for easy reading\n"
        
        return prompt
    
    def get_capabilities(self) -> List[str]:
        """Get the capabilities of this summarizer."""
        return ["summarization"]
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about this summarizer."""
        return {
            "type": "summarizer",
            "supported_styles": [style.value for style in SummaryStyle],
            "uses_llm": True
        }