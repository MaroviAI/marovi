"""
Pipeline steps for MaroviAPI integrations.

This module provides pipeline steps that wrap MaroviAPI client functionality,
allowing seamless integration of MaroviAPI services into processing pipelines.
"""

import logging
import inspect
from typing import Any, Callable, Dict, List, Optional, Union, TypeVar, Generic, Type, get_args

from marovi.pipelines.core import PipelineStep
from marovi.pipelines.context import PipelineContext
from marovi.api.core.client import MaroviAPI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Type variables
InputType = TypeVar('InputType')
OutputType = TypeVar('OutputType')

__all__ = [
    'MaroviAPIStep', 
    'TranslateStep', 
    'LLMTranslateStep', 
    'SummarizeStep', 
    'CompleteStep', 
    'ConvertFormatStep',
    'CleanTextStep'
]

class MaroviAPIStep(PipelineStep[InputType, OutputType], Generic[InputType, OutputType]):
    """
    A dynamic pipeline step that wraps any MaroviAPI client endpoint.
    
    This step allows seamless integration of any MaroviAPI service into a processing pipeline
    by dynamically dispatching to the appropriate client method based on the service and endpoint
    parameters. It handles both single items and batches, automatically leveraging any batch or 
    async capabilities provided by the underlying API endpoints.
    """
    
    def __init__(self, 
                 service: str,
                 endpoint: str,
                 client: Optional[MaroviAPI] = None,
                 endpoint_kwargs: Optional[Dict[str, Any]] = None,
                 input_field: str = "text",
                 output_field: Optional[str] = None,
                 batch_size: int = 10,
                 process_mode: str = "parallel",
                 step_id: Optional[str] = None):
        """
        Initialize a MaroviAPI step.
        
        Args:
            service: Service name in the MaroviAPI client (e.g., 'llm', 'translation', 'custom')
            endpoint: Method name in the service (e.g., 'complete', 'translate', 'summarize')
            client: Optional MaroviAPI client instance (will be created if not provided)
            endpoint_kwargs: Additional kwargs to pass to the endpoint method for all calls
            input_field: Field name to use for the input in the API call
            output_field: Field name to extract from the API response (if None, returns full response)
            batch_size: Number of items to process in a batch
            process_mode: Processing mode ('sequential', 'parallel', or 'custom')
            step_id: Optional unique identifier for this step
        """
        super().__init__(
            batch_size=batch_size,
            batch_handling='inherent',
            process_mode=process_mode,
            step_id=step_id or f"marovi_api_{service}_{endpoint}"
        )
        
        self.service = service
        self.endpoint = endpoint
        self.client = client or MaroviAPI()
        self.endpoint_kwargs = endpoint_kwargs or {}
        self.input_field = input_field
        self.output_field = output_field
        
        # Validate that the service and endpoint exist
        try:
            service_client = getattr(self.client, service)
            endpoint_method = getattr(service_client, endpoint)
            
            # Store the endpoint method for later use
            self.endpoint_method = endpoint_method
            
            # Check if the endpoint has a batch variant
            self.has_batch = hasattr(service_client, f"batch_{endpoint}")
            if self.has_batch:
                self.batch_method = getattr(service_client, f"batch_{endpoint}")
            
            # Check if the endpoint has an async variant
            self.has_async = hasattr(service_client, f"{endpoint}_async")
            if self.has_async:
                self.async_method = getattr(service_client, f"{endpoint}_async")
                
            logger.info(f"Initialized {self.step_id} with endpoint {service}.{endpoint}")
            logger.debug(f"Batch support: {self.has_batch}, Async support: {self.has_async}")
            
        except AttributeError as e:
            raise ValueError(f"Invalid service or endpoint: {service}.{endpoint}") from e
    
    def process(self, inputs: List[InputType], context: PipelineContext) -> List[OutputType]:
        """
        Process inputs using the specified MaroviAPI endpoint.
        
        This method handles different input types (single values or objects) and
        automatically chooses the most efficient processing strategy based on the
        available endpoint variants (batch, async, etc.).
        
        Args:
            inputs: List of input items
            context: Pipeline context
            
        Returns:
            List of output items from the API
        """
        if not inputs:
            logger.warning(f"{self.step_id}: No inputs provided")
            return []
        
        # Determine if inputs are simple values or complex objects
        use_simple_input = not isinstance(inputs[0], dict)
        
        # Handle batch endpoint if available and appropriate
        if self.has_batch and len(inputs) > 1:
            logger.info(f"{self.step_id}: Using batch endpoint for {len(inputs)} items")
            return self._process_batch(inputs, use_simple_input, context)
        
        # Handle individual processing (with or without async)
        logger.info(f"{self.step_id}: Processing {len(inputs)} items individually")
        return self._process_individual(inputs, use_simple_input, context)
    
    def _process_batch(self, inputs: List[InputType], use_simple_input: bool, 
                      context: PipelineContext) -> List[OutputType]:
        """Process inputs using the batch endpoint."""
        try:
            # Prepare batch inputs
            if use_simple_input:
                # For simple inputs like strings, prepare a list for the input field
                batch_inputs = {self.input_field: [item for item in inputs]}
                batch_inputs.update(self.endpoint_kwargs)
            else:
                # For dict inputs, merge with endpoint_kwargs
                batch_inputs = [{**item, **self.endpoint_kwargs} for item in inputs]
            
            # Call the batch endpoint
            batch_results = self.batch_method(**batch_inputs)
            
            # Extract outputs based on output_field
            if self.output_field and isinstance(batch_results, list) and batch_results:
                if isinstance(batch_results[0], dict):
                    return [result.get(self.output_field) for result in batch_results]
            
            return batch_results
            
        except Exception as e:
            logger.error(f"{self.step_id}: Batch processing failed: {str(e)}")
            # Fall back to individual processing
            logger.info(f"{self.step_id}: Falling back to individual processing")
            return self._process_individual(inputs, use_simple_input, context)
    
    def _process_individual(self, inputs: List[InputType], use_simple_input: bool,
                           context: PipelineContext) -> List[OutputType]:
        """Process inputs individually."""
        results = []
        
        for item in inputs:
            try:
                # Prepare input for the endpoint
                if use_simple_input:
                    # For simple inputs like strings
                    kwargs = {self.input_field: item, **self.endpoint_kwargs}
                else:
                    # For dict inputs, merge with endpoint_kwargs
                    kwargs = {**item, **self.endpoint_kwargs}
                
                # Call the endpoint
                result = self.endpoint_method(**kwargs)
                
                # Extract output if needed
                if self.output_field and isinstance(result, dict):
                    result = result.get(self.output_field)
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"{self.step_id}: Processing item failed: {str(e)}")
                # Append None or a default error value
                results.append(None)
        
        return results


# Convenience factory methods for common API endpoints

def TranslateStep(source_lang: str, target_lang: str, provider: str = "google", 
                 batch_size: int = 10, step_id: Optional[str] = None) -> MaroviAPIStep[str, str]:
    """
    Create a step that translates text using the MaroviAPI translation service.
    
    Args:
        source_lang: Source language code
        target_lang: Target language code
        provider: Translation provider to use
        batch_size: Number of items to process in a batch
        step_id: Optional unique identifier for this step
        
    Returns:
        A configured MaroviAPIStep for translation
    """
    return MaroviAPIStep(
        service="translation",
        endpoint="translate",
        endpoint_kwargs={
            "source_lang": source_lang,
            "target_lang": target_lang,
            "provider": provider
        },
        input_field="text",
        batch_size=batch_size,
        step_id=step_id or f"translate_{source_lang}_to_{target_lang}"
    )


def LLMTranslateStep(source_lang: str, target_lang: str, provider: str = "openai", 
                    batch_size: int = 5, step_id: Optional[str] = None) -> MaroviAPIStep[str, Dict[str, str]]:
    """
    Create a step that translates text using the MaroviAPI custom LLM translation endpoint.
    
    Args:
        source_lang: Source language code
        target_lang: Target language code
        provider: LLM provider to use
        batch_size: Number of items to process in a batch
        step_id: Optional unique identifier for this step
        
    Returns:
        A configured MaroviAPIStep for LLM-based translation
    """
    return MaroviAPIStep(
        service="custom",
        endpoint="llm_translate",
        endpoint_kwargs={
            "source_lang": source_lang,
            "target_lang": target_lang,
            "provider": provider
        },
        input_field="text",
        output_field="translated_text",
        batch_size=batch_size,
        step_id=step_id or f"llm_translate_{source_lang}_to_{target_lang}"
    )


def SummarizeStep(style: str = "paragraph", max_length: Optional[int] = None,
                 include_keywords: bool = False, batch_size: int = 5,
                 step_id: Optional[str] = None) -> MaroviAPIStep[str, Dict[str, Any]]:
    """
    Create a step that summarizes text using the MaroviAPI custom summarization endpoint.
    
    Args:
        style: Summary style ("paragraph" or "bullet")
        max_length: Maximum length of the summary in words
        include_keywords: Whether to include keywords in the summary
        batch_size: Number of items to process in a batch
        step_id: Optional unique identifier for this step
        
    Returns:
        A configured MaroviAPIStep for summarization
    """
    endpoint_kwargs = {
        "style": style,
        "include_keywords": include_keywords
    }
    
    if max_length is not None:
        endpoint_kwargs["max_length"] = max_length
    
    return MaroviAPIStep(
        service="custom",
        endpoint="summarize",
        endpoint_kwargs=endpoint_kwargs,
        input_field="text",
        output_field="summary",
        batch_size=batch_size,
        step_id=step_id or f"summarize_{style}"
    )


def CompleteStep(temperature: float = 0.7, max_tokens: int = 150, 
                batch_size: int = 5, step_id: Optional[str] = None) -> MaroviAPIStep[str, str]:
    """
    Create a step that generates text completions using the MaroviAPI LLM service.
    
    Args:
        temperature: Sampling temperature for the LLM
        max_tokens: Maximum number of tokens to generate
        batch_size: Number of items to process in a batch
        step_id: Optional unique identifier for this step
        
    Returns:
        A configured MaroviAPIStep for LLM completion
    """
    return MaroviAPIStep(
        service="llm",
        endpoint="complete",
        endpoint_kwargs={
            "temperature": temperature,
            "max_tokens": max_tokens
        },
        input_field="prompt",
        batch_size=batch_size,
        step_id=step_id or "llm_completion"
    )


def ConvertFormatStep(source_format: str, target_format: str, 
                     preserve_structure: bool = True, preserve_links: bool = True,
                     batch_size: int = 10, step_id: Optional[str] = None) -> MaroviAPIStep[str, Dict[str, str]]:
    """
    Create a step that converts text between formats using the MaroviAPI custom format conversion endpoint.
    
    Args:
        source_format: Source format (e.g., "html", "markdown")
        target_format: Target format (e.g., "html", "markdown")
        preserve_structure: Whether to preserve document structure
        preserve_links: Whether to preserve links
        batch_size: Number of items to process in a batch
        step_id: Optional unique identifier for this step
        
    Returns:
        A configured MaroviAPIStep for format conversion
    """
    return MaroviAPIStep(
        service="custom",
        endpoint="convert_format",
        endpoint_kwargs={
            "source_format": source_format,
            "target_format": target_format,
            "preserve_structure": preserve_structure,
            "preserve_links": preserve_links
        },
        input_field="text",
        output_field="converted_text",
        batch_size=batch_size,
        step_id=step_id or f"convert_{source_format}_to_{target_format}"
    )


def CleanTextStep(format_value: str, remove_html_artifacts: bool = True,
                 preserve_structure: bool = True, preserve_links: bool = True,
                 preserve_images: bool = True, provider: str = "openai", 
                 temperature: float = 0.1, batch_size: int = 5,
                 step_id: Optional[str] = None) -> MaroviAPIStep[str, Dict[str, str]]:
    """
    Create a step that cleans text from artifacts and leftover markup using the MaroviAPI custom text cleaner.
    
    Args:
        format_value: Format of the text to clean (e.g., "wiki", "markdown", "html")
        remove_html_artifacts: Whether to remove HTML artifacts from the text
        preserve_structure: Whether to preserve document structure
        preserve_links: Whether to preserve links
        preserve_images: Whether to preserve image references
        provider: LLM provider to use (e.g., "openai", "anthropic")
        temperature: Sampling temperature for the LLM
        batch_size: Number of items to process in a batch
        step_id: Optional unique identifier for this step
        
    Returns:
        A configured MaroviAPIStep for text cleaning
    """
    return MaroviAPIStep(
        service="custom",
        endpoint="clean_text",
        endpoint_kwargs={
            "format": format_value,
            "remove_html_artifacts": remove_html_artifacts,
            "preserve_structure": preserve_structure,
            "preserve_links": preserve_links,
            "preserve_images": preserve_images,
            "provider": provider,
            "options": {
                "temperature": temperature,
                "structured_response": True
            }
        },
        input_field="text",
        output_field="cleaned_text",
        batch_size=batch_size,
        step_id=step_id or f"clean_{format_value}_text"
    )
