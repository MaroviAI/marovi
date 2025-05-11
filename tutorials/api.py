#!/usr/bin/env python3
"""
MaroviAPI Client Example

This example demonstrates all the available endpoints and features in the MaroviAPI client,
including:

1. Core services:
   - LLM client for text generation
   - Translation client for multilingual capabilities

2. Custom endpoints:
   - Format conversion between different text formats
   - Text summarization with various styles
   - Other available custom endpoints

The script demonstrates how to initialize the client, access different services,
and execute requests with proper error handling.
"""

import logging
import sys
import time
from typing import Dict, List, Any, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout,
)

# Create a logger for this script
logger = logging.getLogger('marovi_api_example')

try:
    # Import the MaroviAPI client
    from marovi.api.core.client import MaroviAPI
    
    # Import core service types
    from marovi.api.core.base import ServiceType
    
    # We'll only import these if custom endpoints are available
    summarization_imports_succeeded = False
    format_conversion_imports_succeeded = False
    
    try:
        # Try to import schemas for summarization
        from marovi.api.custom.schemas import SummarizationRequest, SummaryStyle
        summarization_imports_succeeded = True
    except ImportError as e:
        logger.warning(f"Could not import summarization schemas: {e}")
    
    try:
        # Try to import schemas for format conversion
        from marovi.api.custom.schemas import FormatConversionRequest, SupportedFormat
        format_conversion_imports_succeeded = True
    except ImportError as e:
        logger.warning(f"Could not import format conversion schemas: {e}")
    
    logger.info("Successfully imported Marovi components")
except ImportError as e:
    logger.error(f"Failed to import Marovi components: {str(e)}")
    logger.error("Please make sure the Marovi package is installed correctly.")
    sys.exit(1)

def display_section_header(title):
    """Display a formatted section header."""
    logger.info("\n" + "=" * 80)
    logger.info(f"  {title}")
    logger.info("=" * 80)

def display_result(title, content, metadata=None):
    """Display a formatted result."""
    logger.info("-" * 80)
    logger.info(f"{title}:")
    logger.info(content)
    if metadata:
        logger.info("\nMetadata:")
        for k, v in metadata.items():
            logger.info(f"  {k}: {v}")
    logger.info("-" * 80)

def list_all_endpoints():
    """List all available endpoints in the MaroviAPI client."""
    display_section_header("Available Endpoints in MaroviAPI")
    
    # Initialize the client
    client = MaroviAPI()
    
    # Get all custom endpoints
    try:
        custom_endpoints = dir(client.custom)
        # Filter out Python internal attributes and methods
        real_endpoints = [endpoint for endpoint in custom_endpoints 
                         if not endpoint.startswith('_') and 
                            endpoint not in ('get_client', 'registry', 'retry_config')]
        
        logger.info(f"Found {len(real_endpoints)} custom endpoints:")
        
        for i, endpoint in enumerate(real_endpoints, 1):
            # Try to get capabilities and metadata for each endpoint
            try:
                endpoint_client = getattr(client.custom, endpoint)
                capabilities = endpoint_client.get_capabilities()
                metadata = endpoint_client.get_metadata()
                
                logger.info(f"\n{i}. {endpoint}")
                logger.info(f"   Capabilities: {', '.join(capabilities)}")
                logger.info(f"   Metadata: {metadata}")
            except Exception as e:
                logger.info(f"\n{i}. {endpoint}")
                logger.info(f"   Error getting details: {str(e)}")
    except Exception as e:
        logger.error(f"Error listing custom endpoints: {str(e)}")
    
    # Core services
    logger.info("\nCore Services:")
    logger.info("1. LLM - Text generation service (client.llm)")
    logger.info("2. Translation - Cross-language translation service (client.translation)")
    
    # Access through dot notation
    logger.info("\nAll endpoints can be accessed via client.custom.<endpoint_name>")

def demonstrate_llm_client():
    """Demonstrate the LLM client capabilities."""
    display_section_header("LLM Client Example")
    
    # Initialize the client
    client = MaroviAPI()
    llm = client.llm
    
    # Example prompt
    prompt = "Explain what an API client is in 3 sentences."
    
    try:
        # Basic completion
        start_time = time.time()
        response = llm.complete(prompt, temperature=0.3, max_tokens=150)
        latency = time.time() - start_time
        
        display_result(
            "LLM Response",
            response,
            {
                "model": llm.model,
                "provider": llm.provider_type_str,
                "latency": f"{latency:.2f} seconds"
            }
        )
    except Exception as e:
        logger.error(f"Error with LLM client: {str(e)}")

def demonstrate_translation_client():
    """Demonstrate the Translation client capabilities."""
    display_section_header("Translation Client Example")
    
    # Initialize the client
    client = MaroviAPI()
    
    # Try to use the built-in translation client
    try:
        translator = client.translation
        logger.info("Using built-in translation client")
    except Exception as e:
        # If built-in client fails, use LLM-based translator
        logger.warning(f"Built-in translation client not available: {str(e)}")
        logger.info("Using LLM-based translation instead")
        
        # Check if custom endpoints are available
        try:
            translator = client.custom.llm_translate
        except Exception as e:
            logger.error(f"LLM-based translation not available: {str(e)}")
            return
    
    # Sample text to translate
    text = "The MaroviAPI provides a unified interface for accessing various AI services."
    
    try:
        # Translate from en to es
        start_time = time.time()
        
        if hasattr(translator, 'translate'):
            # LLM-based translation
            result = translator.translate(
                text=text,
                source_lang="en",
                target_lang="es"
            )
            
            if isinstance(result, dict):
                translated_text = result.get("translated_text", "")
            else:
                translated_text = result
        else:
            # Standard translation client
            translated_text = translator.translate(
                text=text,
                source_lang="en",
                target_lang="es"
            )
        
        latency = time.time() - start_time
        
        display_result(
            "Translation Result",
            translated_text,
            {
                "source": "en",
                "target": "es",
                "latency": f"{latency:.2f} seconds"
            }
        )
    except Exception as e:
        logger.error(f"Error with translation client: {str(e)}")

def demonstrate_format_conversion():
    """Demonstrate the Format Conversion endpoint."""
    display_section_header("Format Conversion Example")
    
    # Skip if format conversion imports failed
    if not format_conversion_imports_succeeded:
        logger.warning("Format conversion schemas not available. Skipping demonstration.")
        return
    
    # Initialize the client
    client = MaroviAPI()
    
    # Check if format conversion endpoint is available
    try:
        converter = client.custom.convert_format
    except Exception as e:
        logger.error(f"Format conversion endpoint not available: {str(e)}")
        return
    
    # Sample HTML content
    html_content = """
    <h1>Marovi API</h1>
    <p>The <strong>Marovi API</strong> provides access to:</p>
    <ul>
        <li>LLM services</li>
        <li>Translation services</li>
        <li>Custom endpoints for specialized tasks</li>
    </ul>
    <p>Learn more at <a href="https://example.com/marovi">our website</a>.</p>
    """
    
    try:
        # Convert HTML to Markdown
        start_time = time.time()
        request = FormatConversionRequest(
            text=html_content,
            source_format="html",
            target_format="markdown",
            preserve_structure=True,
            preserve_links=True
        )
        
        result = converter.process(request)
        latency = time.time() - start_time
        
        if hasattr(result, 'success') and result.success:
            display_result(
                "Format Conversion Result (HTML to Markdown)",
                result.converted_text,
                {
                    "source_format": result.source_format,
                    "target_format": result.target_format,
                    "latency": f"{latency:.2f} seconds"
                }
            )
        else:
            logger.error(f"Format conversion failed: {getattr(result, 'error', 'Unknown error')}")
    except Exception as e:
        logger.error(f"Error with format conversion endpoint: {str(e)}")

def demonstrate_summarization():
    """Demonstrate the Summarization endpoint."""
    display_section_header("Summarization Example")
    
    # Skip if summarization imports failed
    if not summarization_imports_succeeded:
        logger.warning("Summarization schemas not available. Skipping demonstration.")
        return
    
    # Initialize the client
    client = MaroviAPI()
    
    # Check if summarization endpoint is available
    try:
        summarizer = client.custom.summarize
    except Exception as e:
        logger.error(f"Summarization endpoint not available: {str(e)}")
        return
    
    # Sample text to summarize
    text = """
    The Marovi API is a comprehensive framework designed to provide unified access to various 
    AI services and capabilities. It includes core services like LLM (Large Language Model) 
    interaction and translation, as well as custom endpoints for specialized tasks such as 
    format conversion and text summarization.
    
    The API client is designed with a focus on usability, flexibility, and robustness. It 
    provides both synchronous and asynchronous interfaces, comprehensive error handling, 
    and detailed logging for observability. The modular architecture allows for easy 
    extension with new capabilities and services.
    
    Custom endpoints in Marovi API follow a standardized interface pattern using Pydantic 
    models for validation, ensuring consistency and type safety across the system. They can 
    leverage other services like LLM and translation, allowing for complex workflows that 
    combine multiple capabilities.
    
    The API supports multiple providers for its core services, allowing users to choose the 
    best solution for their specific needs. It also provides features like automatic retries 
    for transient failures, response caching for improved performance, and comprehensive 
    metadata tracking for analytics and debugging.
    """
    
    try:
        # Create a paragraph-style summary
        start_time = time.time()
        request = SummarizationRequest(
            text=text,
            style="paragraph",
            max_length=80
        )
        
        result = summarizer.process(request)
        latency = time.time() - start_time
        
        if hasattr(result, 'success') and result.success:
            display_result(
                "Paragraph Summary",
                result.summary,
                {
                    "original_words": result.word_count_original,
                    "summary_words": result.word_count_summary,
                    "reduction": f"{(1 - result.word_count_summary / result.word_count_original) * 100:.1f}%",
                    "latency": f"{latency:.2f} seconds"
                }
            )
            
            # Create a bullet-point summary with keywords
            start_time = time.time()
            request = SummarizationRequest(
                text=text,
                style="bullet",
                include_keywords=True
            )
            
            result = summarizer.process(request)
            latency = time.time() - start_time
            
            if hasattr(result, 'success') and result.success:
                display_result(
                    "Bullet Point Summary with Keywords",
                    result.summary,
                    {
                        "original_words": result.word_count_original,
                        "summary_words": result.word_count_summary,
                        "latency": f"{latency:.2f} seconds"
                    }
                )
            else:
                logger.error(f"Bullet point summarization failed: {getattr(result, 'error', 'Unknown error')}")
        else:
            logger.error(f"Paragraph summarization failed: {getattr(result, 'error', 'Unknown error')}")
    except Exception as e:
        logger.error(f"Error with summarization endpoint: {str(e)}")

def main():
    """Run the complete API examples demonstration."""
    logger.info("Starting MaroviAPI examples demonstration")
    
    try:
        # List all available endpoints
        list_all_endpoints()
        
        # Demonstrate core services
        demonstrate_llm_client()
        demonstrate_translation_client()
        
        # Demonstrate custom endpoints
        demonstrate_format_conversion()
        demonstrate_summarization()
        
        logger.info("\nMaroviAPI examples demonstration completed successfully")
    except Exception as e:
        logger.error(f"Error in MaroviAPI examples demonstration: {str(e)}")

if __name__ == "__main__":
    main()
