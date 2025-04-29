#!/usr/bin/env python3
"""
LLM One-Shot Translator Test

This script demonstrates the capabilities of the LLM one-shot translator,
testing its various features and functionality with Spanish translations.
"""

import os
import asyncio
from dotenv import load_dotenv
from marovi.api import Router, ServiceType
from marovi.api.custom.llm_one_shot_translator import LLMOneShotTranslator
from marovi.api.schemas.translation import (
    TranslationRequest, TranslationResponse, TranslationBatchRequest,
    TranslationFormat, GlossaryEntry
)
from marovi.api.providers.openai import OpenAIProvider

# Load environment variables from .env file
load_dotenv()

async def main():
    print("Setting up LLM one-shot translator...\n")
    
    # Create OpenAI provider directly
    openai_provider = OpenAIProvider(api_key=os.getenv("OPENAI_API_KEY"))
    openai_provider.initialize()
    
    # Create LLM one-shot translator
    llm_translator = LLMOneShotTranslator(llm_provider=openai_provider)
    
    # Test basic translation
    print("\n--- Testing Basic Translation ---\n")
    text = "The artificial intelligence revolution is transforming the world as we know it."
    source_lang = "en"
    target_lang = "es"
    
    print(f"Original text: {text}\n")
    
    request = TranslationRequest(
        text=text,
        source_lang=source_lang,
        target_lang=target_lang,
        format=TranslationFormat.TEXT,
        preserve_formatting=True
    )
    response = llm_translator.translate(request)
    print(f"Translation to Spanish:")
    print(f"  Content: {response.content}")
    print(f"  Confidence: {response.confidence}")
    print(f"  Quality metrics: {response.quality_metrics}")
    print(f"  Detected language: {response.detected_lang}")
    print()
    
    # Test glossary support
    print("\n--- Testing Glossary Support ---\n")
    technical_text = "The API endpoint requires authentication using OAuth 2.0."
    glossary = [
        GlossaryEntry(
            source_term="API",
            target_term="API",
            context="technical",
            case_sensitive=True,
            exact_match=True
        ),
        GlossaryEntry(
            source_term="OAuth 2.0",
            target_term="OAuth 2.0",
            context="technical",
            case_sensitive=True,
            exact_match=True
        )
    ]
    
    print(f"Original text: {technical_text}\n")
    
    # Test with glossary
    request = TranslationRequest(
        text=technical_text,
        source_lang="en",
        target_lang="es",
        format=TranslationFormat.TEXT,
        preserve_formatting=True,
        glossary=glossary
    )
    response = llm_translator.translate(request)
    print(f"Translation with glossary:")
    print(f"  Content: {response.content}")
    print(f"  Glossary applied: {response.glossary_applied}")
    print(f"  Quality metrics: {response.quality_metrics}")
    print()
    
    # Test translation refinement
    print("\n--- Testing Translation Refinement ---\n")
    complex_text = "The nuanced implications of artificial general intelligence on society remain largely unexplored, particularly regarding ethical considerations and potential job displacement in various sectors."
    base_translation = "Las implicaciones matizadas de la inteligencia artificial general en la sociedad siguen siendo en gran parte inexploradas, particularmente con respecto a consideraciones éticas y potencial desplazamiento de empleos en varios sectores."
    
    print(f"Original text: {complex_text}\n")
    print(f"Base translation: {base_translation}\n")
    
    # Refine translation
    refined_translation = llm_translator.refine_translation(complex_text, base_translation)
    print(f"Refined translation: {refined_translation}")
    print()
    
    # Test batch translation
    print("\n--- Testing Batch Translation ---\n")
    texts = [
        "Hello, world!",
        "Artificial intelligence is transforming our lives.",
        "The future of technology is exciting."
    ]
    
    # Create batch request
    batch_request = TranslationBatchRequest(
        items=[
            TranslationRequest(
                text=text,
                source_lang="en",
                target_lang="es",
                format=TranslationFormat.TEXT,
                preserve_formatting=True
            )
            for text in texts
        ],
        max_concurrency=3,
        batch_size=2
    )
    
    # Process batch translation
    batch_response = llm_translator.batch_translate(batch_request)
    
    print("Batch translation results:")
    for i, response in enumerate(batch_response.items):
        print(f"\nText {i+1}:")
        print(f"  Original: {texts[i]}")
        print(f"  Translated: {response.content}")
        print(f"  Confidence: {response.confidence}")
        print(f"  Quality metrics: {response.quality_metrics}")
    
    print(f"\nBatch statistics:")
    print(f"  Total characters: {batch_response.total_characters}")
    print(f"  Average confidence: {batch_response.avg_confidence}")
    print(f"  Average latency: {batch_response.avg_latency:.2f}s")
    print(f"  Quality metrics: {batch_response.quality_metrics}")
    
    # Test async translation
    print("\n--- Testing Async Translation ---\n")
    
    async def run_async_translations():
        text = "Artificial intelligence is capable of solving complex problems."
        source_lang = "en"
        target_lang = "es"
        
        print(f"Original text: {text}\n")
        
        # Async translation
        request = TranslationRequest(
            text=text,
            source_lang=source_lang,
            target_lang=target_lang,
            format=TranslationFormat.TEXT,
            preserve_formatting=True
        )
        response = await llm_translator.atranslate(request)
        print(f"Async translation:")
        print(f"  Content: {response.content}")
        print(f"  Confidence: {response.confidence}")
        print(f"  Quality metrics: {response.quality_metrics}")
        print(f"  Latency: {response.latency:.2f}s")
    
    await run_async_translations()
    
    # Test language detection
    print("\n--- Testing Language Detection ---\n")
    texts_to_detect = [
        "Hello, how are you?",
        "Hola, ¿cómo estás?",
        "The weather is beautiful today.",
        "El clima está hermoso hoy."
    ]
    
    print("Language detection results:")
    for text in texts_to_detect:
        detected_lang = llm_translator.detect_language(text)
        print(f"  Text: {text}")
        print(f"  Detected language: {detected_lang}\n")
    
    # Test supported features
    print("\n--- Testing Supported Features ---\n")
    print("Supported languages:", llm_translator.get_supported_languages())
    print("Supported formats:", [f.value for f in llm_translator.get_supported_formats()])
    print("Supported domains:", llm_translator.get_supported_domains())
    print("Supported quality preferences:", llm_translator.get_supported_quality_preferences())
    print("Quality metrics:", llm_translator.get_quality_metrics())
    print("Rate limits:", llm_translator.get_rate_limits())
    
    print("\n--- Test completed ---")

if __name__ == "__main__":
    asyncio.run(main())
