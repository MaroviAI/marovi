#!/usr/bin/env python3
"""
Marovi API - Translation Services Example

This script demonstrates how to use the Marovi API to access various 
translation services (Google Translate REST API, DeepL, and ChatGPT).
"""

import os
import asyncio
from dotenv import load_dotenv
from marovi.api import Router, ServiceType

# Load environment variables from .env file
load_dotenv()

async def main():
    # Create a router instance
    router = Router()
    
    print("Setting up translation clients...\n")
    
    # Register Google Translate client using REST API
    google_client_id = router.register_translation_client(
        provider="google_rest",
        api_key=os.getenv("GOOGLE_TRANSLATE_API_KEY"),
        client_id="google-translate"
    )
    
    # Register DeepL client
    deepl_client_id = router.register_translation_client(
        provider="deepl",
        api_key=os.getenv("DEEPL_API_KEY"),
        client_id="deepl-translate"
    )
    
    # Register ChatGPT for translation (using OpenAI client)
    openai_client_id = router.register_llm_client(
        provider="openai",
        model="gpt-4",  # You can also use gpt-3.5-turbo for more cost-effective translations
        api_key=os.getenv("OPENAI_API_KEY"),
        client_id="openai-translation"
    )
    
    # Get translation clients
    google_client = router.get_translation_client(google_client_id)
    deepl_client = router.get_translation_client(deepl_client_id)
    openai_client = router.get_llm_client(openai_client_id)
    
    # Define a helper function for ChatGPT translation
    def chatgpt_translate(text, source_lang, target_lang):
        lang_names = {
            "es": "Spanish",
            "fr": "French",
            "de": "German",
            "ja": "Japanese",
            "en": "English"
        }
        
        prompt = f"Translate the following text from {lang_names.get(source_lang, source_lang)} to {lang_names.get(target_lang, target_lang)}. Only return the translated text, nothing else.\n\nText: {text}"
        
        response = openai_client.complete(
            prompt=prompt,
            temperature=0.3,
            max_tokens=1000
        )
        
        return response.strip() if isinstance(response, str) else response
    
    # Test single translations
    print("\n--- Testing Single Translations ---\n")
    text = "The artificial intelligence revolution is transforming the world as we know it."
    source_lang = "en"
    target_langs = ["es", "fr", "de", "ja"]  # Spanish, French, German, Japanese
    
    print(f"Original text: {text}\n")
    
    # Google Translate
    print("Google Translate results:")
    for target_lang in target_langs:
        translation = google_client.translate(
            text=text,
            source_lang=source_lang,
            target_lang=target_lang
        )
        print(f"  {target_lang}: {translation}")
    
    # DeepL
    print("\nDeepL results:")
    for target_lang in target_langs:
        # Skip Japanese if not supported by your DeepL plan
        if target_lang == "ja" and target_lang not in deepl_client.get_supported_languages():
            print(f"  {target_lang}: [Not supported by your DeepL plan]")
            continue
            
        translation = deepl_client.translate(
            text=text,
            source_lang=source_lang,
            target_lang=target_lang
        )
        print(f"  {target_lang}: {translation}")
    
    # ChatGPT Translation
    print("\nChatGPT results:")
    for target_lang in target_langs:
        translated_text = chatgpt_translate(text, source_lang, target_lang)
        print(f"  {target_lang}: {translated_text}")
    
    # Async translation
    print("\n--- Testing Async Translation ---\n")
    
    async def run_async_translations():
        text = "Artificial intelligence is capable of solving complex problems."
        source_lang = "en"
        target_lang = "es"  # Spanish
        
        print(f"Original text: {text}\n")
        
        # Async Google Translate
        google_translation = await google_client.atranslate(
            text=text,
            source_lang=source_lang,
            target_lang=target_lang
        )
        print(f"Async Google Translate: {google_translation}")
        
        # Async DeepL
        deepl_translation = await deepl_client.atranslate(
            text=text,
            source_lang=source_lang,
            target_lang=target_lang
        )
        print(f"Async DeepL: {deepl_translation}")
        
        # Async ChatGPT
        prompt = f"Translate the following text from English to Spanish. Only return the translated text, nothing else.\n\nText: {text}"
        openai_response = await openai_client.acomplete(
            prompt=prompt,
            temperature=0.3,
            max_tokens=1000
        )
        print(f"Async ChatGPT: {openai_response.strip() if isinstance(openai_response, str) else openai_response}")
    
    await run_async_translations()
    
    # Translation quality comparison
    print("\n--- Translation Quality Comparison ---\n")
    complex_text = "The nuanced implications of artificial general intelligence on society remain largely unexplored, particularly regarding ethical considerations and potential job displacement in various sectors."
    target_lang = "es"  # Spanish
    
    print(f"Original text: {complex_text}\n")
    
    # Google Translate
    google_translation = google_client.translate(
        text=complex_text,
        source_lang="en",
        target_lang=target_lang
    )
    print(f"Google Translate: {google_translation}\n")
    
    # DeepL
    deepl_translation = deepl_client.translate(
        text=complex_text,
        source_lang="en",
        target_lang=target_lang
    )
    print(f"DeepL: {deepl_translation}\n")
    
    # ChatGPT
    chatgpt_translation = chatgpt_translate(complex_text, "en", target_lang)
    print(f"ChatGPT: {chatgpt_translation}")
    
    # Test formality setting with DeepL
    print("\n--- Testing DeepL Formality Settings ---\n")
    simple_text = "How are you? I hope you're doing well."
    print(f"Original text: {simple_text}\n")
    
    try:
        # Import needed for direct request creation
        from marovi.api.schemas.translation import TranslationRequest
        
        # Formal translation - using direct provider access
        formal_request = TranslationRequest(
            text=simple_text,
            source_lang="en",
            target_lang="es",
            metadata={"formality": "more"}  # Setting formality to "more" for formal translation
        )
        formal_translation = deepl_client.provider.translate(formal_request).content
        print(f"DeepL (formal): {formal_translation}\n")
        
        # Informal translation
        informal_request = TranslationRequest(
            text=simple_text,
            source_lang="en",
            target_lang="es",
            metadata={"formality": "less"}  # Setting formality to "less" for informal translation
        )
        informal_translation = deepl_client.provider.translate(informal_request).content
        print(f"DeepL (informal): {informal_translation}")
    except Exception as e:
        print(f"Note: Formality settings test failed - this feature may require API access or configuration: {e}")
    
    print("\n--- Demo completed ---")

if __name__ == "__main__":
    asyncio.run(main()) 