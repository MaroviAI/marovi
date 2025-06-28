#!/usr/bin/env python
"""
PO Translation Example

This script demonstrates how to translate PO files using both
Google Translate and LLM-based translation directly through the API.
"""

import os
import sys
import logging
import argparse
import tempfile
import re
from pathlib import Path

# Import Marovi components
from marovi.api.core.client import MaroviAPI
from marovi.modules.parsing.po_parser import POParser

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def chunk_markdown_text(text, max_chunk_size=800):
    """
    Split markdown text into manageable chunks for translation.

    Args:
        text: Markdown text to split
        max_chunk_size: Maximum chunk size in characters

    Returns:
        List of text chunks
    """
    if len(text) <= max_chunk_size:
        return [text]

    # Try to split at paragraphs first
    paragraphs = re.split(r"\n\n+", text)

    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        # If adding this paragraph would exceed the max size, start a new chunk
        if len(current_chunk) + len(paragraph) > max_chunk_size and current_chunk:
            chunks.append(current_chunk)
            current_chunk = paragraph
        else:
            if current_chunk:
                current_chunk += "\n\n" + paragraph
            else:
                current_chunk = paragraph

    # Add the last chunk if not empty
    if current_chunk:
        chunks.append(current_chunk)

    # If we still have chunks that are too large, split them at sentences
    final_chunks = []
    for chunk in chunks:
        if len(chunk) <= max_chunk_size:
            final_chunks.append(chunk)
        else:
            # Split at logical sentence boundaries
            sentences = re.split(r"([.!?]\s+)", chunk)
            current_chunk = ""

            # Join sentences back with their punctuation
            i = 0
            while i < len(sentences) - 1:
                sentence = (
                    sentences[i] + sentences[i + 1]
                    if i + 1 < len(sentences)
                    else sentences[i]
                )
                i += 2

                if (
                    len(current_chunk) + len(sentence) > max_chunk_size
                    and current_chunk
                ):
                    final_chunks.append(current_chunk)
                    current_chunk = sentence
                else:
                    current_chunk += sentence

            # Add the last chunk
            if current_chunk:
                final_chunks.append(current_chunk)

    return final_chunks


def translate_large_text(text, source_lang, target_lang, translator_type, api_client):
    """
    Translate large text by chunking it into smaller parts.

    Args:
        text: Text to translate
        source_lang: Source language code
        target_lang: Target language code
        translator_type: Type of translator ('google' or 'llm')
        api_client: MaroviAPI client instance

    Returns:
        Translated text
    """
    # If text is not that large, translate directly
    if len(text) < 1000:
        return translate_text(
            text, source_lang, target_lang, translator_type, api_client
        )

    # Chunk the text
    chunks = chunk_markdown_text(text)
    logger.info(f"Split large text into {len(chunks)} chunks for translation")

    # Translate each chunk
    translated_chunks = []
    for i, chunk in enumerate(chunks):
        logger.debug(f"Translating chunk {i+1}/{len(chunks)} ({len(chunk)} chars)")
        translated_chunk = translate_text(
            chunk, source_lang, target_lang, translator_type, api_client
        )
        translated_chunks.append(translated_chunk)

    # Join the translated chunks
    return "\n\n".join(translated_chunks)


def translate_text(text, source_lang, target_lang, translator_type, api_client):
    """
    Translate a single text using the specified translator.

    Args:
        text: Text to translate
        source_lang: Source language code
        target_lang: Target language code
        translator_type: Type of translator ('google' or 'llm')
        api_client: MaroviAPI client instance

    Returns:
        Translated text
    """
    try:
        if translator_type == "google":
            return api_client.translation.translate(
                text=text,
                source_lang=source_lang,
                target_lang=target_lang,
                provider="google",
            )
        else:  # LLM translation
            response = api_client.custom.llm_translate(
                text=text,
                source_lang=source_lang,
                target_lang=target_lang,
                provider="openai",
            )
            return response.get("translated_text", "")
    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        # Fall back to original text on error
        return text


def translate_po_file(
    input_path: str,
    output_path: str,
    source_lang: str = "en",
    target_lang: str = "es",
    translator_type: str = "google",
    include_translated: bool = False,
    api_client=None,
) -> bool:
    """
    Translate a PO file using either Google Translate or LLM.

    Args:
        input_path: Path to input PO file
        output_path: Path to save translated PO file
        source_lang: Source language code
        target_lang: Target language code
        translator_type: Type of translator ('google' or 'llm')
        include_translated: Whether to include already translated entries
        api_client: Optional MaroviAPI client instance

    Returns:
        True if successful, False otherwise
    """
    try:
        # Initialize API client if not provided
        if api_client is None:
            api_client = MaroviAPI()

        # Initialize parser
        parser = POParser()

        logger.info(f"Loading PO file: {input_path}")
        parser.load_from_file(input_path, include_translated=include_translated)

        # Get messages to translate
        messages = parser.get_messages()
        if not messages:
            logger.warning("No messages found in the PO file to translate.")
            # Save the file anyway as a copy
            parser.save_translated_po(output_path, [])
            return True

        logger.info(f"Found {len(messages)} messages to translate")

        # Extract source texts
        texts_to_translate = [msg[0] for msg in messages]

        # Set translator type
        translator_type = translator_type.lower()
        if translator_type not in ["google", "llm"]:
            logger.warning(f"Unknown translator type: {translator_type}. Using Google.")
            translator_type = "google"

        # Translate texts
        logger.info(
            f"Translating {len(texts_to_translate)} messages from {source_lang} to {target_lang} using {translator_type}"
        )
        translated_texts = []

        for i, text in enumerate(texts_to_translate):
            try:
                # Check if this is likely to be Markdown or Wiki content
                is_large_content = (
                    len(text) > 1000 or "[[" in text or "==" in text or "\n\n" in text
                )

                if is_large_content:
                    translated_text = translate_large_text(
                        text=text,
                        source_lang=source_lang,
                        target_lang=target_lang,
                        translator_type=translator_type,
                        api_client=api_client,
                    )
                else:
                    translated_text = translate_text(
                        text=text,
                        source_lang=source_lang,
                        target_lang=target_lang,
                        translator_type=translator_type,
                        api_client=api_client,
                    )

                translated_texts.append(translated_text)
                logger.info(f"Translated message {i+1}/{len(texts_to_translate)}")
            except Exception as e:
                logger.error(f"Error translating message {i+1}: {str(e)}")
                # Fall back to source text if translation fails
                translated_texts.append(text)

        # Save translated PO file
        logger.info(f"Saving translations to: {output_path}")
        parser.save_translated_po(output_path, translated_texts)

        # Display sample translations (up to 5)
        logger.info("Sample translations:")
        po = parser.po_file
        for i, entry in enumerate(po):
            if i < 5:  # Show only up to 5 translations
                logger.info(
                    f"  {entry.msgctxt or i}: '{entry.msgid[:50]}...' -> '{entry.msgstr[:50]}...'"
                )
            else:
                break

        # Update PO file headers for target language
        po.metadata["Language"] = target_lang
        po.metadata["X-Language-Code"] = target_lang

        logger.info(f"Translation completed successfully. Saved to {output_path}")
        return True

    except Exception as e:
        logger.error(f"Translation failed: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


def create_sample_po_file(output_path: str = None):
    """
    Create a sample PO file with content for demonstration purposes.

    Args:
        output_path: Path to save the sample file (if None, a temp file is created)

    Returns:
        Path to the created PO file
    """
    import polib

    # Create a POFile object
    po = polib.POFile()
    po.metadata = {
        "Project-Id-Version": "Sample PO File",
        "Language": "en",
        "MIME-Version": "1.0",
        "Content-Type": "text/plain; charset=utf-8",
        "Content-Transfer-Encoding": "8bit",
    }

    # Add sample entries
    entries = [
        polib.POEntry(msgid="Hello, world!", msgstr="", msgctxt="greeting"),
        polib.POEntry(
            msgid="Welcome to the Marovi AI translation system.",
            msgstr="",
            msgctxt="welcome",
        ),
        polib.POEntry(
            msgid="This text will be translated by an AI system.",
            msgstr="",
            msgctxt="explanation",
        ),
        polib.POEntry(
            msgid="Different translators may produce different results.",
            msgstr="",
            msgctxt="comparison",
        ),
        polib.POEntry(
            msgid="Please provide feedback on translation quality.",
            msgstr="",
            msgctxt="feedback",
        ),
    ]

    for entry in entries:
        po.append(entry)

    # Add a large markdown entry for chunking demonstration
    large_markdown = """
# Markdown Translation Example

This is a large markdown document that will be automatically chunked for translation.

## Introduction

Markdown is a lightweight markup language with plain text formatting syntax. Its design allows it to be converted to many output formats, but the original tool by the same name only supports HTML.

## Features

Markdown provides several ways to format text:

* **Bold text** and *italic text*
* Lists (like this one)
* [Links](https://example.com)
* Code blocks
* And more

## Code Example

```python
def hello_world():
    print("Hello, world!")
```

## Conclusion

Markdown is widely used for:
1. Documentation
2. Readme files
3. Forum posts
4. Wikis and knowledge bases
5. Note-taking

It's particularly useful for content that needs to be readable in plain text format but can also be converted to rich text when needed.
"""

    po.append(polib.POEntry(msgid=large_markdown, msgstr="", msgctxt="markdown_sample"))

    # Determine output path
    if output_path is None:
        temp_dir = tempfile.mkdtemp()
        output_path = os.path.join(temp_dir, "sample_en.po")

    # Save the file
    po.save(output_path)
    logger.info(f"Created sample PO file at {output_path}")

    return output_path


def main():
    """Run the PO translation example."""
    parser = argparse.ArgumentParser(description="PO Translation Example")
    parser.add_argument(
        "--input", help="Input PO file path (if not provided, creates a sample file)"
    )
    parser.add_argument("--output", help="Output directory for translated files")
    parser.add_argument(
        "--source", default="en", help="Source language code (default: en)"
    )
    parser.add_argument(
        "--target", default="es", help="Target language code (default: es)"
    )
    parser.add_argument(
        "--translator",
        default="google",
        choices=["google", "llm"],
        help="Translator type to use (default: google)",
    )
    parser.add_argument(
        "--include-translated",
        action="store_true",
        help="Include already translated entries (default: False)",
    )

    args = parser.parse_args()

    # Initialize API client (reused for multiple translations)
    api_client = MaroviAPI()

    # Handle input file
    input_path = args.input
    if not input_path:
        logger.info("No input file provided. Creating a sample PO file...")
        input_path = create_sample_po_file()
    elif not os.path.exists(input_path):
        logger.error(f"Input file not found: {input_path}")
        return 1

    # Determine output path
    if args.output:
        os.makedirs(args.output, exist_ok=True)
        output_filename = f"{Path(input_path).stem}_{args.target}.po"
        output_path = os.path.join(args.output, output_filename)
    else:
        output_dir = os.path.dirname(input_path)
        output_filename = f"{Path(input_path).stem}_{args.target}.po"
        output_path = os.path.join(output_dir, output_filename)

    # Translate the file
    success = translate_po_file(
        input_path=input_path,
        output_path=output_path,
        source_lang=args.source,
        target_lang=args.target,
        translator_type=args.translator,
        include_translated=args.include_translated,
        api_client=api_client,
    )

    if success:
        logger.info("Example completed successfully")
        return 0
    else:
        logger.error("Example failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
