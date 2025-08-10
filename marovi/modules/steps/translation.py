"""
PO File Translation Steps

This module provides specialized pipeline steps for translating PO (gettext) files
using various translation services. These steps handle the extraction of translatable
text from PO files, translation to multiple languages, and saving the results.
"""

import os
import logging
from typing import Dict, List, Optional, Union, Any, Tuple

from marovi.pipelines.core import PipelineStep
from marovi.pipelines.context import PipelineContext
from marovi.modules.parsing.po_parser import POParser
from marovi.api.core.client import MaroviAPI

# Import translation steps
from marovi.modules.steps.marovi_api import TranslateStep, LLMTranslateStep

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class POFileTranslatorStep(PipelineStep[str, Dict[str, List[str]]]):
    """
    A pipeline step for translating PO file content to multiple target languages.

    This step takes a path to a PO file, extracts translatable content, and translates
    it to multiple target languages using the specified translation service.
    """

    def __init__(
        self,
        source_lang: str = "en",
        target_langs: List[str] = None,
        translator_type: str = "google",
        batch_size: int = 5,
        include_translated: bool = False,
        step_id: str = "po_file_translator",
    ):
        """
        Initialize the PO file translator step.

        Args:
            source_lang: Source language code
            target_langs: List of target language codes
            translator_type: Type of translator to use ('google' or 'llm')
            batch_size: Number of items to process in a batch
            include_translated: Whether to include already translated entries
            step_id: Unique identifier for this step
        """
        super().__init__(
            batch_size=batch_size,
            batch_handling="wrap",
            process_mode="sequential",
            step_id=step_id,
        )
        self.source_lang = source_lang
        self.target_langs = target_langs or [
            "es",
            "fr",
            "de",
        ]  # Default to common languages
        self.translator_type = translator_type
        self.include_translated = include_translated
        self.api_client = MaroviAPI()

        # Create translator functions for each target language
        self.translators = {}
        for lang in self.target_langs:
            if translator_type.lower() == "llm":
                self.translators[lang] = self._create_llm_translator(lang)
            else:
                self.translators[lang] = self._create_google_translator(lang)

    def _create_google_translator(self, target_lang: str):
        """Create a Google Translate function for the target language."""

        def translate_text(text: str) -> str:
            return self.api_client.translation.translate(
                text=text,
                source_lang=self.source_lang,
                target_lang=target_lang,
                provider="google",
            )

        return translate_text

    def _create_llm_translator(self, target_lang: str):
        """Create an LLM-based translate function for the target language."""

        def translate_text(text: str) -> str:
            response = self.api_client.custom.llm_translate(
                text=text,
                source_lang=self.source_lang,
                target_lang=target_lang,
                provider="openai",
            )
            return response.get("translated_text", text)

        return translate_text

    def process(
        self, inputs: List[str], context: PipelineContext
    ) -> List[Dict[str, List[str]]]:
        """
        Process a list of PO file paths and translate their content.

        Args:
            inputs: List of PO file paths
            context: Pipeline context

        Returns:
            List of dictionaries mapping target languages to translations
        """
        if not inputs:
            logger.warning(f"{self.step_id}: No inputs provided")
            return []

        results = []
        file_metadata = []

        for file_path in inputs:
            try:
                # Parse the PO file
                parser = POParser()
                parser.load_from_file(
                    file_path, include_translated=self.include_translated
                )

                # Extract messages to translate
                messages = parser.get_messages()
                if not messages:
                    logger.warning(f"No messages found in {file_path}")
                    continue

                texts_to_translate = [msg[0] for msg in messages]

                # Translate to each target language
                translations = {}
                for lang in self.target_langs:
                    logger.info(
                        f"Translating {len(texts_to_translate)} messages to {lang}"
                    )
                    translator = self.translators[lang]

                    # Translate each message
                    translated_texts = []
                    for text in texts_to_translate:
                        try:
                            translated = translator(text)
                            translated_texts.append(translated)
                        except Exception as e:
                            logger.error(f"Error translating to {lang}: {str(e)}")
                            translated_texts.append(text)  # Fall back to original text

                    translations[lang] = translated_texts

                # Create metadata for this file
                metadata = {
                    "file_path": file_path,
                    "source_lang": self.source_lang,
                    "target_langs": self.target_langs,
                    "message_count": len(messages),
                    "parser": parser,  # Store the parser instance for later use
                }
                file_metadata.append(metadata)

                results.append(translations)

            except Exception as e:
                logger.error(f"{self.step_id}: Error processing {file_path}: {str(e)}")

        # Store metadata in context
        context.add_metadata(f"{self.step_id}_metadata", file_metadata)

        return results


class POFileBatchTranslatorStep(PipelineStep[List[str], List[Dict[str, List[str]]]]):
    """
    A pipeline step for batch translating multiple PO files to multiple languages.

    This step efficiently processes multiple PO files by batching translation requests
    for better performance with translation APIs.
    """

    def __init__(
        self,
        source_lang: str = "en",
        target_langs: List[str] = None,
        translator_type: str = "google",
        batch_size: int = 10,
        include_translated: bool = False,
        step_id: str = "po_file_batch_translator",
    ):
        """
        Initialize the PO file batch translator step.

        Args:
            source_lang: Source language code
            target_langs: List of target language codes
            translator_type: Type of translator to use ('google' or 'llm')
            batch_size: Batch size for translation API calls
            include_translated: Whether to include already translated entries
            step_id: Unique identifier for this step
        """
        super().__init__(
            batch_size=batch_size,
            batch_handling="inherent",
            process_mode="sequential",
            step_id=step_id,
        )
        self.source_lang = source_lang
        self.target_langs = target_langs or ["es", "fr", "de"]
        self.translator_type = translator_type
        self.include_translated = include_translated
        self.api_client = MaroviAPI()

    def _batch_translate(self, texts: List[str], target_lang: str) -> List[str]:
        """
        Translate a batch of texts to the target language.

        Args:
            texts: List of texts to translate
            target_lang: Target language code

        Returns:
            List of translated texts
        """
        if not texts:
            return []

        try:
            if self.translator_type.lower() == "llm":
                # Use LLM translation (less efficient for batches but higher quality)
                translations = []
                for text in texts:
                    response = self.api_client.custom.llm_translate(
                        text=text,
                        source_lang=self.source_lang,
                        target_lang=target_lang,
                        provider="openai",
                    )
                    translations.append(response.get("translated_text", text))
                return translations
            else:
                # Use Google Translate's batch capability
                return self.api_client.translation.batch_translate(
                    texts=texts,
                    source_lang=self.source_lang,
                    target_lang=target_lang,
                    provider="google",
                )
        except Exception as e:
            logger.error(f"Batch translation error: {str(e)}")
            # Fall back to individual translation
            translations = []
            for text in texts:
                try:
                    if self.translator_type.lower() == "llm":
                        response = self.api_client.custom.llm_translate(
                            text=text,
                            source_lang=self.source_lang,
                            target_lang=target_lang,
                            provider="openai",
                        )
                        translations.append(response.get("translated_text", text))
                    else:
                        translation = self.api_client.translation.translate(
                            text=text,
                            source_lang=self.source_lang,
                            target_lang=target_lang,
                            provider="google",
                        )
                        translations.append(translation)
                except Exception as inner_e:
                    logger.error(f"Individual translation error: {str(inner_e)}")
                    translations.append(text)  # Fall back to original text
            return translations

    def process(
        self, inputs: List[List[str]], context: PipelineContext
    ) -> List[List[Dict[str, List[str]]]]:
        """
        Process batches of PO file paths and translate their content.

        Args:
            inputs: Batches of PO file paths
            context: Pipeline context

        Returns:
            Batches of dictionaries mapping target languages to translations
        """
        if not inputs:
            logger.warning(f"{self.step_id}: No inputs provided")
            return []

        # Flatten the input batches for easier processing
        flat_inputs = [item for batch in inputs for item in batch]

        all_texts = []
        all_parsers = {}
        file_metadata = []

        # First, parse all PO files and extract texts to translate
        for file_path in flat_inputs:
            try:
                # Parse the PO file
                parser = POParser()
                parser.load_from_file(
                    file_path, include_translated=self.include_translated
                )

                # Extract messages to translate
                messages = parser.get_messages()
                if not messages:
                    logger.warning(f"No messages found in {file_path}")
                    continue

                texts_to_translate = [msg[0] for msg in messages]

                # Store for later
                all_texts.append(texts_to_translate)
                all_parsers[file_path] = parser

                # Create metadata for this file
                metadata = {
                    "file_path": file_path,
                    "source_lang": self.source_lang,
                    "target_langs": self.target_langs,
                    "message_count": len(messages),
                    "parser": parser,  # Store the parser instance for later use
                }
                file_metadata.append(metadata)

            except Exception as e:
                logger.error(f"{self.step_id}: Error processing {file_path}: {str(e)}")

        # Store metadata in context
        context.add_metadata(f"{self.step_id}_metadata", file_metadata)

        if not all_texts:
            logger.warning(f"{self.step_id}: No texts found to translate")
            return []

        # Translate for each target language
        results = []
        for i, file_texts in enumerate(all_texts):
            file_path = flat_inputs[i] if i < len(flat_inputs) else "unknown"

            translations = {}
            for lang in self.target_langs:
                logger.info(
                    f"Translating {len(file_texts)} messages from {file_path} to {lang}"
                )

                # Batch translate all texts for this file
                translated_texts = self._batch_translate(file_texts, lang)

                # Ensure we have translations for all texts
                if len(translated_texts) < len(file_texts):
                    logger.warning(
                        f"Missing translations: expected {len(file_texts)}, got {len(translated_texts)}"
                    )
                    # Fill in missing translations with original text
                    translated_texts.extend(file_texts[len(translated_texts) :])

                translations[lang] = translated_texts

            results.append(translations)

        # Re-batch the results to match the input structure
        batched_results = []
        start_idx = 0
        for batch in inputs:
            batch_size = len(batch)
            end_idx = start_idx + batch_size
            batched_results.append(results[start_idx:end_idx])
            start_idx = end_idx

        return batched_results


class POFileWriterWithMultiLanguageSupport(
    PipelineStep[Dict[str, List[str]], Dict[str, List[str]]]
):
    """
    A pipeline step for writing translated PO files for multiple target languages.

    This step takes dictionaries of translations (one per target language) and writes
    them to PO files, creating separate files for each target language.
    """

    def __init__(
        self,
        output_dir: Optional[str] = None,
        batch_size: int = 5,
        step_id: str = "po_file_multi_writer",
    ):
        """
        Initialize the multi-language PO file writer step.

        Args:
            output_dir: Directory to save output files (if None, uses same directory as input)
            batch_size: Number of files to process in a batch
            step_id: Unique identifier for this step
        """
        super().__init__(
            batch_size=batch_size,
            batch_handling="wrap",
            process_mode="sequential",
            step_id=step_id,
        )
        self.output_dir = output_dir

    def _get_output_path(self, input_path: str, target_lang: str) -> str:
        """
        Determine the output path for a translated PO file.

        Args:
            input_path: Path to the input file
            target_lang: Target language code

        Returns:
            Path to the output file
        """
        # If no output directory specified, use same directory as input
        output_dir = self.output_dir or os.path.dirname(input_path)

        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Extract the input filename
        basename = os.path.basename(input_path)

        # Insert the target language into the filename
        # Assume filename format: something_XX.po or page-Something_XX.po
        if "_" in basename:
            parts = basename.split("_")
            # Replace or add the language code
            if len(parts) > 1 and len(parts[-1]) >= 2:
                # Replace the language code
                new_basename = "_".join(parts[:-1]) + f"_{target_lang}.po"
            else:
                # Add the language code
                new_basename = basename.replace(".po", f"_{target_lang}.po")
        else:
            # No language code in the filename, just add it
            new_basename = basename.replace(".po", f"_{target_lang}.po")

        return os.path.join(output_dir, new_basename)

    def process(
        self, inputs: List[Dict[str, List[str]]], context: PipelineContext
    ) -> List[Dict[str, List[str]]]:
        """
        Process a list of translation dictionaries and write them to PO files.

        Args:
            inputs: List of dictionaries mapping target languages to translations
            context: Pipeline context

        Returns:
            List of dictionaries mapping target languages to output file paths
        """
        if not inputs:
            logger.warning(f"{self.step_id}: No inputs provided")
            return []

        # Get metadata from context
        metadata_key = None
        for key in context.metadata:
            if key.endswith("_metadata") and key != f"{self.step_id}_metadata":
                metadata_key = key
                break

        if not metadata_key:
            logger.error(f"{self.step_id}: No metadata found in context")
            return []

        file_metadata = context.get_metadata(metadata_key)
        if len(inputs) != len(file_metadata):
            logger.error(
                f"{self.step_id}: Metadata count ({len(file_metadata)}) does not match input count ({len(inputs)})"
            )
            return []

        results = []

        for translations_dict, metadata in zip(inputs, file_metadata):
            input_path = metadata.get("file_path", "unknown")
            parser = metadata.get("parser")

            if not parser:
                logger.error(
                    f"{self.step_id}: No parser found in metadata for {input_path}"
                )
                continue

            output_paths = {}

            for lang, translations in translations_dict.items():
                output_path = self._get_output_path(input_path, lang)

                try:
                    # Save translations to PO file
                    parser.save_translated_po(output_path, translations)
                    logger.info(
                        f"{self.step_id}: Saved {lang} translations to {output_path}"
                    )

                    output_paths[lang] = output_path

                except Exception as e:
                    logger.error(
                        f"{self.step_id}: Error saving {lang} translations to {output_path}: {str(e)}"
                    )
                    output_paths[lang] = None

            results.append(output_paths)

        return results
