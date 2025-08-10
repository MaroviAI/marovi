"""
Pipeline steps for parsing and processing various file formats.

This module provides pipeline steps for parsing different file formats,
extracting content for processing, and saving processed content back to files.
"""

import os
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any, Union, TypeVar, Generic

from marovi.pipelines.core import PipelineStep
from marovi.pipelines.context import PipelineContext
from marovi.modules.parsing.po_parser import POParser

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Type variables for parser steps
InputPathType = TypeVar("InputPathType")
ParsedType = TypeVar("ParsedType")
OutputType = TypeVar("OutputType")

__all__ = [
    "ParserStep",
    "WriterStep",
    "TranslationConnectorStep",
    "POFileParserStep",
    "POFileWriterStep",
]


class ParserStep(
    PipelineStep[InputPathType, ParsedType], Generic[InputPathType, ParsedType], ABC
):
    """
    A generic pipeline step for parsing files and extracting their content.

    This base class defines the interface for parser steps and provides
    common functionality like storing metadata in the pipeline context.
    """

    def __init__(
        self,
        batch_size: int = 5,
        process_mode: str = "sequential",
        step_id: str = "parser",
    ):
        """
        Initialize the generic parser step.

        Args:
            batch_size: Number of files to process in a batch
            process_mode: Processing mode ('sequential' or 'parallel')
            step_id: Unique identifier for this step
        """
        super().__init__(
            batch_size=batch_size,
            batch_handling="wrap",
            process_mode=process_mode,
            step_id=step_id,
        )

    @abstractmethod
    def parse_file(self, file_path: str) -> Tuple[ParsedType, Dict[str, Any]]:
        """
        Parse a single file and extract its content.

        Args:
            file_path: Path to the file to parse

        Returns:
            Tuple containing (parsed_content, metadata)
        """
        pass

    def process(
        self, inputs: List[InputPathType], context: PipelineContext
    ) -> List[ParsedType]:
        """
        Process a list of file paths and extract content.

        Args:
            inputs: List of file paths or input sources
            context: Pipeline context

        Returns:
            List of parsed content
        """
        if not inputs:
            logger.warning(f"{self.step_id}: No inputs provided")
            return []

        results = []
        metadata_list = []

        for input_source in inputs:
            try:
                # Get file path if input is not a string
                file_path = (
                    str(input_source)
                    if isinstance(input_source, (str, os.PathLike))
                    else input_source
                )

                logger.info(f"{self.step_id}: Processing input {file_path}")

                # Parse the file
                parsed_content, metadata = self.parse_file(file_path)

                # Store results and metadata
                results.append(parsed_content)
                metadata_list.append(metadata)

            except Exception as e:
                logger.error(
                    f"{self.step_id}: Error processing {input_source}: {str(e)}"
                )

        # Store metadata in context
        context.add_metadata(f"{self.step_id}_metadata", metadata_list)

        return results


class WriterStep(
    PipelineStep[InputPathType, OutputType], Generic[InputPathType, OutputType], ABC
):
    """
    A generic pipeline step for writing processed content back to files.

    This base class defines the interface for writer steps and provides
    common functionality for determining output paths and writing files.
    """

    def __init__(
        self,
        output_dir: Optional[str] = None,
        batch_size: int = 5,
        process_mode: str = "sequential",
        step_id: str = "writer",
    ):
        """
        Initialize the generic writer step.

        Args:
            output_dir: Directory to save output files (if None, uses same directory as input)
            batch_size: Number of files to process in a batch
            process_mode: Processing mode ('sequential' or 'parallel')
            step_id: Unique identifier for this step
        """
        super().__init__(
            batch_size=batch_size,
            batch_handling="wrap",
            process_mode=process_mode,
            step_id=step_id,
        )
        self.output_dir = output_dir

    @abstractmethod
    def write_file(self, content: Any, metadata: Dict[str, Any]) -> str:
        """
        Write content to a file.

        Args:
            content: Content to write
            metadata: Metadata about the file

        Returns:
            Path to the written file
        """
        pass

    def get_output_path(self, input_path: str) -> str:
        """
        Determine the output path for a processed file.

        Args:
            input_path: Path to the input file

        Returns:
            Path to the output file
        """
        if not self.output_dir:
            return input_path

        filename = os.path.basename(input_path)
        return os.path.join(self.output_dir, filename)

    def process(
        self, inputs: List[InputPathType], context: PipelineContext
    ) -> List[OutputType]:
        """
        Process a list of items and write them to files.

        Args:
            inputs: List of items to write
            context: Pipeline context with metadata

        Returns:
            List of outputs (usually file paths)
        """
        if not inputs:
            logger.warning(f"{self.step_id}: No inputs provided")
            return []

        # Get most recent parser metadata from context
        metadata_key = None
        for key in context.metadata:
            if key.endswith("_metadata") and key != f"{self.step_id}_metadata":
                metadata_key = key
                break

        if not metadata_key:
            logger.error(f"{self.step_id}: No metadata found in context")
            return []

        metadata_list = context.metadata.get(metadata_key, [])
        if len(inputs) != len(metadata_list):
            logger.error(
                f"{self.step_id}: Number of inputs ({len(inputs)}) does not match metadata count ({len(metadata_list)})"
            )
            return []

        results = []

        for item, metadata in zip(inputs, metadata_list):
            try:
                # Write the file
                output_path = self.write_file(item, metadata)
                results.append(output_path)

            except Exception as e:
                logger.error(f"{self.step_id}: Error writing file: {str(e)}")

        return results


class TranslationConnectorStep(PipelineStep[List[str], Dict[str, Any]]):
    """
    A connector step that combines translated texts with original file metadata.

    This step is necessary to connect the translation step with the writer step,
    ensuring that translations are properly associated with their source files.
    """

    def __init__(
        self,
        parser_metadata_key: Optional[str] = None,
        step_id: str = "translation_connector",
    ):
        """
        Initialize the connector step.

        Args:
            parser_metadata_key: Key to use for retrieving parser metadata (if None, auto-detect)
            step_id: Unique identifier for this step
        """
        super().__init__(
            batch_handling="wrap", process_mode="sequential", step_id=step_id
        )
        self.parser_metadata_key = parser_metadata_key

    def process(
        self, inputs: List[List[str]], context: PipelineContext
    ) -> List[Dict[str, Any]]:
        """
        Combine translated texts with file metadata.

        Args:
            inputs: List of lists of translated texts
            context: Pipeline context with file metadata

        Returns:
            List of dictionaries with file metadata and translations
        """
        # Determine metadata key if not provided
        metadata_key = self.parser_metadata_key
        if not metadata_key:
            for key in context.metadata:
                if key.endswith("_metadata"):
                    metadata_key = key
                    break

        # Retrieve file metadata from context
        file_metadata = context.metadata.get(metadata_key, []) if metadata_key else None
        if not file_metadata:
            logger.error(f"{self.step_id}: No file metadata found in context")
            return []

        if len(inputs) != len(file_metadata):
            logger.error(
                f"{self.step_id}: Number of translation sets ({len(inputs)}) does not match metadata count ({len(file_metadata)})"
            )
            return []

        # Combine translations with metadata
        results = []
        for translations, metadata in zip(inputs, file_metadata):
            result = metadata.copy()
            result["translations"] = translations
            results.append(result)

        return results


class POFileParserStep(ParserStep[str, List[str]]):
    """
    A pipeline step for parsing PO files and extracting their content for translation.

    This step takes file paths as input and returns lists of messages to translate.
    It also stores file metadata in the pipeline context for later retrieval.
    """

    def __init__(
        self,
        source_lang: str = "en",
        include_translated: bool = False,
        batch_size: int = 5,
        step_id: str = "po_file_parser",
    ):
        """
        Initialize the PO file parser step.

        Args:
            source_lang: Source language code (e.g., 'en')
            include_translated: Whether to include already translated entries
            batch_size: Number of files to process in a batch
            step_id: Unique identifier for this step
        """
        super().__init__(
            batch_size=batch_size, process_mode="sequential", step_id=step_id
        )
        self.source_lang = source_lang
        self.include_translated = include_translated

    def parse_file(self, file_path: str) -> Tuple[List[str], Dict[str, Any]]:
        """
        Parse a PO file and extract its content.

        Args:
            file_path: Path to the PO file

        Returns:
            Tuple containing (texts_to_translate, metadata)
        """
        # Validate file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File does not exist: {file_path}")

        # Parse the PO file
        parser = POParser()
        parser.load_from_file(file_path, include_translated=self.include_translated)

        # Extract messages
        messages = parser.get_messages()

        # Extract target language from filename or path
        target_lang = self._extract_target_lang(file_path)

        # Create default metadata even if there are no messages
        metadata = {
            "file_path": file_path,
            "source_lang": self.source_lang,
            "target_lang": target_lang,
            "message_count": len(messages),
            "parser": parser,  # Store the parser instance for later use
        }

        # If no messages found, return empty list but still include metadata
        if not messages:
            logger.warning(f"No messages found in {file_path}")
            return [], metadata

        # Extract texts to translate
        texts_to_translate = [msg[0] for msg in messages]

        return texts_to_translate, metadata

    def _extract_target_lang(self, file_path: str) -> str:
        """
        Extract target language from file path.

        Args:
            file_path: Path to the PO file

        Returns:
            Language code extracted from the filename or 'unknown'
        """
        # Try to extract language code from filename (e.g., page-Welcome_es.po)
        filename = os.path.basename(file_path)
        if "_" in filename:
            lang_code = filename.split("_")[-1].split(".")[0]
            if len(lang_code) == 2:  # Basic validation
                return lang_code

        return "unknown"


class POFileWriterStep(WriterStep[Dict[str, Any], str]):
    """
    A pipeline step for writing translated content back to PO files.

    This step takes dictionaries with parsed content and translations,
    then saves the translations back to PO files.
    """

    def __init__(
        self,
        output_dir: Optional[str] = None,
        batch_size: int = 5,
        step_id: str = "po_file_writer",
    ):
        """
        Initialize the PO file writer step.

        Args:
            output_dir: Directory to save output files (if None, uses same directory as input)
            batch_size: Number of files to process in a batch
            step_id: Unique identifier for this step
        """
        super().__init__(
            output_dir=output_dir,
            batch_size=batch_size,
            process_mode="sequential",
            step_id=step_id,
        )

    def write_file(self, content: Dict[str, Any], metadata: Dict[str, Any]) -> str:
        """
        Write translations to a PO file.

        Args:
            content: Dictionary with translations
            metadata: Metadata about the file

        Returns:
            Path to the saved PO file
        """
        file_path = metadata.get("file_path")
        parser = metadata.get("parser")

        if not file_path or not parser:
            raise ValueError("Missing required data for saving")

        # Get translations or use empty list if not available
        translations = content.get("translations", [])

        # Determine output path
        output_path = self.get_output_path(file_path)

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save translations to PO file
        parser.save_translated_po(output_path, translations)
        logger.info(f"{self.step_id}: Saved translations to {output_path}")

        return output_path
