"""
PO Translation Pipeline

This module provides a pipeline for translating PO (gettext) files used in MediaWiki
and other translation systems. It handles parsing PO files, translating their content, and
saving the translations back to PO files.
"""

import os
import logging
from typing import List, Optional, Dict, Any, Union

from marovi.pipelines.core import Pipeline
from marovi.pipelines.context import PipelineContext
from pipelines.utils.base import BasePipeline
from marovi.modules.steps.parsing import (
    POFileParserStep,
    POFileWriterStep,
    TranslationConnectorStep,
)
from marovi.modules.steps.marovi_api import TranslateStep, LLMTranslateStep
from marovi.api.core.client import MaroviAPI

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class POTranslationPipeline(BasePipeline):
    """
    Pipeline for translating PO files from one language to another.

    This pipeline handles the entire process of:
    1. Parsing PO files
    2. Translating their content
    3. Saving the translations back to PO files

    It supports both Google Translate and LLM-based translation.
    """

    def __init__(
        self,
        name: str = "po_translation_pipeline",
        description: str = "Translate PO files",
    ):
        """
        Initialize the PO translation pipeline.

        Args:
            name: Name of the pipeline
            description: Description of the pipeline's purpose
        """
        super().__init__(name=name, description=description)
        self.pipeline = None
        self.context = None

    def build_pipeline(
        self,
        source_lang: str = "en",
        target_lang: str = "es",
        output_dir: Optional[str] = None,
        translator_type: str = "google",
        batch_size: int = 5,
        **kwargs,
    ) -> Pipeline:
        """
        Build the pipeline with the given configuration.

        Args:
            source_lang: Source language code
            target_lang: Target language code
            output_dir: Directory to save output files (if None, uses same directory as input)
            translator_type: Type of translator to use ('google' or 'llm')
            batch_size: Batch size for processing
            **kwargs: Additional configuration parameters

        Returns:
            The constructed pipeline
        """
        # Create parser step
        parser_step = POFileParserStep(source_lang=source_lang, batch_size=batch_size)

        # Create appropriate translator step based on type
        if translator_type.lower() == "llm":
            logger.info(f"Using LLM translator for {source_lang} to {target_lang}")
            translator_step = LLMTranslateStep(
                source_lang=source_lang, target_lang=target_lang, batch_size=batch_size
            )
        else:  # Default to Google
            logger.info(f"Using Google translator for {source_lang} to {target_lang}")
            translator_step = TranslateStep(
                source_lang=source_lang, target_lang=target_lang, batch_size=batch_size
            )

        # Create connector step
        connector_step = TranslationConnectorStep(
            parser_metadata_key=f"{parser_step.step_id}_metadata",
            step_id="po_translation_connector",
        )

        # Create writer step
        writer_step = POFileWriterStep(output_dir=output_dir, batch_size=batch_size)

        # Create the pipeline
        steps = [parser_step, translator_step, connector_step, writer_step]
        pipeline = Pipeline(steps=steps, name=self.name)

        return pipeline

    def create_context(
        self,
        source_lang: str = "en",
        target_lang: str = "es",
        translator_type: str = "google",
        **kwargs,
    ) -> PipelineContext:
        """
        Create a pipeline context with the given configuration.

        Args:
            source_lang: Source language code
            target_lang: Target language code
            translator_type: Type of translator to use
            **kwargs: Additional configuration parameters

        Returns:
            The configured pipeline context
        """
        metadata = {
            "description": f"PO file translation from {source_lang} to {target_lang}",
            "source_language": source_lang,
            "target_language": target_lang,
            "translator_type": translator_type,
        }

        # Add any additional metadata from kwargs
        metadata.update({k: v for k, v in kwargs.items() if k not in metadata})

        return PipelineContext(metadata=metadata)

    def prepare_inputs(self, inputs: Union[str, List[str]], **kwargs) -> List[str]:
        """
        Prepare inputs for the pipeline.

        Args:
            inputs: File path, directory path, or list of file paths
            **kwargs: Additional configuration parameters

        Returns:
            List of PO file paths to process
        """
        recursive = kwargs.get("recursive", True)

        if isinstance(inputs, str):
            # Single input (file or directory)
            if os.path.isfile(inputs):
                if not inputs.endswith(".po"):
                    logger.warning(f"Input file is not a PO file: {inputs}")
                return [inputs]
            elif os.path.isdir(inputs):
                # Find PO files in the directory
                return self._find_po_files(inputs, recursive)
            else:
                logger.warning(f"Input path does not exist: {inputs}")
                return []
        elif isinstance(inputs, list):
            # List of inputs
            file_paths = []
            for input_path in inputs:
                if os.path.isfile(input_path):
                    if not input_path.endswith(".po"):
                        logger.warning(f"Input file is not a PO file: {input_path}")
                        continue
                    file_paths.append(input_path)
                elif os.path.isdir(input_path):
                    # Find PO files in the directory
                    file_paths.extend(self._find_po_files(input_path, recursive))
                else:
                    logger.warning(f"Input path does not exist: {input_path}")
            return file_paths
        else:
            logger.warning(f"Invalid input type: {type(inputs)}")
            return []

    def _find_po_files(self, directory: str, recursive: bool = True) -> List[str]:
        """
        Find all PO files in a directory.

        Args:
            directory: Directory to search in
            recursive: Whether to search recursively

        Returns:
            List of paths to PO files
        """
        po_files = []

        if recursive:
            for root, _, files in os.walk(directory):
                for file in files:
                    if file.endswith(".po"):
                        po_files.append(os.path.join(root, file))
        else:
            for file in os.listdir(directory):
                if file.endswith(".po"):
                    po_files.append(os.path.join(directory, file))

        return po_files

    def process_outputs(self, outputs: List[str], **kwargs) -> Dict[str, str]:
        """
        Process outputs from the pipeline.

        Args:
            outputs: List of output file paths
            **kwargs: Additional configuration parameters

        Returns:
            Dictionary mapping input files to their corresponding output files
        """
        # Build a mapping of input to output files
        input_output_map = {}

        if self.context:
            # Get file metadata from context
            file_metadata = self.context.metadata.get("po_file_parser_metadata", [])

            for metadata, output_path in zip(file_metadata, outputs):
                input_path = metadata.get("file_path", "unknown")
                input_output_map[input_path] = output_path

        return {"translated_files": outputs, "input_output_map": input_output_map}

    def run(self, inputs: Union[str, List[str]], **kwargs) -> Dict[str, Any]:
        """
        Run the pipeline on the given inputs.

        Args:
            inputs: File path, directory path, or list of file paths
            **kwargs: Configuration parameters:
                - source_lang: Source language code (default: "en")
                - target_lang: Target language code (default: "es")
                - output_dir: Output directory (default: None)
                - translator_type: Type of translator (default: "google")
                - batch_size: Batch size (default: 5)
                - recursive: Whether to search directories recursively (default: True)

        Returns:
            Dictionary with the results of the pipeline run
        """
        # Extract configuration from kwargs
        source_lang = kwargs.get("source_lang", "en")
        target_lang = kwargs.get("target_lang", "es")
        output_dir = kwargs.get("output_dir")
        translator_type = kwargs.get("translator_type", "google")
        batch_size = kwargs.get("batch_size", 5)

        # Build pipeline if needed
        if self.pipeline is None:
            self.pipeline = self.build_pipeline(
                source_lang=source_lang,
                target_lang=target_lang,
                output_dir=output_dir,
                translator_type=translator_type,
                batch_size=batch_size,
            )

        # Create context if needed
        if self.context is None:
            self.context = self.create_context(
                source_lang=source_lang,
                target_lang=target_lang,
                translator_type=translator_type,
            )

        # Prepare inputs
        prepared_inputs = self.prepare_inputs(inputs, **kwargs)

        if not prepared_inputs:
            logger.warning("No PO files found to process")
            return {"translated_files": [], "input_output_map": {}}

        # Log pipeline execution
        logger.info(
            f"Running PO translation pipeline with {len(prepared_inputs)} files"
        )
        logger.info(f"  Source language: {source_lang}")
        logger.info(f"  Target language: {target_lang}")
        logger.info(f"  Translator: {translator_type}")

        # Run pipeline
        try:
            raw_outputs = self.pipeline.run(prepared_inputs, self.context)

            # Process outputs
            processed_outputs = self.process_outputs(raw_outputs, **kwargs)

            # Log completion
            logger.info(
                f"Pipeline completed successfully: {len(raw_outputs)} files translated"
            )

            return processed_outputs

        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            raise

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get execution metrics from the pipeline run.

        Returns:
            Dictionary containing execution metrics
        """
        if self.context is None:
            return {}

        metrics = {}
        for key, value in self.context.metrics.items():
            metrics[key] = value

        return metrics

    def print_metrics(self) -> None:
        """
        Print execution metrics to the console.
        """
        metrics = self.get_metrics()
        if not metrics:
            logger.info("No metrics available")
            return

        logger.info(f"Pipeline '{self.name}' execution metrics:")
        for key, value in metrics.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  {key}: {value}")

    def get_cli_arguments(self) -> Dict[str, Dict[str, Any]]:
        """
        Get CLI argument definitions for this pipeline.

        Returns:
            Dictionary mapping argument names to their configurations
        """
        return {
            "input": {
                "help": "Input PO file or directory containing PO files",
                "type": str,
                "required": True,
            },
            "source_lang": {
                "help": "Source language code",
                "type": str,
                "default": "en",
            },
            "target_lang": {
                "help": "Target language code",
                "type": str,
                "default": "es",
            },
            "output_dir": {
                "help": "Output directory (default: same as input)",
                "type": str,
                "required": False,
            },
            "translator_type": {
                "help": "Type of translator to use",
                "type": str,
                "choices": ["google", "llm"],
                "default": "google",
            },
            "batch_size": {
                "help": "Batch size for processing",
                "type": int,
                "default": 5,
            },
            "recursive": {
                "help": "Search directories recursively",
                "action": "store_true",
            },
        }


def translate_po_files(
    po_files: Union[str, List[str]],
    source_lang: str = "en",
    target_lang: str = "es",
    output_dir: Optional[str] = None,
    translator_type: str = "google",
    batch_size: int = 5,
    recursive: bool = True,
) -> Dict[str, Any]:
    """
    Translate PO files from one language to another.

    This is a convenience function that creates and runs a POTranslationPipeline.

    Args:
        po_files: Path to PO file, directory, or list of file paths
        source_lang: Source language code
        target_lang: Target language code
        output_dir: Directory to save output files (if None, uses same directory as input)
        translator_type: Type of translator to use ('google' or 'llm')
        batch_size: Batch size for processing
        recursive: Whether to search directories recursively

    Returns:
        Dictionary with the results of the pipeline run
    """
    pipeline = POTranslationPipeline()

    return pipeline.run(
        inputs=po_files,
        source_lang=source_lang,
        target_lang=target_lang,
        output_dir=output_dir,
        translator_type=translator_type,
        batch_size=batch_size,
        recursive=recursive,
    )
