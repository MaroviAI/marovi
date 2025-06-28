#!/usr/bin/env python
"""
PO Translation Pipeline Runner

This script demonstrates how to run the PO Translation Pipeline directly from code,
rather than through the CLI interface. This is useful for integrating the pipeline
into larger workflows or for scripting translation jobs.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

from marovi.pipelines.po_translation import POTranslationPipeline, translate_po_files

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def example_direct_usage(
    input_file: str = "data/wiki_pages/marovi/Welcome/page-Welcome_en.po",
    output_dir: str = "data/translations",
    target_lang: str = "es",
) -> None:
    """
    Example showing direct usage of the POTranslationPipeline class.

    Args:
        input_file: Path to a PO file to translate
        output_dir: Directory to save translated files
        target_lang: Target language code
    """
    logger.info("=== Starting direct pipeline usage example ===")

    # Validate input file exists
    if not os.path.exists(input_file):
        logger.error(f"Input file not found: {input_file}")
        return

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Method 1: Using the convenience function
        logger.info("Method 1: Using the translate_po_files convenience function")
        es_result = translate_po_files(
            po_files=input_file,
            target_lang=target_lang,  # Spanish
            output_dir=output_dir,
        )
        logger.info(
            f"Translated {len(es_result['translated_files'])} file(s) to {target_lang}"
        )

        # Method 2: Creating and running the pipeline directly
        logger.info("Method 2: Creating and running the pipeline directly")
        pipeline = POTranslationPipeline()
        fr_result = pipeline.run(
            inputs=input_file,
            source_lang="en",
            target_lang="fr",  # French
            output_dir=output_dir,
        )
        logger.info(f"Translated {len(fr_result['translated_files'])} file(s) to fr")

    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        import traceback

        traceback.print_exc()


def example_runner_usage(
    input_file: str = "data/wiki_pages/marovi/Welcome/page-Welcome_en.po",
    output_dir: str = "data/translations",
) -> None:
    """
    Example showing usage of the PipelineRunner with the PO translation pipeline.

    Args:
        input_file: Path to a PO file to translate
        output_dir: Directory to save translated files
    """
    logger.info("=== Starting runner-based pipeline usage example ===")

    # Validate input file exists
    if not os.path.exists(input_file):
        logger.error(f"Input file not found: {input_file}")
        return

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Import the PipelineRunner
        from marovi.pipelines.utils.runner import PipelineRunner

        # Create a runner instance
        runner = PipelineRunner()

        # List available pipelines
        logger.info("Available pipelines:")
        pipelines = runner.list_pipelines()
        for pipeline_name in pipelines:
            logger.info(f"  - {pipeline_name}")

        # Get pipeline info
        pipeline_info = runner.get_pipeline_info()
        logger.info("Pipeline descriptions:")
        for name, info in pipeline_info.items():
            logger.info(f"  - {name}: {info['description']}")

        # Run the PO translation pipeline
        result = runner.run_pipeline(
            pipeline_name="po_translation",
            inputs=input_file,
            target_lang="de",  # German
            output_dir=output_dir,
        )

        if result:
            logger.info(
                f"Runner-based pipeline translated {len(result['translated_files'])} file(s)"
            )
        else:
            logger.error("Runner-based pipeline execution failed")

    except Exception as e:
        logger.error(f"Runner-based pipeline execution failed: {str(e)}")
        import traceback

        traceback.print_exc()


def main() -> None:
    """
    Main function to run the examples.
    """
    parser = argparse.ArgumentParser(description="Run PO Translation Pipeline examples")
    parser.add_argument(
        "--input",
        type=str,
        default="data/wiki_pages/marovi/Welcome/page-Welcome_en.po",
        help="Path to a PO file to translate",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/translations",
        help="Directory to save translated files",
    )

    args = parser.parse_args()

    # Run both examples
    example_direct_usage(args.input, args.output_dir)
    example_runner_usage(args.input, args.output_dir)


if __name__ == "__main__":
    main()
