#!/usr/bin/env python
"""
Marovi Pipelines CLI

This module provides a command-line interface for running Marovi pipelines.
It allows users to list available pipelines, view their descriptions, and run
them with appropriate arguments.
"""

import os
import sys
import argparse
import logging
from typing import Dict, Any, List, Optional

from marovi.pipelines.utils.runner import PipelineRunner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_parser() -> argparse.ArgumentParser:
    """
    Set up the argument parser for the CLI.
    
    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="Marovi Pipelines CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available pipelines
  python -m marovi.pipelines.cli list
  
  # Show details about a specific pipeline
  python -m marovi.pipelines.cli info po_translation
  
  # Run a pipeline
  python -m marovi.pipelines.cli run po_translation --input path/to/file.po --target_lang es
"""
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    subparsers.required = True
    
    # List command
    list_parser = subparsers.add_parser("list", help="List available pipelines")
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Get information about a pipeline")
    info_parser.add_argument("pipeline", help="Name of the pipeline")
    
    # Run command
    run_parser = subparsers.add_parser("run", help="Run a pipeline")
    run_parser.add_argument("pipeline", help="Name of the pipeline to run")
    
    # The run command will have additional arguments added dynamically
    # when the pipeline is selected
    
    return parser


def add_pipeline_arguments(parser: argparse.ArgumentParser, pipeline_name: str) -> None:
    """
    Add pipeline-specific arguments to the parser.
    
    Args:
        parser: Argument parser to add arguments to
        pipeline_name: Name of the pipeline
    """
    runner = PipelineRunner()
    pipeline = runner.create_pipeline(pipeline_name)
    
    if pipeline is None:
        logger.error(f"Pipeline not found: {pipeline_name}")
        return
    
    for arg_name, arg_config in pipeline.get_cli_arguments().items():
        # Extract argument configuration
        help_text = arg_config.get("help", f"Argument {arg_name}")
        arg_type = arg_config.get("type", str)
        required = arg_config.get("required", False)
        choices = arg_config.get("choices", None)
        default = arg_config.get("default", None)
        action = arg_config.get("action", None)
        
        # Build argument
        args = []
        if len(arg_name) == 1:
            args.append(f"-{arg_name}")
        else:
            args.append(f"--{arg_name}")
        
        kwargs = {"help": help_text}
        
        if arg_type:
            kwargs["type"] = arg_type
        
        if required:
            kwargs["required"] = required
        
        if choices:
            kwargs["choices"] = choices
        
        if default is not None:
            kwargs["default"] = default
        
        if action:
            kwargs["action"] = action
        
        # Add the argument
        parser.add_argument(*args, **kwargs)


def list_pipelines() -> None:
    """
    List all available pipelines.
    """
    runner = PipelineRunner()
    pipeline_info = runner.get_pipeline_info()
    
    if not pipeline_info:
        print("No pipelines found")
        return
    
    print("Available pipelines:")
    for name, info in pipeline_info.items():
        print(f"  {name}: {info['description']}")


def show_pipeline_info(pipeline_name: str) -> None:
    """
    Show detailed information about a pipeline.
    
    Args:
        pipeline_name: Name of the pipeline
    """
    runner = PipelineRunner()
    pipeline = runner.create_pipeline(pipeline_name)
    
    if pipeline is None:
        print(f"Pipeline not found: {pipeline_name}")
        return
    
    print(f"Pipeline: {pipeline.name}")
    print(f"Description: {pipeline.description}")
    print("\nArguments:")
    
    for arg_name, arg_config in pipeline.get_cli_arguments().items():
        help_text = arg_config.get("help", "")
        required = arg_config.get("required", False)
        default = arg_config.get("default", None)
        
        # Format argument display
        arg_str = f"  --{arg_name}"
        if required:
            arg_str += " (required)"
        if default is not None:
            arg_str += f" [default: {default}]"
        
        print(arg_str)
        print(f"    {help_text}")


def run_pipeline(args: argparse.Namespace) -> None:
    """
    Run a pipeline with the given arguments.
    
    Args:
        args: Command-line arguments
    """
    pipeline_name = args.pipeline
    
    # Get all arguments as a dictionary
    arg_dict = vars(args)
    
    # Remove the command and pipeline arguments
    arg_dict.pop("command")
    arg_dict.pop("pipeline")
    
    # Get input files or directories
    inputs = arg_dict.pop("input", None)
    if inputs is None:
        logger.error("No input specified")
        return
    
    # Run the pipeline
    runner = PipelineRunner()
    result = runner.run_pipeline(pipeline_name, inputs, **arg_dict)
    
    if result is not None:
        logger.info(f"Pipeline {pipeline_name} completed successfully")
    else:
        logger.error(f"Pipeline {pipeline_name} failed")


def main() -> None:
    """
    Main entry point for the CLI.
    """
    parser = setup_parser()
    
    # Parse initial arguments to get the command and pipeline
    args, unknown = parser.parse_known_args()
    
    # If the command is 'run', add pipeline-specific arguments
    if args.command == "run":
        pipeline_name = args.pipeline
        
        # Create a new parser with pipeline-specific arguments
        run_parser = argparse.ArgumentParser(
            description=f"Run {pipeline_name} pipeline",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        
        # Add pipeline-specific arguments
        add_pipeline_arguments(run_parser, pipeline_name)
        
        # Parse again with the updated parser
        run_args = run_parser.parse_args(unknown)
        
        # Combine the arguments
        for key, value in vars(run_args).items():
            setattr(args, key, value)
    
    # Execute the appropriate command
    if args.command == "list":
        list_pipelines()
    elif args.command == "info":
        show_pipeline_info(args.pipeline)
    elif args.command == "run":
        run_pipeline(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.exception("An error occurred")
        sys.exit(1) 