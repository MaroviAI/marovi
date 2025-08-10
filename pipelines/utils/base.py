"""
Base Pipeline Definition

This module defines the BasePipeline abstract class that serves as the foundation
for all pipeline implementations in the Marovi framework. It ensures a consistent
interface across different pipeline types.
"""

import os
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BasePipeline(ABC):
    """
    Abstract base class for pipeline implementations.

    This class defines the interface that all pipeline implementations must follow,
    ensuring a consistent API for programmatic and CLI usage.
    """

    def __init__(self, name: str = "unnamed_pipeline", description: str = ""):
        """
        Initialize a pipeline with name and description.

        Args:
            name: Name of the pipeline
            description: Description of the pipeline's purpose
        """
        self.name = name
        self.description = description
        self.metrics = {}

    @abstractmethod
    def run(self, inputs: Any, **kwargs) -> Any:
        """
        Run the pipeline on the given inputs.

        Args:
            inputs: Input data for the pipeline
            **kwargs: Additional configuration parameters

        Returns:
            Results of the pipeline run
        """
        raise NotImplementedError

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get metrics from the last pipeline run.

        Returns:
            Dictionary of pipeline metrics
        """
        return self.metrics

    def print_metrics(self) -> None:
        """
        Print metrics from the last pipeline run.
        """
        if not self.metrics:
            logger.info("No metrics available.")
            return

        logger.info(f"Metrics for {self.name}:")
        for key, value in self.metrics.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  {key}: {value}")

    def get_cli_arguments(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the CLI arguments for this pipeline.

        Returns:
            Dictionary mapping argument names to their configurations
        """
        return {
            "input": {
                "help": "Input file or directory path",
                "type": str,
                "required": True,
            }
        }
