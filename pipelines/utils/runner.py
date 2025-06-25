"""
Pipeline Runner

This module provides a PipelineRunner class that can dynamically load and run
pipeline implementations by name, facilitating both programmatic and CLI-based
pipeline execution.
"""

import os
import sys
import importlib
import logging
import time
from typing import Dict, Any, Optional, Type, List, Union

from pipelines.utils.base import BasePipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PipelineRunner:
    """
    A runner for dynamically loading and executing pipelines.
    
    This class provides functionality to discover, load, and run pipeline
    implementations by name, making it easy to execute pipelines both
    programmatically and from the command line.
    """
    
    def __init__(self, pipelines_package: str = "marovi.pipelines"):
        """
        Initialize the pipeline runner.
        
        Args:
            pipelines_package: Base package where pipeline implementations are located
        """
        self.pipelines_package = pipelines_package
        self.pipeline_registry: Dict[str, Type[BasePipeline]] = {}
        self.discover_pipelines()
    
    def discover_pipelines(self) -> None:
        """
        Discover available pipeline implementations.
        
        This method searches for pipeline implementations in the specified
        package and registers them by name.
        """
        # Get the actual package object
        try:
            package = importlib.import_module(self.pipelines_package)
        except ImportError:
            logger.error(f"Could not import pipelines package: {self.pipelines_package}")
            return
        
        # Get the package directory
        if not hasattr(package, "__path__"):
            logger.error(f"Package {self.pipelines_package} is not a directory package")
            return
        
        package_dir = package.__path__[0]
        
        # Find Python modules in the package
        for filename in os.listdir(package_dir):
            if not filename.endswith(".py") or filename.startswith("_"):
                continue
            
            module_name = filename[:-3]  # Remove .py extension
            full_module_name = f"{self.pipelines_package}.{module_name}"
            
            try:
                module = importlib.import_module(full_module_name)
                
                # Look for pipeline classes in the module
                for item_name in dir(module):
                    item = getattr(module, item_name)
                    
                    # Check if it's a pipeline class
                    if (isinstance(item, type) and 
                        issubclass(item, BasePipeline) and 
                        item is not BasePipeline):
                        
                        # Register the pipeline
                        pipeline_name = item_name.lower().replace("pipeline", "")
                        self.pipeline_registry[pipeline_name] = item
                        logger.debug(f"Registered pipeline: {pipeline_name}")
            
            except (ImportError, AttributeError) as e:
                logger.warning(f"Error loading module {full_module_name}: {e}")
    
    def get_pipeline_class(self, name: str) -> Optional[Type[BasePipeline]]:
        """
        Get a pipeline class by name.
        
        Args:
            name: Name of the pipeline
            
        Returns:
            The pipeline class if found, None otherwise
        """
        # Normalize the name
        normalized_name = name.lower().replace("pipeline", "").replace("_", "")
        
        # Try exact match first
        if normalized_name in self.pipeline_registry:
            return self.pipeline_registry[normalized_name]
        
        # Try partial match
        for key, pipeline_class in self.pipeline_registry.items():
            if normalized_name in key or key in normalized_name:
                return pipeline_class
        
        return None
    
    def create_pipeline(self, name: str, **kwargs) -> Optional[BasePipeline]:
        """
        Create a pipeline instance by name.
        
        Args:
            name: Name of the pipeline
            **kwargs: Additional arguments for the pipeline constructor
            
        Returns:
            The pipeline instance if found, None otherwise
        """
        pipeline_class = self.get_pipeline_class(name)
        if pipeline_class is None:
            return None
        
        return pipeline_class(**kwargs)
    
    def run_pipeline(self, 
                     pipeline_name: str, 
                     inputs: Any, 
                     **kwargs) -> Any:
        """
        Run a pipeline by name.
        
        Args:
            pipeline_name: Name of the pipeline to run
            inputs: Inputs for the pipeline
            **kwargs: Additional arguments for the pipeline run
            
        Returns:
            The pipeline outputs if successful, None otherwise
        """
        # Create the pipeline
        pipeline = self.create_pipeline(pipeline_name)
        if pipeline is None:
            logger.error(f"Pipeline not found: {pipeline_name}")
            return None
        
        # Run the pipeline
        start_time = time.time()
        try:
            outputs = pipeline.run(inputs, **kwargs)
            end_time = time.time()
            logger.info(f"Pipeline {pipeline_name} completed in {end_time - start_time:.2f} seconds")
            
            # Print metrics
            pipeline.print_metrics()
            
            return outputs
            
        except Exception as e:
            end_time = time.time()
            logger.error(f"Pipeline {pipeline_name} failed after {end_time - start_time:.2f} seconds")
            logger.exception(e)
            return None
    
    def list_pipelines(self) -> List[str]:
        """
        Get a list of available pipeline names.
        
        Returns:
            List of pipeline names
        """
        return list(self.pipeline_registry.keys())
    
    def get_pipeline_info(self) -> Dict[str, Dict[str, str]]:
        """
        Get information about all available pipelines.
        
        Returns:
            Dictionary mapping pipeline names to their info
        """
        info = {}
        for name, pipeline_class in self.pipeline_registry.items():
            # Create a temporary instance to get the description
            pipeline = pipeline_class()
            info[name] = {
                "name": pipeline.name,
                "description": pipeline.description
            }
        
        return info
