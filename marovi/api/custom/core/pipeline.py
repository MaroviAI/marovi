"""
Pipeline architecture for complex custom endpoint workflows.

This module provides classes for building and executing pipelines of processing stages
to create complex workflows from simpler components.
"""

import time
from typing import List, Dict, Any, Callable, Optional, Union
from .base import CustomEndpoint

class ProcessingStage:
    """A single stage in a processing pipeline."""
    
    def __init__(self, processor: Callable, config: Optional[Dict[str, Any]] = None, name: Optional[str] = None):
        self.processor = processor
        self.config = config or {}
        self.name = name or getattr(processor, "__name__", "unnamed_stage")
    
    def execute(self, input_data, context=None, **kwargs):
        """Execute this processing stage."""
        merged_kwargs = {**self.config, **(kwargs or {})}
        context_dict = context or {}
        
        # Add context to kwargs if processor expects it
        if "context" in merged_kwargs:
            merged_kwargs["context"] = context_dict
            
        return self.processor(input_data, **merged_kwargs)

class Pipeline(CustomEndpoint):
    """A pipeline of processing stages."""
    
    def __init__(self, name: str, description: str = None):
        self.name = name
        self.description = description or f"Pipeline: {name}"
        self.stages: List[ProcessingStage] = []
        self.capabilities: List[str] = []
    
    def add_stage(self, processor, config=None, name=None):
        """Add a processing stage to the pipeline."""
        stage = ProcessingStage(processor, config, name)
        self.stages.append(stage)
        
        # Add capabilities from the processor if it's a CustomEndpoint
        if isinstance(processor, CustomEndpoint):
            for capability in processor.get_capabilities():
                if capability not in self.capabilities:
                    self.capabilities.append(capability)
                    
        return self  # Enable method chaining
    
    def execute(self, input_data, **kwargs):
        """Execute the entire pipeline."""
        result = input_data
        context = kwargs.pop("context", {}) or {}
        context["pipeline_name"] = self.name
        context["stages"] = []
        
        for i, stage in enumerate(self.stages):
            context["current_stage"] = i
            context["current_stage_name"] = stage.name
            
            # Execute the stage
            start_time = time.time()
            result = stage.execute(result, context, **kwargs)
            elapsed = time.time() - start_time
            
            # Update context
            context["stages"].append({
                "name": stage.name,
                "elapsed_seconds": elapsed
            })
            
        return result
    
    def get_capabilities(self) -> List[str]:
        return self.capabilities
    
    def get_metadata(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "stages": [stage.name for stage in self.stages]
        } 