"""
Base classes for custom endpoints.

This module provides base classes for creating custom endpoints.
"""

import logging
from typing import Dict, Any, Optional, Type, List
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, Generic, TypeVar
from pydantic import BaseModel

# Type variables for request and response models
RequestT = TypeVar('RequestT', bound=BaseModel)
ResponseT = TypeVar('ResponseT', bound=BaseModel)

class CustomEndpoint(ABC, Generic[RequestT, ResponseT]):
    """Base class for all custom endpoints."""
    
    def __init__(self, request_model: Type[RequestT], response_model: Type[ResponseT]):
        """Initialize the endpoint with its request and response models."""
        self.request_model = request_model
        self.response_model = response_model
    
    @abstractmethod
    def process(self, request: RequestT) -> ResponseT:
        """Process a request and return a response.
        
        Args:
            request: The validated request model
            
        Returns:
            The validated response model
        """
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """Return a list of capabilities offered by this endpoint."""
        pass
    
    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Return metadata about this endpoint."""
        pass

class Pipeline(CustomEndpoint[RequestT, ResponseT]):
    """A pipeline of processing stages."""
    
    def __init__(self, name: str, request_model: Type[RequestT], response_model: Type[ResponseT], description: str = None):
        """Initialize the pipeline."""
        super().__init__(request_model, response_model)
        self.name = name
        self.description = description or f"Pipeline: {name}"
        self.stages: List[ProcessingStage] = []
        self.capabilities: List[str] = []
    
    def add_stage(self, processor: CustomEndpoint, config: Optional[Dict[str, Any]] = None, name: Optional[str] = None):
        """Add a processing stage to the pipeline."""
        stage = ProcessingStage(processor, config, name)
        self.stages.append(stage)
        
        # Add capabilities from the processor
        for capability in processor.get_capabilities():
            if capability not in self.capabilities:
                self.capabilities.append(capability)
                
        return self  # Enable method chaining
    
    def process(self, request: RequestT) -> ResponseT:
        """Process a request through the pipeline."""
        result = request
        context = {"pipeline_name": self.name, "stages": []}
        
        for i, stage in enumerate(self.stages):
            context["current_stage"] = i
            context["current_stage_name"] = stage.name
            
            # Execute the stage
            result = stage.process(result, context)
            
            # Update context
            context["stages"].append({
                "name": stage.name,
                "elapsed_seconds": stage.elapsed_seconds
            })
        
        return self.response_model(**result.dict())
    
    def get_capabilities(self) -> List[str]:
        return self.capabilities
    
    def get_metadata(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "stages": [stage.name for stage in self.stages],
            "request_model": self.request_model.__name__,
            "response_model": self.response_model.__name__
        }

class ProcessingStage:
    """A single stage in a processing pipeline."""
    
    def __init__(self, processor: CustomEndpoint, config: Optional[Dict[str, Any]] = None, name: Optional[str] = None):
        """Initialize the processing stage."""
        self.processor = processor
        self.config = config or {}
        self.name = name or getattr(processor, "__name__", "unnamed_stage")
        self.elapsed_seconds = 0.0
    
    def process(self, input_data: BaseModel, context: Optional[Dict[str, Any]] = None) -> BaseModel:
        """Process input data through this stage."""
        import time
        start_time = time.time()
        
        # Merge config with context
        merged_kwargs = {**self.config}
        if context:
            merged_kwargs["context"] = context
        
        # Process the data
        result = self.processor.process(input_data, **merged_kwargs)
        
        # Update elapsed time
        self.elapsed_seconds = time.time() - start_time
        
        return result 