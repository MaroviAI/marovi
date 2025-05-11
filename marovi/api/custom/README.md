# Marovi API Custom Endpoints

This directory contains the implementation of Marovi's custom endpoints system, which allows you to extend the Marovi API with specialized functionality beyond basic LLM and translation services.

## Architecture Overview

The custom endpoints system follows these design principles:

1. **Modularity**: Each endpoint is a self-contained module that can be registered and used independently.
2. **Standardization**: All endpoints follow a consistent interface pattern using Pydantic models for validation.
3. **Discoverability**: Endpoints are automatically registered and discoverable through the MaroviAPI client.
4. **Interoperability**: Custom endpoints can leverage other services like LLM and translation.

## Directory Structure

```
marovi/api/custom/
├── __init__.py           # Package initialization and endpoint registration
├── core/                 # Core functionality for custom endpoints
│   ├── base.py           # Base classes for custom endpoints
│   └── registry.py       # Registry system for endpoints
├── endpoints/            # Concrete endpoint implementations
│   ├── convert_format.py # Format conversion endpoint
│   ├── llm_translate.py  # LLM-based translation
│   └── summarize.py      # Text summarization
├── prompts/              # Templates for LLM prompts
│   ├── convert_format.jinja  # Format conversion prompt
│   ├── prompt_registry.yaml  # Registry of available prompts
│   └── summarize.jinja       # Summarization prompt
```

## Creating a New Custom Endpoint

To create a new custom endpoint, follow these steps:

### 1. Define Schemas

Create request and response Pydantic models `schemas/`:

```python
class MyEndpointRequest(BaseModel):
    text: str
    option1: str
    option2: Optional[int] = None

class MyEndpointResponse(BaseModel):
    result: str
    success: bool = True
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
```

### 2. Create the Endpoint Implementation

Create a new file in the `endpoints/` directory (e.g., `my_endpoint.py`):

```python
from ..core.base import CustomEndpoint
from ..schemas import MyEndpointRequest, MyEndpointResponse

class MyEndpoint(CustomEndpoint):
    def __init__(self, llm_client=None):
        self.request_model = MyEndpointRequest
        self.response_model = MyEndpointResponse
        self.llm_client = llm_client
        
    def _get_llm_client(self):
        """Get LLM client from router if not explicitly provided."""
        if self.llm_client:
            return self.llm_client
        
        from ...core.router import default_router
        from ...core.base import ServiceType
        return default_router.get_service(ServiceType.LLM)
    
    def process(self, request: MyEndpointRequest) -> MyEndpointResponse:
        """Process the request and return a response."""
        try:
            # Your implementation here
            result = "Processed: " + request.text
            
            return MyEndpointResponse(
                result=result,
                success=True,
                metadata={"processing_info": "some_value"}
            )
        except Exception as e:
            return MyEndpointResponse(
                result="",
                success=False,
                error=str(e)
            )
    
    def get_capabilities(self) -> List[str]:
        """Return a list of capabilities offered by this endpoint."""
        return ["capability1", "capability2"]
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return metadata about this endpoint."""
        return {
            "uses_llm": True,
            "supports_batch": False,
            "version": "1.0"
        }
```

### 3. Create Prompt Templates (if using LLM)

If your endpoint uses LLM, create a Jinja template in the `prompts/` directory:

1. Create a file `my_endpoint.jinja` with your template:
```jinja
Please process the following text based on option1 "{{ option1 }}" 
and option2 "{{ option2 }}":

{{ text }}

Your output should be formatted as follows:
...
```

2. Register your prompt in `prompt_registry.yaml`:
```yaml
templates:
  # ... existing templates ...
  my_endpoint:
    file: "my_endpoint.jinja"
    description: "Template for my custom endpoint"
    input_fields:
      schema: "my_endpoint.MyEndpointRequest"
    output_format: "text"  # or "json"
    output_fields:
      schema: "my_endpoint.MyEndpointResponse"
```

### 4. Register the Endpoint

Update the registration in `core/registry.py` to include your endpoint:

```python
def register_default_endpoints(registry=None):
    # ... existing code ...
    
    # Register MyEndpoint endpoint
    try:
        from ..endpoints.my_endpoint import MyEndpoint
        my_endpoint = MyEndpoint()
        registry.register_endpoint("my_endpoint", my_endpoint)
        logger.debug("Registered MyEndpoint endpoint")
    except (ImportError, Exception) as e:
        logger.warning(f"Failed to register MyEndpoint endpoint: {str(e)}")
```

### 5. Update Package Imports

Update the imports in `__init__.py` to include your new endpoint:

```python
# Import my endpoint
from .endpoints.my_endpoint import (
    MyEndpoint,
    MyEndpointError
)

__all__ = [
    # ... existing exports ...
    
    # My endpoint
    "MyEndpoint",
    "MyEndpointError",
]
```

## Using Custom Endpoints

Custom endpoints are accessible through the MaroviAPI client:

```python
from marovi.api.core.client import MaroviAPI
from marovi.api.custom.schemas import MyEndpointRequest

# Create an API client
client = MaroviAPI()

# Access the endpoint
my_endpoint = client.custom.my_endpoint

# Create a request and process it
request = MyEndpointRequest(text="Some text", option1="value", option2=42)
response = my_endpoint.process(request)

if response.success:
    print(f"Result: {response.result}")
else:
    print(f"Error: {response.error}")
```

## Best Practices

1. **Error Handling**: Always handle exceptions and return meaningful error messages.
2. **Validation**: Use Pydantic models for request and response validation.
3. **Testing**: Write unit tests for your endpoint using mock services.
4. **Documentation**: Provide clear documentation in docstrings and README files.
5. **Logging**: Use the logging module to log important events and errors.
6. **Batch Processing**: Consider implementing batch processing for efficiency.
7. **Async Support**: Consider adding async methods for non-blocking operations.

## Examples

See the existing endpoints for examples:
- `convert_format.py`: Format conversion between HTML, Markdown, etc.
- `summarize.py`: Text summarization with various styles
- `llm_translate.py`: Translation using LLM capabilities
