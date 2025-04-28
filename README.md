# Marovi API

A state-of-the-art, enterprise-ready API client for LLM and translation services.

## Features

- Unified interface for multiple LLM providers (OpenAI, Anthropic)
- Translation services (Google Translate, DeepL, ChatGPT)
- Extensible architecture for custom endpoints
- Comprehensive logging and monitoring
- Automatic retry logic with exponential backoff
- Response caching
- Batch processing support
- Async and sync interfaces
- Type hints and validation with Pydantic

## Quick Start

```python
from marovi.api import Router, ServiceType

# Create a router instance
router = Router()

# Register an LLM client
router.register_llm_client(
    provider="openai",
    model="gpt-4",
    api_key="your-api-key"
)

# Register a translation client
router.register_translation_client(
    provider="google",
    api_key="your-api-key"
)

# Get the LLM client
llm_client = router.get_llm_client()

# Generate text
response = llm_client.complete(
    prompt="Write a poem about AI",
    temperature=0.7
)
print(response.content)

# Get the translation client
translation_client = router.get_translation_client()

# Translate text
translation = translation_client.translate(
    text="Hello, world!",
    source_lang="en",
    target_lang="es"
)
print(translation.content)
```

## Configuration

The package can be configured using environment variables or a `.env` file:

```env
# General settings
DEBUG=false
LOG_LEVEL=INFO

# OpenAI settings
OPENAI_API_KEY=your-api-key
OPENAI_DEFAULT_MODEL=gpt-4

# Anthropic settings
ANTHROPIC_API_KEY=your-api-key
ANTHROPIC_DEFAULT_MODEL=claude-3-sonnet-20240229

# Google Translate settings
GOOGLE_TRANSLATE_API_KEY=your-api-key

# DeepL settings
DEEPL_API_KEY=your-api-key

# Retry settings
DEFAULT_MAX_RETRIES=3
DEFAULT_INITIAL_BACKOFF=1.0
DEFAULT_MAX_BACKOFF=10.0
DEFAULT_BACKOFF_FACTOR=2.0

# Cache settings
DEFAULT_CACHE_SIZE=100
ENABLE_CACHE=false

# Concurrency settings
DEFAULT_MAX_CONCURRENCY=5
```

## Advanced Usage

### Provider Registry

The package includes a provider registry that manages all available LLM and translation providers. The registry is loaded from a YAML configuration file at `marovi/api/providers/registry.yaml`.

```python
from marovi.api import provider_registry

# List all available providers
providers = provider_registry.get_all_providers()

# List providers that support LLM services
llm_providers = provider_registry.get_providers_for_service("llm")

# Get information about a specific provider
openai_info = provider_registry.get_provider_info("openai")

# Get models supported by a provider
openai_models = provider_registry.get_models("openai", "llm")

# Get features supported by a provider
google_translate_features = provider_registry.get_features("google", "translation")
```

You can also use the Router to query the provider registry:

```python
from marovi.api import Router, ServiceType

router = Router()

# List available providers for LLM services
llm_providers = router.list_available_providers(ServiceType.LLM)

# Get models supported by OpenAI for LLM services
openai_models = router.get_provider_models("openai", ServiceType.LLM)
```

### Multiple Clients

The Router supports managing multiple clients for each service type:

```python
from marovi.api import Router

router = Router()

# Register multiple LLM clients
openai_client_id = router.register_llm_client(
    provider="openai",
    api_key="your-openai-api-key"
)

anthropic_client_id = router.register_llm_client(
    provider="anthropic",
    api_key="your-anthropic-api-key"
)

# Register multiple translation clients
google_client_id = router.register_translation_client(
    provider="google",
    api_key="your-google-api-key"
)

deepl_client_id = router.register_translation_client(
    provider="deepl",
    api_key="your-deepl-api-key"
)

# Get a specific client by ID
openai_client = router.get_llm_client(openai_client_id)
google_client = router.get_translation_client(google_client_id)

# List all registered services
services = router.list_services()

# Check if a specific client is registered
has_anthropic = router.has_service(ServiceType.LLM, anthropic_client_id)
```

### Custom Providers

You can create custom providers by implementing the `LLMProvider` or `TranslationProvider` interfaces:

```python
from marovi.api.providers.base import LLMProvider
from marovi.api.schemas.llm import LLMRequest, LLMResponse

class CustomLLMProvider(LLMProvider):
    def initialize(self) -> None:
        pass
    
    def get_default_model(self) -> str:
        return "custom-model"
    
    def get_supported_models(self) -> List[str]:
        return ["custom-model"]
    
    def get_supported_languages(self) -> List[str]:
        return []
    
    def complete(self, request: LLMRequest) -> LLMResponse:
        # Implement your custom logic here
        pass
    
    async def acomplete(self, request: LLMRequest) -> LLMResponse:
        # Implement your custom async logic here
        pass
    
    async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        # Implement your custom streaming logic here
        pass
    
    def batch_complete(self, request: LLMBatchRequest) -> LLMBatchResponse:
        # Implement your custom batch logic here
        pass
    
    async def abatch_complete(self, request: LLMBatchRequest) -> LLMBatchResponse:
        # Implement your custom async batch logic here
        pass
```

### Custom Endpoints

You can register custom endpoints with the router:

```python
def custom_endpoint(request: Dict[str, Any]) -> Dict[str, Any]:
    # Implement your custom logic here
    return {"result": "custom response"}

router.register_custom_endpoint("custom", custom_endpoint)
custom_handler = router.get_custom_endpoint("custom")
```

### Batch Processing

The package supports batch processing for both LLM and translation services:

```python
# Batch LLM completions
responses = llm_client.batch_complete(
    prompts=["Prompt 1", "Prompt 2", "Prompt 3"],
    max_concurrency=5
)

# Batch translations
translations = translation_client.batch_translate(
    texts=["Text 1", "Text 2", "Text 3"],
    source_lang="en",
    target_lang="es",
    max_concurrency=5
)
```

### Async Usage

All operations support both synchronous and asynchronous interfaces:

```python
# Async LLM completion
response = await llm_client.acomplete(
    prompt="Write a poem about AI",
    temperature=0.7
)

# Async translation
translation = await translation_client.atranslate(
    text="Hello, world!",
    source_lang="en",
    target_lang="es"
)

# Async batch processing
responses = await llm_client.abatch_complete(
    prompts=["Prompt 1", "Prompt 2", "Prompt 3"],
    max_concurrency=5
)
```

## Environment Variables

This project uses environment variables for configuration. To set up your environment:

1. Copy the `.env.example` file to `.env`:
   ```bash
   cp .env.example .env
   ```

2. Edit the `.env` file and add your API keys and configuration settings.

IMPORTANT: Never commit your `.env` file with real API keys to version control.

For team onboarding, we use a `.env.example` file that shows the structure of required environment variables without the actual values. New team members can copy this file and add their own API keys.


## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

