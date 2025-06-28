# Codex Environment Setup

This guide describes how to prepare a development environment so Codex can execute the code in this repository.

## Requirements
- **Python** 3.10 or later
- **Poetry** for dependency management

## Installation Steps

1. **Clone the repository** and change into its directory.
2. **Install dependencies** using Poetry:
   ```bash
   poetry install
   ```
3. **Activate the virtual environment** or prefix commands with `poetry run` to execute within the Poetry environment.
4. **Create your `.env` file**. Copy the provided sample and fill in your API keys:
   ```bash
   cp .env.example .env
   ```
5. Edit `.env` to provide values for keys such as `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_TRANSLATE_API_KEY`, and other settings shown in `.env.example`.
6. Use the examples or pipelines to confirm the installation:
   ```bash
   python examples/po_translation_example.py
   # or list pipelines
   python -m marovi.pipelines.cli list
   ```

## Quick Start Usage
A minimal example using the API is shown in the README:
```python
from marovi.api import Router, ServiceType

router = Router()
router.register_llm_client(
    provider="openai",
    model="gpt-4",
    api_key="your-api-key",
)
router.register_translation_client(
    provider="google",
    api_key="your-api-key",
)
llm_client = router.get_llm_client()
response = llm_client.complete(
    prompt="Write a poem about AI",
    temperature=0.7,
)
print(response.content)
translation_client = router.get_translation_client()
translation = translation_client.translate(
    text="Hello, world!",
    source_lang="en",
    target_lang="es",
)
print(translation.content)
```

## Environment Variables
The README outlines how to configure environment variables:
1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
2. Add your API keys and settings to `.env`. Do **not** commit the file with real credentials.

For reference, `.env.example` lists keys such as `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, and various defaults controlling retries, caching, and concurrency.

## Codex UI Setup

When launching the project in OpenAI's Codex interface, put the following commands in the **Setup Commands** field. These install dependencies and prepare the environment just as a developer would locally:

```bash
python --version          # confirm a Python 3.10+ runtime
pip install -U pip        # upgrade pip in the Codex container
pip install poetry        # install Poetry for dependency management
poetry install            # install project dependencies
cp .env.example .env      # create the environment file
poetry run pytest -q      # optional: run tests to verify installation
```

Fill in your secret keys in the Codex **Environment Variables** section, matching the names from `.env.example`:

- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GOOGLE_API_KEY`
- `GOOGLE_TRANSLATE_API_KEY`
- `DEEPL_API_KEY`

Other variables such as `DEFAULT_MAX_RETRIES` or `ENABLE_CACHE` can also be added if you want to override the defaults from `.env.example`.

