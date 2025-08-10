## Using Cursor with Marovi

This repo is optimized for Cursor with `.cursorrules` and `.cursorignore`. Below is a quick guide for typical tasks.

### Setup
- Install deps: `poetry install`
- Run tests: `poetry run pytest -q`
- Example script: `poetry run python dev/arxiv_download_example.py`

### Key entry points
- `marovi/api/core/client.py`: `MaroviAPI` main client
- `marovi/api/core/router.py`: service discovery and client cache
- `marovi/api/providers/`: provider implementations and registry
- `marovi/api/custom/`: custom endpoints (authoring guide in `custom/README.md`)
- `marovi/api/schemas/`: Pydantic request/response models
- `marovi/api/config.py`: environment configuration (`settings`)

### Common workflows
- Add a custom endpoint
  1. Create request/response models in `marovi/api/custom/schemas/`
  2. Implement endpoint in `marovi/api/custom/endpoints/<name>.py`
  3. Register in `marovi/api/custom/core/registry.py` and/or `endpoints/endpoint_registry.yaml`
  4. Add prompt templates under `marovi/api/custom/prompts/`
- Add a provider
  1. Create provider class in `marovi/api/providers/<id>.py`
  2. Add a lazy getter in `marovi/api/providers/__init__.py` and update `register_default_providers`
  3. Update provider metadata if needed
- Extend schemas/clients
  - Update `marovi/api/schemas` and wire through the matching client (`clients/llm.py`, `clients/translation.py`, `clients/custom.py`)

### Environment
Use `marovi.api.config.settings` for configuration (no direct `os.getenv` in business logic).
- Keys: `MAROVI_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `GOOGLE_TRANSLATE_API_KEY`, `DEEPL_API_KEY`.

### Tips for AI edits
- Keep edits small and localized; avoid breaking public APIs.
- Maintain lazy imports to prevent circular dependencies.
- Add/keep type hints; use Pydantic models for IO.
- Ensure tests run green after changes.

### Troubleshooting
- Circular import? Convert cross-layer imports to lazy (import inside function) and review `__all__`.
- Provider not available? Missing dependency/keys should warn, not crash; see logs.
- Cache issues? `Router` caches clients by key (provider:type:model). 