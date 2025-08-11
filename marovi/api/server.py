"""FastAPI application exposing Marovi services as HTTP endpoints."""

from fastapi import FastAPI, HTTPException
from typing import Any, Dict

from .core.router import Router
from .schemas.llm import LLMRequest, LLMResponse
from .schemas.translation import TranslationRequest, TranslationResponse

# Create a router with default providers so the server is usable out of the box
router = Router.create(default_providers=True)

app = FastAPI(title="Marovi API", description="Unified API for LLMs, translation, and custom workflows")


@app.post("/chat/completions", response_model=LLMResponse)
def chat_completions(request: LLMRequest, provider: str | None = None, model: str | None = None) -> LLMResponse:
    """Return a chat completion from the selected LLM provider."""
    try:
        client = router.get_llm(provider=provider, model=model or request.model)
        return client.complete(request)
    except Exception as exc:  # pragma: no cover - pass through error message
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/translate", response_model=TranslationResponse)
def translate(request: TranslationRequest, provider: str | None = None) -> TranslationResponse:
    """Translate text using the chosen provider."""
    try:
        client = router.get_translation(provider=provider)
        return client.translate(request)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/custom/{endpoint}")
def custom_endpoint(endpoint: str, payload: Dict[str, Any]) -> Any:
    """Invoke a registered custom endpoint by name."""
    try:
        handler = getattr(router.custom, endpoint)
        return handler(**payload)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=400, detail=str(exc)) from exc
