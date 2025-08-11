"""Helpers for interacting with a LiteLLM translation gateway."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Union

import requests

try:  # pragma: no cover - allow running without pydantic
    from .schemas.translation import TranslationResponse, TranslationFormat
except Exception:  # pragma: no cover
    from types import SimpleNamespace

    class TranslationFormat:
        TEXT = "text"

    class TranslationResponse(SimpleNamespace):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            if not hasattr(self, "success"):
                self.success = True


def translate(
    provider: str,
    text: Union[str, List[str]],
    source_lang: str,
    target_lang: str,
    opts: Optional[Dict[str, Any]] = None,
) -> TranslationResponse:
    """Translate text using the LiteLLM gateway.

    Args:
        provider: Translation provider identifier (google, deepl, chatgpt)
        text: Text or list of texts to translate
        source_lang: Source language code
        target_lang: Target language code
        opts: Optional parameters to pass through to the gateway

    Returns:
        TranslationResponse with translated content
    """
    base_url = os.getenv("LITELLM_API_BASE")
    if not base_url:
        raise ValueError("LITELLM_API_BASE environment variable not set")
    api_key = os.getenv("MAROVI_API_KEY")
    if not api_key:
        raise ValueError("MAROVI_API_KEY environment variable not set")

    provider_keys = {
        "google": os.getenv("GOOGLE_API_KEY"),
        "deepl": os.getenv("DEEPL_API_KEY"),
        "chatgpt": os.getenv("OPENAI_API_KEY"),
    }
    headers = {"Authorization": f"Bearer {api_key}"}
    provider_key = provider_keys.get(provider)
    if provider_key:
        headers["X-Provider-Key"] = provider_key

    opts = opts or {}
    if provider in {"google", "deepl"}:
        payload: Dict[str, Any] = {
            "provider": provider,
            "text": text,
            "source_lang": source_lang,
            "target_lang": target_lang,
        }
        payload.update(opts)
        resp = requests.post(f"{base_url}/v1/translate", json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        return TranslationResponse(**data)

    if provider == "chatgpt":
        system_prompt = (
            f"You are a translation engine. Translate from {source_lang} to {target_lang}."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ]
        payload = {
            "model": opts.get("model", "gpt-4o-mini"),
            "messages": messages,
        }
        resp = requests.post(
            f"{base_url}/v1/chat/completions", json=payload, headers=headers
        )
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        total_chars = (
            len(text) if isinstance(text, str) else sum(len(t) for t in text)
        )
        return TranslationResponse(
            content=content,
            format=TranslationFormat.TEXT,
            provider="chatgpt",
            source_lang=source_lang,
            target_lang=target_lang,
            total_characters=total_chars,
            success=True,
        )

    raise ValueError(f"Unsupported provider: {provider}")
