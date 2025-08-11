import os
from types import SimpleNamespace

import sys
import types

# Provide a minimal litellm stub to avoid external dependency
litellm_stub = types.SimpleNamespace(
    completion=lambda **kwargs: {
        "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
        "model": kwargs.get("model", "stub"),
        "usage": {},
    },
    acompletion=lambda **kwargs: None,
    astream_completion=lambda **kwargs: iter(()),
)
sys.modules.setdefault("litellm", litellm_stub)

# Provide a minimal pydantic stub
pydantic_stub = types.SimpleNamespace(
    BaseModel=object,
    Field=lambda *a, **k: None,
    validator=lambda *a, **k: (lambda f: f),
)
sys.modules.setdefault("pydantic", pydantic_stub)

from marovi.api import litellm_gateway
from marovi.api.clients import llm as llm_module
from marovi.api.providers import litellm as litellm_provider

class FakeTR(SimpleNamespace):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not hasattr(self, "success"):
            self.success = True

litellm_gateway.TranslationResponse = FakeTR
litellm_gateway.TranslationFormat = types.SimpleNamespace(TEXT="text")

class FakeLLMRequest(SimpleNamespace):
    def __init__(self, **kwargs):
        kwargs.setdefault("response_format", None)
        super().__init__(**kwargs)

llm_module.LLMRequest = FakeLLMRequest
class FakeLLMResponse(SimpleNamespace):
    pass
litellm_provider.LLMResponse = FakeLLMResponse

import marovi.api.clients.translation as t
from marovi.api.clients.llm import LLMClient


def test_litellm_completion_adapter(monkeypatch):
    monkeypatch.setenv("MAROVI_API_KEY", "m-key")
    monkeypatch.setenv("LITELLM_API_BASE", "https://gw")

    called = {}

    def fake_completion(**kwargs):
        called.update(kwargs)
        return {
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "model": kwargs["model"],
            "usage": {}
        }

    monkeypatch.setattr("litellm.completion", fake_completion)

    client = LLMClient(provider="litellm")
    resp = client.complete("hi", model="gpt-4o-mini")

    assert resp == "ok"
    assert called["api_base"] == "https://gw"
    assert called["api_key"] == "m-key"


def test_translate_google(monkeypatch):
    monkeypatch.setenv("MAROVI_API_KEY", "m-key")
    monkeypatch.setenv("LITELLM_API_BASE", "https://gw")
    monkeypatch.setenv("GOOGLE_API_KEY", "g-key")

    def fake_post(url, json=None, headers=None):
        assert url == "https://gw/v1/translate"
        assert headers["Authorization"] == "Bearer m-key"
        assert headers["X-Provider-Key"] == "g-key"
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {
                "content": "hola",
                "format": "text",
                "provider": "google",
                "source_lang": "en",
                "target_lang": "es",
                "total_characters": 5,
                "success": True,
            },
        )

    monkeypatch.setattr("requests.post", fake_post)

    client = t.TranslationClient(provider="google")
    result = client.translate("hello", "en", "es")
    assert result == "hola"


def test_translate_deepl(monkeypatch):
    monkeypatch.setenv("MAROVI_API_KEY", "m-key")
    monkeypatch.setenv("LITELLM_API_BASE", "https://gw")
    monkeypatch.setenv("DEEPL_API_KEY", "d-key")

    def fake_post(url, json=None, headers=None):
        assert url == "https://gw/v1/translate"
        assert headers["Authorization"] == "Bearer m-key"
        assert headers["X-Provider-Key"] == "d-key"
        assert json["provider"] == "deepl"
        assert json["text"] == "hallo"
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {
                "content": "hello",
                "format": "text",
                "provider": "deepl",
                "source_lang": "de",
                "target_lang": "en",
                "total_characters": 5,
                "success": True,
            },
        )

    monkeypatch.setattr("requests.post", fake_post)

    client = t.TranslationClient(provider="deepl")
    result = client.translate("hallo", "de", "en")
    assert result == "hello"


def test_translate_chatgpt(monkeypatch):
    monkeypatch.setenv("MAROVI_API_KEY", "m-key")
    monkeypatch.setenv("LITELLM_API_BASE", "https://gw")
    monkeypatch.setenv("OPENAI_API_KEY", "o-key")

    def fake_post(url, json=None, headers=None):
        assert url == "https://gw/v1/chat/completions"
        assert headers["X-Provider-Key"] == "o-key"
        assert json["messages"][0]["role"] == "system"
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {
                "choices": [{"message": {"content": "hola"}}]
            },
        )

    monkeypatch.setattr("requests.post", fake_post)

    client = t.TranslationClient(provider="chatgpt")
    result = client.translate("hello", "en", "es")
    assert result == "hola"
