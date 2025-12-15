"""Tests for LLM configuration and async client."""

import json
from pathlib import Path

import pytest
import requests

from src.models.llms import AsyncLLM, LLMConfig


def _write_config(tmp_path: Path) -> Path:
    data = {
        "demo": {
            "provider": "openai",
            "description": "demo model",
            "base_url": "https://example.com/v1",
            "api_key": "test-key",
            "temperature": 0.5,
        }
    }
    config_path = tmp_path / "models.json"
    config_path.write_text(json.dumps(data), encoding="utf-8")
    return config_path


def test_llmconfig_loads_model(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)

    config = LLMConfig.load("demo", path=config_path)

    assert config.name == "demo"
    assert config.provider == "openai"
    assert config.base_url == "https://example.com/v1"
    assert config.temperature == 0.5


def test_llmconfig_missing_model_raises(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)

    with pytest.raises(ValueError):
        LLMConfig.load("missing", path=config_path)


@pytest.mark.asyncio
async def test_async_llm_generate_uses_config(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    captured: dict[str, object] = {}

    class FakeResponse:
        def raise_for_status(self) -> None:  # noqa: D401 - simple stub
            return None

        def json(self) -> dict:
            return {"choices": [{"message": {"content": "hello"}}]}

    class FakeSession:
        def post(self, url, headers=None, json=None, timeout=None):
            captured["url"] = url
            captured["headers"] = headers
            captured["payload"] = json
            captured["timeout"] = timeout
            return FakeResponse()

    llm = AsyncLLM("demo", config_path=config_path, session=FakeSession())

    result = await llm.generate("hi", timeout=5)

    assert result == "hello"
    assert captured["payload"]["model"] == "demo"
    assert captured["headers"]["Authorization"] == "Bearer test-key"
    assert captured["timeout"] == 5
    assert str(captured["url"]).endswith("/chat/completions")


@pytest.mark.asyncio
async def test_async_llm_generate_raises_on_http_error(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)

    class ErrorResponse:
        def raise_for_status(self):
            raise requests.HTTPError("bad response")

        def json(self):
            return {}

    class FakeSession:
        def post(self, *_, **__):
            return ErrorResponse()

    llm = AsyncLLM("demo", config_path=config_path, session=FakeSession())

    with pytest.raises(requests.HTTPError):
        await llm.generate("hello")


@pytest.mark.asyncio
async def test_async_llm_generate_validates_response(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)

    class EmptyResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {}

    class FakeSession:
        def post(self, *_, **__):
            return EmptyResponse()

    llm = AsyncLLM("demo", config_path=config_path, session=FakeSession())

    with pytest.raises(ValueError):
        await llm.generate("hello")
