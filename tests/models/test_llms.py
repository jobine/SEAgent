'''Tests for LLM configuration and async client.'''

import json
from pathlib import Path
from typing import Any

import pytest
import src.models.llms as llms
from src.models.llms import (
    AsyncBaseLLM,
    AsyncGeminiLLM,
    AsyncLLM,
    AsyncOpenAILLM,
    LLMConfig,
)


def _write_config(tmp_path: Path, provider: str = 'openai') -> Path:
    data = {
        'demo': {
            'provider': provider,
            'description': 'demo model',
            'base_url': 'https://example.com/v1',
            'api_key': 'test-key',
            'temperature': 0.5,
            'top_p': 0.9
        },
        'gemini-model': {
            'provider': 'gemini',
            'description': 'gemini demo model',
            'base_url': '',
            'api_key': 'gemini-test-key',
            'temperature': 0.7,
            'top_p': 1.0
        }
    }
    config_path = tmp_path / 'models.json'
    config_path.write_text(json.dumps(data), encoding='utf-8')
    return config_path


@pytest.fixture(autouse=True)
def reset_llmconfig_cache():
    '''Clear caches before and after each test to avoid cross-test coupling.'''
    llms.LLMConfig._file_cache.clear()
    llms.LLMConfig._instance_cache.clear()
    yield
    llms.LLMConfig._file_cache.clear()
    llms.LLMConfig._instance_cache.clear()


# =============================================================================
# LLMConfig Tests
# =============================================================================

def test_llmconfig_loads_model(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)

    config = LLMConfig.load('demo', path=config_path)

    assert config.name == 'demo'
    assert config.provider == 'openai'
    assert config.base_url == 'https://example.com/v1'
    assert config.temperature == 0.5
    assert config.top_p == 0.9


def test_llmconfig_missing_model_raises(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)

    with pytest.raises(ValueError):
        LLMConfig.load('missing', path=config_path)


def test_llmconfig_caches_json_load(monkeypatch, tmp_path: Path) -> None:
    data = {
        'a': {'provider': 'x', 'base_url': 'https://example.com', 'api_key': 'k'},
        'b': {'provider': 'y', 'base_url': 'https://example.com', 'api_key': 'k'},
    }
    config_path = tmp_path / 'models.json'
    config_path.write_text(json.dumps(data), encoding='utf-8')

    original_load = llms.json.load
    calls = {'count': 0}

    def fake_load(fp):
        calls['count'] += 1
        return original_load(fp)

    monkeypatch.setattr(llms, 'json', llms.json)
    monkeypatch.setattr(llms.json, 'load', fake_load)

    LLMConfig.load('a', path=config_path)
    LLMConfig.load('b', path=config_path)

    assert calls['count'] == 1


# =============================================================================
# AsyncLLM Factory Tests
# =============================================================================

def test_async_llm_factory_returns_openai_for_openai_provider(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, provider='openai')

    llm = AsyncLLM('demo', config_path=config_path)

    assert isinstance(llm, AsyncOpenAILLM)
    assert llm.config.provider == 'openai'


def test_async_llm_factory_returns_openai_for_azure_provider(tmp_path: Path) -> None:
    data = {'azure-model': {'provider': 'azure', 'base_url': 'https://azure.com', 'api_key': 'k'}}
    config_path = tmp_path / 'models.json'
    config_path.write_text(json.dumps(data), encoding='utf-8')

    llm = AsyncLLM('azure-model', config_path=config_path)

    assert isinstance(llm, AsyncOpenAILLM)


def test_async_llm_factory_returns_gemini_for_gemini_provider(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)

    llm = AsyncLLM('gemini-model', config_path=config_path)

    assert isinstance(llm, AsyncGeminiLLM)
    assert llm.config.provider == 'gemini'


def test_async_llm_factory_returns_gemini_for_google_provider(tmp_path: Path) -> None:
    data = {'google-model': {'provider': 'google', 'base_url': '', 'api_key': 'k'}}
    config_path = tmp_path / 'models.json'
    config_path.write_text(json.dumps(data), encoding='utf-8')

    llm = AsyncLLM('google-model', config_path=config_path)

    assert isinstance(llm, AsyncGeminiLLM)


def test_async_llm_factory_defaults_to_openai_for_unknown_provider(tmp_path: Path) -> None:
    data = {'unknown-model': {'provider': 'unknown', 'base_url': 'https://example.com', 'api_key': 'k'}}
    config_path = tmp_path / 'models.json'
    config_path.write_text(json.dumps(data), encoding='utf-8')

    llm = AsyncLLM('unknown-model', config_path=config_path)

    assert isinstance(llm, AsyncOpenAILLM)


def test_async_llm_register_provider(tmp_path: Path) -> None:
    '''Test registering a custom provider.'''

    class CustomLLM(AsyncBaseLLM):
        async def __call__(self, prompt: str, **kwargs: Any) -> str:
            return 'custom response'

    data = {'custom-model': {'provider': 'custom', 'base_url': '', 'api_key': 'k'}}
    config_path = tmp_path / 'models.json'
    config_path.write_text(json.dumps(data), encoding='utf-8')

    # Register custom provider
    AsyncLLM.register_provider('custom', CustomLLM)

    llm = AsyncLLM('custom-model', config_path=config_path)

    assert isinstance(llm, CustomLLM)

    # Clean up
    del AsyncLLM._provider_registry['custom']


# =============================================================================
# AsyncOpenAILLM Tests
# =============================================================================

@pytest.mark.asyncio
async def test_async_openai_llm_generate_uses_config(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    captured: dict[str, object] = {}

    class FakeMessage:
        def __init__(self, content: str):
            self.content = content

    class FakeChoice:
        def __init__(self, message):
            self.message = message

    class FakeResponse:
        def __init__(self, content: str):
            self.choices = [FakeChoice(FakeMessage(content))]

    class FakeCompletions:
        async def create(self, **kwargs):
            captured['payload'] = kwargs
            return FakeResponse('hello')

    class FakeChat:
        def __init__(self):
            self.completions = FakeCompletions()

    class FakeClient:
        def __init__(self):
            self.chat = FakeChat()

    llm = AsyncLLM('demo', config_path=config_path, client=FakeClient())

    result = await llm('hi', timeout=5)

    assert result == 'hello'
    assert captured['payload']['model'] == 'demo'
    assert captured['payload']['messages'] == [{'role': 'user', 'content': 'hi'}]
    assert captured['payload']['temperature'] == 0.5
    assert captured['payload']['top_p'] == 0.9


@pytest.mark.asyncio
async def test_async_openai_llm_payload_override(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    captured: dict[str, object] = {}

    class FakeMessage:
        def __init__(self, content: str):
            self.content = content

    class FakeChoice:
        def __init__(self, message):
            self.message = message

    class FakeResponse:
        def __init__(self, content: str):
            self.choices = [FakeChoice(FakeMessage(content))]

    class FakeCompletions:
        async def create(self, **kwargs):
            captured['payload'] = kwargs
            return FakeResponse('hello')

    class FakeChat:
        def __init__(self):
            self.completions = FakeCompletions()

    class FakeClient:
        def __init__(self):
            self.chat = FakeChat()

    llm = AsyncLLM('demo', config_path=config_path, client=FakeClient())

    await llm('hi', payload_override={'temperature': 0.9, 'max_tokens': 100})

    assert captured['payload']['temperature'] == 0.9
    assert captured['payload']['max_tokens'] == 100


@pytest.mark.asyncio
async def test_async_openai_llm_generate_raises_on_http_error(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)

    class FakeCompletions:
        async def create(self, **_):
            raise RuntimeError('bad response')

    class FakeChat:
        def __init__(self):
            self.completions = FakeCompletions()

    class FakeClient:
        def __init__(self):
            self.chat = FakeChat()

    llm = AsyncLLM('demo', config_path=config_path, client=FakeClient())

    with pytest.raises(RuntimeError):
        await llm('hello')


@pytest.mark.asyncio
async def test_async_openai_llm_generate_validates_response(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)

    class FakeCompletions:
        async def create(self, **_):
            return {}

    class FakeChat:
        def __init__(self):
            self.completions = FakeCompletions()

    class FakeClient:
        def __init__(self):
            self.chat = FakeChat()

    llm = AsyncLLM('demo', config_path=config_path, client=FakeClient())

    with pytest.raises(ValueError):
        await llm('hello')


def test_async_openai_llm_extract_content_from_dict() -> None:
    response = {
        'choices': [
            {'message': {'content': 'test content'}}
        ]
    }
    result = AsyncOpenAILLM._extract_content(response)
    assert result == 'test content'


def test_async_openai_llm_extract_content_from_text_field() -> None:
    response = {
        'choices': [
            {'text': 'text content'}
        ]
    }
    result = AsyncOpenAILLM._extract_content(response)
    assert result == 'text content'


# =============================================================================
# AsyncGeminiLLM Tests
# =============================================================================

class FakeGeminiModels:
    '''Reusable fake models class for Gemini tests.'''

    def __init__(self, response=None, error=None, captured=None):
        self._response = response
        self._error = error
        self._captured = captured

    async def generate_content(self, **kwargs):
        if self._captured is not None:
            self._captured['payload'] = kwargs
        if self._error:
            raise self._error
        return self._response


class FakeGeminiAio:
    '''Fake async context manager for Gemini client.aio.'''

    def __init__(self, models):
        self.models = models

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class FakeGeminiClient:
    '''Fake Gemini client for testing.'''

    def __init__(self, response=None, error=None, captured=None):
        models = FakeGeminiModels(response=response, error=error, captured=captured)
        self.aio = FakeGeminiAio(models)


class FakeGeminiResponse:
    '''Fake Gemini response.'''

    def __init__(self, text: str | None = None):
        self.text = text
        self.candidates = None


@pytest.mark.asyncio
async def test_async_gemini_llm_generate_uses_config(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    captured: dict[str, object] = {}

    client = FakeGeminiClient(
        response=FakeGeminiResponse(text='gemini hello'),
        captured=captured
    )
    llm = AsyncLLM('gemini-model', config_path=config_path, client=client)

    result = await llm('hi')

    assert result == 'gemini hello'
    assert captured['payload']['model'] == 'gemini-model'
    assert captured['payload']['contents'] == 'hi'


@pytest.mark.asyncio
async def test_async_gemini_llm_payload_override(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    captured: dict[str, object] = {}

    client = FakeGeminiClient(
        response=FakeGeminiResponse(text='gemini hello'),
        captured=captured
    )
    llm = AsyncLLM('gemini-model', config_path=config_path, client=client)

    await llm('hi', payload_override={'config': {'temperature': 0.9}})

    # Verify config was applied (it would be in the config object)
    assert 'config' in captured['payload']


@pytest.mark.asyncio
async def test_async_gemini_llm_raises_on_error(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)

    client = FakeGeminiClient(error=RuntimeError('gemini error'))
    llm = AsyncLLM('gemini-model', config_path=config_path, client=client)

    with pytest.raises(RuntimeError):
        await llm('hello')


@pytest.mark.asyncio
async def test_async_gemini_llm_validates_response(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)

    client = FakeGeminiClient(response=FakeGeminiResponse(text=None))
    llm = AsyncLLM('gemini-model', config_path=config_path, client=client)

    with pytest.raises(ValueError):
        await llm('hello')


def test_async_gemini_llm_extract_content_from_text() -> None:
    class FakeResponse:
        text = 'gemini text'

    result = AsyncGeminiLLM._extract_content(FakeResponse())
    assert result == 'gemini text'


def test_async_gemini_llm_extract_content_from_candidates() -> None:
    class FakePart:
        text = 'candidate text'

    class FakeContent:
        parts = [FakePart()]

    class FakeCandidate:
        content = FakeContent()

    class FakeResponse:
        text = None
        candidates = [FakeCandidate()]

    result = AsyncGeminiLLM._extract_content(FakeResponse())
    assert result == 'candidate text'
