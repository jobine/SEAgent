'''Tests for LLM configuration and async client.'''

import json
from pathlib import Path

import pytest
import src.models.llms as llms
from src.models.llms import AsyncLLM, LLMConfig


def _write_config(tmp_path: Path) -> Path:
    data = {
        'demo': {
            'provider': 'openai',
            'description': 'demo model',
            'base_url': 'https://example.com/v1',
            'api_key': 'test-key',
            'temperature': 0.5,
            'top_p': 0.9
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


@pytest.mark.asyncio
async def test_async_llm_generate_uses_config(tmp_path: Path) -> None:
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
async def test_async_llm_generate_raises_on_http_error(tmp_path: Path) -> None:
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
async def test_async_llm_generate_validates_response(tmp_path: Path) -> None:
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
