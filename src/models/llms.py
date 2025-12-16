'''Async LLM helpers backed by JSON configuration.'''

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Dict, Type

from openai import AsyncOpenAI
from google import genai

@dataclass
class LLMConfig:
	'''Configuration for a language model loaded from models.json.'''

	name: str
	provider: str
	description: str
	base_url: str
	api_key: str
	temperature: float | None = None
	top_p: float | None = None
	_file_cache: ClassVar[Dict[Path, Dict[str, Dict[str, Any]]]] = {}
	_instance_cache: ClassVar[Dict[tuple[Path, str], 'LLMConfig']] = {}

	@classmethod
	def load(cls, model: str, path: str | Path | None = None) -> 'LLMConfig':
		'''Load configuration for a given model key.

		Args:
			model: Model key defined in the JSON file.
			path: Optional override path to the JSON config file.
		'''
		config_path = (Path(path) if path else Path(__file__).with_name('models.json')).resolve()
		cache_key = (config_path, model)

		if cache_key in cls._instance_cache:
			return cls._instance_cache[cache_key]

		if not config_path.is_file():
			raise FileNotFoundError(f'Config file not found: {config_path}')

		if config_path not in cls._file_cache:
			with config_path.open('r', encoding='utf-8') as fp:
				cls._file_cache[config_path] = json.load(fp)

		data = cls._file_cache[config_path]

		if model not in data:
			raise ValueError(f'Model "{model}" not found in {config_path}')

		entry = data[model]
		instance = cls(
			name=model,
			provider=entry.get('provider', entry.get('type', '')),
			description=entry.get('description', ''),
			base_url=entry.get('base_url', ''),
			api_key=entry.get('api_key', ''),
			temperature=entry.get('temperature', 0.7),
			top_p=entry.get('top_p', 1.0),
		)
		cls._instance_cache[cache_key] = instance
		return instance


class AsyncBaseLLM(ABC):
	'''Abstract base class for async LLM implementations.'''

	def __init__(self, config: LLMConfig) -> None:
		self.config = config

	@abstractmethod
	async def __call__(self, prompt: str, **kwargs: Any) -> str:
		'''Call the LLM asynchronously and return text.'''
		...


class AsyncOpenAILLM(AsyncBaseLLM):
	'''Async wrapper for OpenAI-compatible APIs.'''

	def __init__(
		self,
		config: LLMConfig,
		*,
		client: AsyncOpenAI | None = None,
	) -> None:
		super().__init__(config)
		self._client = client or AsyncOpenAI(
			api_key=self.config.api_key or None,
			base_url=self.config.base_url or None,
		)

	async def __call__(self, prompt: str, **kwargs: Any) -> str:
		'''Call the LLM asynchronously via OpenAI Chat Completions API and return text.'''
		payload: Dict[str, Any] = {
			'model': self.config.name,
			'messages': [{'role': 'user', 'content': prompt}],
			'temperature': self.config.temperature,
			'top_p': self.config.top_p,
		}
		payload_override: Dict[str, Any] = kwargs.pop('payload_override', {})
		payload.update(payload_override)

		response = await self._client.chat.completions.create(**payload)
		return self._extract_content(response)

	@staticmethod
	def _extract_content(response: Any) -> str:
		choices = getattr(response, 'choices', None)
		if choices is None and isinstance(response, dict):
			choices = response.get('choices')

		if choices:
			first = choices[0]
			message = getattr(first, 'message', None) or (first.get('message') if isinstance(first, dict) else None)
			if message:
				content = getattr(message, 'content', None) or (message.get('content') if isinstance(message, dict) else None)
				if content is not None:
					return str(content)

			text = getattr(first, 'text', None) or (first.get('text') if isinstance(first, dict) else None)
			if text is not None:
				return str(text)

		raise ValueError('LLM response missing generated content')


class AsyncGeminiLLM(AsyncBaseLLM):
	'''Async wrapper for Google Gemini API.'''

	def __init__(
		self,
		config: LLMConfig,
		*,
		client: genai.Client | None = None,
	) -> None:
		super().__init__(config)
		self._client = client or genai.Client(api_key=self.config.api_key or None)

	async def __call__(self, prompt: str, **kwargs: Any) -> str:
		'''Call the Gemini LLM asynchronously and return text.'''
		payload: Dict[str, Any] = {
			'model': self.config.name,
			'contents': prompt,
		}

		# Build generation config
		config_dict: Dict[str, Any] = {}
		if self.config.temperature is not None:
			config_dict['temperature'] = self.config.temperature
		if self.config.top_p is not None:
			config_dict['top_p'] = self.config.top_p

		payload_override: Dict[str, Any] = kwargs.pop('payload_override', {})
		config_dict.update(payload_override.pop('config', {}))

		if config_dict:
			payload['config'] = genai.types.GenerateContentConfig(**config_dict)

		payload.update(payload_override)

		async with self._client.aio as aio_client:
			response = await aio_client.models.generate_content(**payload)

		return self._extract_content(response)

	@staticmethod
	def _extract_content(response: Any) -> str:
		# Handle Gemini response structure
		if hasattr(response, 'text') and response.text:
			return str(response.text)

		if hasattr(response, 'candidates') and response.candidates:
			candidate = response.candidates[0]
			if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
				parts = candidate.content.parts
				if parts and hasattr(parts[0], 'text'):
					return str(parts[0].text)

		raise ValueError('Gemini response missing generated content')


class AsyncLLM:
	'''Factory class for creating async LLM instances based on provider.'''

	_provider_registry: ClassVar[Dict[str, Type[AsyncBaseLLM]]] = {
		'openai': AsyncOpenAILLM,
		'azure': AsyncOpenAILLM,
		'azure_openai': AsyncOpenAILLM,
		'gemini': AsyncGeminiLLM,
		'google': AsyncGeminiLLM,
	}

	def __new__(
		cls,
		model: str,
		*,
		config_path: str | Path | None = None,
		**kwargs: Any,
	) -> AsyncBaseLLM:
		'''Create an async LLM instance based on the provider in config.

		Args:
			model: Model key defined in the JSON file.
			config_path: Optional override path to the JSON config file.
			**kwargs: Additional arguments passed to the LLM implementation.

		Returns:
			AsyncBaseLLM instance (AsyncOpenAILLM or AsyncGeminiLLM).
		'''
		config = LLMConfig.load(model, path=config_path)
		provider = config.provider.lower()

		llm_class = cls._provider_registry.get(provider)
		if llm_class is None:
			# Default to OpenAI-compatible API for unknown providers
			llm_class = AsyncOpenAILLM

		return llm_class(config, **kwargs)

	@classmethod
	def register_provider(cls, provider: str, llm_class: Type[AsyncBaseLLM]) -> None:
		'''Register a custom LLM implementation for a provider.

		Args:
			provider: Provider name (case-insensitive).
			llm_class: LLM class that extends AsyncBaseLLM.
		'''
		cls._provider_registry[provider.lower()] = llm_class


if __name__ == '__main__':
	# Example usage
	import asyncio

	async def main():
		# OpenAI example
		openai_llm = AsyncLLM('gpt-4o-mini')
		result = await openai_llm('Hello, world!')
		print(result)

		# Gemini example (uncomment if configured)
		gemini_llm = AsyncLLM('gemini-3-pro-preview')
		result = await gemini_llm('Hello from Gemini!')
		print(result)

	asyncio.run(main())
