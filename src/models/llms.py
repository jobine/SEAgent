'''Async LLM helpers backed by JSON configuration.'''

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Dict

from openai import AsyncOpenAI


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


class AsyncLLM:
	'''Async wrapper around HTTP LLM calls using JSON-defined config.'''

	def __init__(
		self,
		model: str,
		*,
		config_path: str | Path | None = None,
		client: AsyncOpenAI | None = None,
	) -> None:
		self.config = LLMConfig.load(model, path=config_path)
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
	

if __name__ == '__main__':
	# Example usage
	import asyncio

	async def main():
		llm = AsyncLLM('gpt-4o-mini')
		result = await llm('Hello, world!')
		print(result)

	asyncio.run(main())
