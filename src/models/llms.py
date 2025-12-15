"""Async LLM helpers backed by JSON configuration."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import requests


@dataclass
class LLMConfig:
	"""Configuration for a language model loaded from models.json."""

	name: str
	provider: str
	description: str
	base_url: str
	api_key: str
	temperature: Optional[float]

	@classmethod
	def load(cls, model: str, path: str | Path | None = None) -> "LLMConfig":
		"""Load configuration for a given model key.

		Args:
			model: Model key defined in the JSON file.
			path: Optional override path to the JSON config file.
		"""
		config_path = Path(path) if path else Path(__file__).with_name("models.json")

		if not config_path.is_file():
			raise FileNotFoundError(f"Config file not found: {config_path}")

		with config_path.open("r", encoding="utf-8") as fp:
			data: Dict[str, Dict[str, Any]] = json.load(fp)

		if model not in data:
			raise ValueError(f"Model '{model}' not found in {config_path}")

		entry = data[model]
		return cls(
			name=model,
			provider=entry.get("provider", ""),
			description=entry.get("description", ""),
			base_url=entry.get("base_url", ""),
			api_key=entry.get("api_key", ""),
			temperature=entry.get("temperature"),
		)


class AsyncLLM:
	"""Async wrapper around HTTP LLM calls using JSON-defined config."""

	def __init__(
		self,
		model: str,
		*,
		config_path: str | Path | None = None,
		session: Optional[requests.Session] = None,
	) -> None:
		self.config = LLMConfig.load(model, path=config_path)
		self._session = session or requests.Session()

	async def generate(self, prompt: str, **kwargs: Any) -> str:
		"""Call the LLM asynchronously and return the generated text."""
		payload: Dict[str, Any] = {
			"model": self.config.name,
			"messages": [{"role": "user", "content": prompt}],
			"temperature": self.config.temperature,
		}
		payload_override: Dict[str, Any] = kwargs.pop("payload_override", {})
		payload.update(payload_override)

		headers = {"Content-Type": "application/json"}
		if self.config.api_key:
			headers["Authorization"] = f"Bearer {self.config.api_key}"

		url = f"{self.config.base_url.rstrip('/')}/chat/completions"
		timeout = kwargs.pop("timeout", 30)

		# Run blocking HTTP request in a thread to keep the async interface.
		response = await asyncio.to_thread(
			self._session.post,
			url,
			headers=headers,
			json=payload,
			timeout=timeout,
		)
		response.raise_for_status()
		data = response.json()
		return self._extract_content(data)

	@staticmethod
	def _extract_content(data: Dict[str, Any]) -> str:
		choices = data.get("choices") or []
		if not choices:
			raise ValueError("LLM response missing 'choices'")

		first = choices[0]
		message = first.get("message") or {}
		if "content" in message:
			return str(message["content"])

		if "text" in first:
			return str(first["text"])

		raise ValueError("LLM response missing generated content")
