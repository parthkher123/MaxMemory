"""LLM providers behind one interface.

Every model call in the layer asks for one thing: a system prompt and a user
message in, a validated pydantic object out. That is the whole interface, so
any provider that can return JSON can sit behind it.

  anthropic  - Claude, via the Anthropic SDK's structured output.
  openai     - any OpenAI-compatible Chat Completions endpoint: OpenAI itself,
               or Ollama, Groq, OpenRouter, LM Studio, vLLM, Together, ... by
               pointing LLM_BASE_URL at them.
  none       - no model. The layer stores text verbatim instead of failing.
"""

from __future__ import annotations

import json
import logging
from typing import Protocol, TypeVar

import httpx
from pydantic import BaseModel

from .config import Settings, settings

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

OPENAI_BASE_URL = "https://api.openai.com/v1"


class LLM(Protocol):
    async def parse(
        self, *, model: str, system: str, user: str, schema: type[T], max_tokens: int
    ) -> T: ...


class AnthropicLLM:
    def __init__(self, api_key: str) -> None:
        from anthropic import AsyncAnthropic

        self._client = AsyncAnthropic(api_key=api_key)

    async def parse(
        self, *, model: str, system: str, user: str, schema: type[T], max_tokens: int
    ) -> T:
        response = await self._client.messages.parse(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_format=schema,
        )
        return response.parsed_output


class OpenAICompatibleLLM:
    """Chat Completions with a JSON schema response format.

    Uses plain HTTP rather than the openai SDK so that every compatible server
    works the same way and no extra dependency is needed. The schema is sent
    non-strict (pydantic schemas carry defaults that strict mode rejects) and
    also described in the prompt, since some local servers ignore
    response_format; the reply is validated with pydantic either way.
    """

    def __init__(self, base_url: str, api_key: str | None, timeout: float = 120.0) -> None:
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._timeout = timeout

    async def parse(
        self, *, model: str, system: str, user: str, schema: type[T], max_tokens: int
    ) -> T:
        json_schema = schema.model_json_schema()
        instructions = (
            f"{system}\n\nRespond with a single JSON object matching this schema, "
            f"and nothing else:\n{json.dumps(json_schema)}"
        )
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                self._url,
                headers=self._headers,
                json={
                    "model": model,
                    "max_completion_tokens": max_tokens,
                    "messages": [
                        {"role": "system", "content": instructions},
                        {"role": "user", "content": user},
                    ],
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {"name": schema.__name__.lstrip("_"), "schema": json_schema},
                    },
                },
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"] or ""
        return schema.model_validate_json(_strip_fences(content))


def _strip_fences(text: str) -> str:
    """Some models wrap JSON in ```json fences despite being asked not to."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.rsplit("```", 1)[0]
    return text.strip()


def build_llm(cfg: Settings | None = None) -> LLM | None:
    """The configured provider, or None to run without extraction."""
    cfg = cfg or settings()
    if cfg.llm_provider == "anthropic" and cfg.llm_api_key:
        return AnthropicLLM(cfg.llm_api_key)
    if cfg.llm_provider == "openai" and (cfg.llm_api_key or cfg.llm_base_url != OPENAI_BASE_URL):
        # A custom base URL with no key is a local server such as Ollama.
        return OpenAICompatibleLLM(cfg.llm_base_url, cfg.llm_api_key)
    if cfg.llm_provider != "none":
        log.warning(
            "LLM provider %r selected but no API key set; storing text verbatim "
            "with no fact extraction.",
            cfg.llm_provider,
        )
    return None
