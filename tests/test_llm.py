"""Provider selection and the OpenAI-compatible client, with no network."""

from __future__ import annotations

import json

import httpx
import pytest
from pydantic import BaseModel

from memory_mcp import llm
from memory_mcp.embeddings import HashEmbedder, OpenAIEmbedder, build_embedder
from memory_mcp.extraction import Extractor
from memory_mcp.llm import AnthropicLLM, OpenAICompatibleLLM, _strip_fences, build_llm
from tests.conftest import make_settings


class _Answer(BaseModel):
    ok: bool
    reason: str = ""


class TestBuildLLM:
    def test_none_means_no_model(self):
        assert build_llm(make_settings(llm_provider="none")) is None

    def test_anthropic_needs_a_key(self):
        assert build_llm(make_settings(llm_provider="anthropic")) is None
        built = build_llm(make_settings(llm_provider="anthropic", llm_api_key="sk-ant-x"))
        assert isinstance(built, AnthropicLLM)

    def test_openai_with_key(self):
        built = build_llm(make_settings(llm_provider="openai", llm_api_key="sk-x"))
        assert isinstance(built, OpenAICompatibleLLM)

    def test_openai_hosted_without_key_is_disabled(self):
        assert build_llm(make_settings(llm_provider="openai")) is None

    def test_local_server_needs_no_key(self):
        cfg = make_settings(llm_provider="openai", llm_base_url="http://localhost:11434/v1")
        assert isinstance(build_llm(cfg), OpenAICompatibleLLM)

    def test_extractor_without_model_stores_verbatim(self):
        extractor = Extractor(make_settings(llm_provider="none"))
        assert extractor._client is None


class TestOpenAICompatible:
    @pytest.fixture
    def captured(self, monkeypatch):
        """Route the client's requests to a handler instead of the network."""
        seen: dict = {}
        reply = {"content": '{"ok": true, "reason": "fine"}'}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["url"] = str(request.url)
            seen["headers"] = request.headers
            seen["body"] = json.loads(request.content)
            return httpx.Response(
                200, json={"choices": [{"message": {"content": reply["content"]}}]}
            )

        real = httpx.AsyncClient
        monkeypatch.setattr(
            llm.httpx,
            "AsyncClient",
            lambda **kw: real(transport=httpx.MockTransport(handler), **kw),
        )
        seen["reply"] = reply
        return seen

    async def test_request_shape_and_parsed_result(self, captured):
        client = OpenAICompatibleLLM("http://localhost:11434/v1/", api_key=None)
        answer = await client.parse(
            model="llama3.1", system="Judge it.", user="data", schema=_Answer, max_tokens=64
        )
        assert answer == _Answer(ok=True, reason="fine")
        assert captured["url"] == "http://localhost:11434/v1/chat/completions"
        assert "authorization" not in captured["headers"]
        body = captured["body"]
        assert body["model"] == "llama3.1"
        assert body["response_format"]["type"] == "json_schema"
        assert body["messages"][0]["role"] == "system"
        assert "schema" in body["messages"][0]["content"]
        assert body["messages"][1] == {"role": "user", "content": "data"}

    async def test_sends_bearer_key(self, captured):
        client = OpenAICompatibleLLM("https://api.openai.com/v1", api_key="sk-x")
        await client.parse(model="m", system="s", user="u", schema=_Answer, max_tokens=8)
        assert captured["headers"]["authorization"] == "Bearer sk-x"

    async def test_fenced_json_is_accepted(self, captured):
        captured["reply"]["content"] = '```json\n{"ok": false}\n```'
        client = OpenAICompatibleLLM("https://api.openai.com/v1", api_key="sk-x")
        answer = await client.parse(model="m", system="s", user="u", schema=_Answer, max_tokens=8)
        assert answer.ok is False

    async def test_invalid_json_raises(self, captured):
        captured["reply"]["content"] = "not json"
        client = OpenAICompatibleLLM("https://api.openai.com/v1", api_key="sk-x")
        with pytest.raises(ValueError):
            await client.parse(model="m", system="s", user="u", schema=_Answer, max_tokens=8)


def test_strip_fences():
    assert _strip_fences('{"a": 1}') == '{"a": 1}'
    assert _strip_fences('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert _strip_fences('```\n{"a": 1}```') == '{"a": 1}'


class TestBuildEmbedder:
    def test_openai_compatible_local_server(self):
        cfg = make_settings(
            embedding_provider="openai", embedding_base_url="http://localhost:11434/v1"
        )
        assert isinstance(build_embedder(cfg), OpenAIEmbedder)

    def test_openai_hosted_without_key_falls_back(self):
        assert isinstance(build_embedder(make_settings(embedding_provider="openai")), HashEmbedder)
