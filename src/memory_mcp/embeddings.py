"""Embedding providers behind one interface.

`hash` is a deterministic offline fallback: it makes the whole layer runnable
in tests and on a plane, at the cost of real semantic similarity.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from typing import Protocol

import httpx

from .config import Settings, settings
from .llm import OPENAI_BASE_URL

log = logging.getLogger(__name__)

_TOKEN = re.compile(r"[a-z0-9']+")


class Embedder(Protocol):
    dim: int

    async def embed(self, texts: list[str], *, query: bool = False) -> list[list[float]]: ...


def _l2(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec] if norm else vec


class HashEmbedder:
    """Hashed bag-of-words. Deterministic, no network, weak semantics."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    #: Slots written per token. One slot means a single-word input occupies a
    #: single dimension, and any two words sharing that dimension come back
    #: perfectly similar - enough to merge unrelated entities. Three
    #: independent slots make that vanishingly unlikely.
    SLOTS = 3

    async def embed(self, texts: list[str], *, query: bool = False) -> list[list[float]]:
        out = []
        for text in texts:
            vec = [0.0] * self.dim
            for token in _TOKEN.findall(text.lower()):
                for slot in range(self.SLOTS):
                    digest = hashlib.blake2b(
                        token.encode(), digest_size=8, salt=str(slot).encode()
                    ).digest()
                    idx = int.from_bytes(digest[:4], "big") % self.dim
                    sign = 1.0 if digest[4] % 2 else -1.0
                    vec[idx] += sign
            out.append(_l2(vec))
        return out


class VoyageEmbedder:
    """Voyage AI - strong retrieval quality, asymmetric query/document types."""

    def __init__(self, api_key: str, model: str, dim: int) -> None:
        self._key = api_key
        self._model = model
        self.dim = dim

    async def embed(self, texts: list[str], *, query: bool = False) -> list[list[float]]:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                "https://api.voyageai.com/v1/embeddings",
                headers={"Authorization": f"Bearer {self._key}"},
                json={
                    "input": texts,
                    "model": self._model,
                    "input_type": "query" if query else "document",
                },
            )
            resp.raise_for_status()
            data = resp.json()["data"]
        return [item["embedding"] for item in sorted(data, key=lambda d: d["index"])]


class OpenAIEmbedder:
    """OpenAI, or any server exposing the same /embeddings endpoint (Ollama, ...)."""

    def __init__(
        self, api_key: str | None, model: str, dim: int, base_url: str = OPENAI_BASE_URL
    ) -> None:
        self._url = base_url.rstrip("/") + "/embeddings"
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._model = model
        self.dim = dim

    async def embed(self, texts: list[str], *, query: bool = False) -> list[list[float]]:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                self._url,
                headers=self._headers,
                json={"input": texts, "model": self._model, "dimensions": self.dim},
            )
            resp.raise_for_status()
            data = resp.json()["data"]
        return [item["embedding"] for item in sorted(data, key=lambda d: d["index"])]


def build_embedder(cfg: Settings | None = None) -> Embedder:
    cfg = cfg or settings()
    if cfg.embedding_provider == "voyage" and cfg.voyage_api_key:
        return VoyageEmbedder(cfg.voyage_api_key, cfg.embedding_model, cfg.embedding_dim)
    if cfg.embedding_provider == "openai" and (
        cfg.openai_api_key or cfg.embedding_base_url != OPENAI_BASE_URL
    ):
        # A custom base URL with no key is a local server such as Ollama.
        return OpenAIEmbedder(
            cfg.openai_api_key, cfg.embedding_model, cfg.embedding_dim, cfg.embedding_base_url
        )

    # No key configured - stay usable rather than failing at import time, but
    # keep the dimension the collection was built with. A fallback that
    # silently changed dimension would make every vector write fail instead.
    if cfg.embedding_provider != "hash":
        log.warning(
            "%s selected but no API key set; falling back to the offline hash "
            "embedder at dim=%d. Recall quality will be poor.",
            cfg.embedding_provider,
            cfg.embedding_dim,
        )
    return HashEmbedder(cfg.embedding_dim)
