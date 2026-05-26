"""OpenAI embedding function for hirn.

Requires the ``openai`` package::

    pip install hirn[openai]
    # or: pip install openai

Usage::

    from hirn.embeddings.openai import OpenAIEmbeddings

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    # or with explicit API key:
    embeddings = OpenAIEmbeddings(api_key="sk-...")
"""

from __future__ import annotations

from typing import Optional


class OpenAIEmbeddings:
    """OpenAI embedding function using the official ``openai`` Python SDK.

    Args:
        model: Model name (default: ``text-embedding-3-small``).
        api_key: OpenAI API key. Falls back to ``OPENAI_API_KEY`` env var.
        dimensions: Override output dimensions (requires model support).
        max_batch_size: Maximum texts per API call (default: 2048).
    """

    _DIMENSION_MAP = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        *,
        api_key: Optional[str] = None,
        dimensions: Optional[int] = None,
        max_batch_size: int = 2048,
    ) -> None:
        try:
            import openai
        except ImportError as e:
            raise ImportError(
                "OpenAI embeddings require the 'openai' package. "
                "Install it with: pip install hirn[openai]"
            ) from e

        self._model = model
        self._dimensions = dimensions or self._DIMENSION_MAP.get(model, 1536)
        self._client = openai.OpenAI(api_key=api_key)
        self._max_batch_size = max_batch_size

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if len(texts) <= self._max_batch_size:
            return self._embed_batch(texts)
        # Chunk large inputs to stay within API limits
        result: list[list[float]] = []
        for i in range(0, len(texts), self._max_batch_size):
            chunk = texts[i : i + self._max_batch_size]
            result.extend(self._embed_batch(chunk))
        return result

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(
            input=texts,
            model=self._model,
        )
        if len(response.data) != len(texts):
            raise RuntimeError(
                f"OpenAI returned {len(response.data)} embeddings for {len(texts)} texts"
            )
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_batch([text])[0]


__all__ = ["OpenAIEmbeddings"]
