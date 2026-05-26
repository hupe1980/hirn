"""Ollama embedding function for hirn.

Requires the ``ollama`` package::

    pip install hirn[ollama]
    # or: pip install ollama

Usage::

    from hirn.embeddings.ollama import OllamaEmbeddings

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
"""

from __future__ import annotations

from typing import Optional


class OllamaEmbeddings:
    """Ollama embedding function using the official ``ollama`` Python SDK.

    Args:
        model: Model name (default: ``nomic-embed-text``).
        host: Ollama server URL. Falls back to ``OLLAMA_HOST`` env var,
              then ``http://localhost:11434``.
        dimensions: Embedding dimensions (default: 768).
    """

    def __init__(
        self,
        model: str = "nomic-embed-text",
        *,
        host: Optional[str] = None,
        dimensions: int = 768,
    ) -> None:
        try:
            import ollama
        except ImportError as e:
            raise ImportError(
                "Ollama embeddings require the 'ollama' package. "
                "Install it with: pip install hirn[ollama]"
            ) from e

        import os

        self._model = model
        self._dimensions = dimensions
        self._host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self._client = ollama.Client(host=self._host)

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embed(model=self._model, input=texts)
        if not response.embeddings or len(response.embeddings) != len(texts):
            raise RuntimeError(
                f"Ollama returned {len(response.embeddings) if response.embeddings else 0} "
                f"embeddings for {len(texts)} texts"
            )
        return response.embeddings

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


__all__ = ["OllamaEmbeddings"]
