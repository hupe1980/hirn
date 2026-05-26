"""Pluggable embedding functions for hirn.

Use one of the built-in implementations or bring your own by implementing
the :class:`EmbeddingFunction` protocol.

Built-in implementations (install the optional extra to use):

- :class:`~hirn.embeddings.openai.OpenAIEmbeddings` — ``pip install hirn[openai]``
- :class:`~hirn.embeddings.ollama.OllamaEmbeddings` — ``pip install hirn[ollama]``
- :class:`~hirn.embeddings.sentence_transformers.SentenceTransformerEmbeddings` — ``pip install hirn[sentence-transformers]``
- :class:`~hirn.embeddings.fake.FakeEmbeddings` — always available (deterministic hash, for testing)
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingFunction(Protocol):
    """Protocol that all embedding implementations must satisfy.

    Implement this protocol with your preferred embedding provider::

        class MyEmbeddings:
            @property
            def dimensions(self) -> int:
                return 768

            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                return [my_embed(t) for t in texts]

            def embed_query(self, text: str) -> list[float]:
                return my_embed(text)
    """

    @property
    def dimensions(self) -> int:
        """Return the dimensionality of the embedding vectors."""
        ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of documents.

        Args:
            texts: List of document strings to embed.

        Returns:
            List of embedding vectors (one per document).
        """
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string.

        For asymmetric models (e.g. E5, BGE), this may use a different
        prompt/prefix than :meth:`embed_documents`.

        Args:
            text: Query string to embed.

        Returns:
            Embedding vector.
        """
        ...


def _detect_embeddings() -> EmbeddingFunction | None:
    """Auto-detect an embedding function from the environment.

    Checks (in order):
    1. ``OPENAI_API_KEY`` → :class:`OpenAIEmbeddings`
    2. ``OLLAMA_HOST`` → :class:`OllamaEmbeddings`
    3. ``sentence_transformers`` importable → :class:`SentenceTransformerEmbeddings`

    Returns ``None`` if no provider is detected.
    """
    import importlib
    import logging
    import os

    log = logging.getLogger("hirn")

    if os.environ.get("OPENAI_API_KEY"):
        try:
            from hirn.embeddings.openai import OpenAIEmbeddings

            log.debug("auto-detected OpenAI embeddings via OPENAI_API_KEY")
            return OpenAIEmbeddings()
        except ImportError:
            log.debug("OPENAI_API_KEY set but 'openai' package not installed")

    if os.environ.get("OLLAMA_HOST"):
        try:
            from hirn.embeddings.ollama import OllamaEmbeddings

            log.debug("auto-detected Ollama embeddings via OLLAMA_HOST")
            return OllamaEmbeddings()
        except ImportError:
            log.debug("OLLAMA_HOST set but 'ollama' package not installed")

    if importlib.util.find_spec("sentence_transformers") is not None:
        try:
            from hirn.embeddings.sentence_transformers import SentenceTransformerEmbeddings

            log.debug("auto-detected SentenceTransformer embeddings")
            return SentenceTransformerEmbeddings()
        except ImportError:
            log.debug("sentence_transformers found but failed to import")

    log.debug("no embedding provider detected; will fall back to FakeEmbeddings")
    return None


__all__ = ["EmbeddingFunction", "_detect_embeddings"]
