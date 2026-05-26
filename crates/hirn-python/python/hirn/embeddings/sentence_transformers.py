"""Sentence Transformers embedding function for hirn.

Requires the ``sentence-transformers`` package::

    pip install hirn[sentence-transformers]
    # or: pip install sentence-transformers

Usage::

    from hirn.embeddings.sentence_transformers import SentenceTransformerEmbeddings

    embeddings = SentenceTransformerEmbeddings("all-MiniLM-L6-v2")
"""

from __future__ import annotations

from typing import Optional


class SentenceTransformerEmbeddings:
    """Sentence Transformers embedding function (local ONNX/PyTorch).

    Args:
        model: Model name or path (default: ``all-MiniLM-L6-v2``).
        device: Device to use (``cpu``, ``cuda``, ``mps``). Auto-detected if None.
    """

    def __init__(
        self,
        model: str = "all-MiniLM-L6-v2",
        *,
        device: Optional[str] = None,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "Sentence Transformers embeddings require the 'sentence-transformers' package. "
                "Install it with: pip install hirn[sentence-transformers]"
            ) from e

        self._model = SentenceTransformer(model, device=device)
        self._dimensions = self._model.get_sentence_embedding_dimension() or 384

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        embedding = self._model.encode(text, convert_to_numpy=True)
        return embedding.tolist()


__all__ = ["SentenceTransformerEmbeddings"]
