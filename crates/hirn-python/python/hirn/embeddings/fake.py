"""Deterministic fake embeddings for testing.

Always available — no external dependencies required.

Usage::

    from hirn.embeddings.fake import FakeEmbeddings

    fake = FakeEmbeddings(dimensions=64)
    vec = fake.embed_query("hello")
    assert len(vec) == 64
"""

from __future__ import annotations

import hashlib
import math
import struct


class FakeEmbeddings:
    """Deterministic hash-based embeddings for testing.

    Produces consistent vectors from text input using SHA-256.
    Not semantically meaningful — use only for tests and development.

    Args:
        dimensions: Embedding vector size (default: 64).
    """

    def __init__(self, dimensions: int = 64) -> None:
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._hash_embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._hash_embed(text)

    def _hash_embed(self, text: str) -> list[float]:
        """Generate a deterministic embedding vector from text via SHA-256 expansion."""
        result: list[float] = []
        counter = 0
        while len(result) < self._dimensions:
            digest = hashlib.sha256(f"{counter}:{text}".encode("utf-8")).digest()
            # Unpack 8 floats from the 32-byte digest (4 bytes each)
            floats = struct.unpack(f"{len(digest) // 4}f", digest)
            # Replace NaN/Inf with 0.0 (some byte patterns produce non-finite floats)
            result.extend(0.0 if not math.isfinite(f) else f for f in floats)
            counter += 1
        # Truncate to exact dimensions and normalize
        result = result[: self._dimensions]
        norm = sum(x * x for x in result) ** 0.5
        if norm > 0:
            result = [x / norm for x in result]
        return result


__all__ = ["FakeEmbeddings"]
