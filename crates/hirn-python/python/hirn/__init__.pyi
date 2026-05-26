"""Type stubs for hirn's public Python API."""

from __future__ import annotations

from typing import Any, AsyncIterator, Literal, Optional

from hirn.embeddings import EmbeddingFunction

__version__: str

RecallSnapshotKind = Literal["observed", "recorded", "revision"]


class HirnError(Exception):
    """Base exception for all hirn errors."""


class NotFoundError(HirnError):
    """Raised when a record or entity is not found."""


class QueryError(HirnError):
    """Raised when a HirnQL query fails."""


class Stats:
    """Database statistics."""

    @property
    def working_count(self) -> int: ...
    @property
    def episodic_count(self) -> int: ...
    @property
    def semantic_count(self) -> int: ...
    @property
    def total_count(self) -> int: ...
    @property
    def file_size_bytes(self) -> int: ...
    def __repr__(self) -> str: ...


class RecallResult:
    """A single result from a recall operation."""

    @property
    def id(self) -> str: ...
    @property
    def layer(self) -> str: ...
    @property
    def similarity(self) -> float: ...
    @property
    def composite_score(self) -> float: ...
    @property
    def activation(self) -> float: ...
    @property
    def importance(self) -> float: ...
    @property
    def recency(self) -> float: ...
    @property
    def causal_relevance(self) -> float: ...
    @property
    def surprise(self) -> float: ...
    @property
    def source_reliability(self) -> float: ...
    @property
    def logical_memory_id(self) -> Optional[str]: ...
    @property
    def revision_id(self) -> Optional[str]: ...
    @property
    def revision_state(self) -> Optional[str]: ...
    def __repr__(self) -> str: ...


class Context:
    """Result of a think operation — assembled context for an LLM prompt."""

    @property
    def context(self) -> str: ...
    @property
    def token_count(self) -> int: ...
    @property
    def records_included(self) -> list[str]: ...
    @property
    def query_time_ms(self) -> float: ...
    def __repr__(self) -> str: ...


class QueryResult:
    """Result of a HirnQL query, returned as JSON-compatible data."""

    @property
    def type(self) -> str: ...
    @property
    def json(self) -> dict[str, Any]: ...
    def __repr__(self) -> str: ...


class WatchStream(AsyncIterator[dict[str, Any]]):
    """Async iterator for memory change events."""

    def __aiter__(self) -> WatchStream: ...
    async def __anext__(self) -> dict[str, Any]: ...
    def next_event(self, *, timeout_ms: int = 200) -> Optional[dict[str, Any]]: ...
    def cancel(self) -> None: ...
    def is_done(self) -> bool: ...


class Memory:
    """Zero-config memory API with pluggable embeddings."""

    @staticmethod
    def open(
        path: str,
        *,
        embeddings: Optional[EmbeddingFunction] = None,
        agent_id: str = "anonymous",
        token_budget: int = 4096,
        tokenizer_name: Optional[str] = None,
    ) -> Memory: ...

    def close(self) -> None: ...
    def __enter__(self) -> Memory: ...
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool: ...
    def remember(
        self,
        content: str,
        *,
        agent_id: Optional[str] = None,
        importance: float = 0.5,
    ) -> str: ...
    def batch_remember(
        self,
        contents: list[str],
        *,
        agent_id: Optional[str] = None,
        importance: float = 0.5,
    ) -> list[str]: ...
    def recall(
        self,
        query: str,
        *,
        limit: int = 10,
        threshold: Optional[float] = None,
        as_of: Optional[str] = None,
        snapshot_kind: Optional[RecallSnapshotKind] = None,
        agent_id: Optional[str] = None,
    ) -> list[RecallResult]: ...
    def think(
        self,
        query: str,
        *,
        budget: int = 4096,
        agent_id: Optional[str] = None,
    ) -> Context: ...
    def correct(
        self,
        memory_id: str,
        *,
        description: Optional[str] = None,
        confidence: Optional[float] = None,
        evidence_count: Optional[int] = None,
        reason: Optional[str] = None,
        observed_at: Optional[str] = None,
        caused_by: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> QueryResult: ...
    def supersede(
        self,
        memory_id: str,
        *,
        description: Optional[str] = None,
        confidence: Optional[float] = None,
        evidence_count: Optional[int] = None,
        reason: Optional[str] = None,
        observed_at: Optional[str] = None,
        caused_by: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> QueryResult: ...
    def merge(
        self,
        source_ids: list[str],
        target_id: str,
        *,
        description: Optional[str] = None,
        confidence: Optional[float] = None,
        evidence_count: Optional[int] = None,
        reason: Optional[str] = None,
        observed_at: Optional[str] = None,
        caused_by: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> QueryResult: ...
    def retract(
        self,
        memory_id: str,
        *,
        reason: Optional[str] = None,
        observed_at: Optional[str] = None,
        caused_by: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> QueryResult: ...
    def query(self, hirnql: str, *, agent_id: Optional[str] = None) -> QueryResult: ...
    def forget(self, memory_id: str, *, agent_id: Optional[str] = None) -> None: ...
    def stats(self) -> Stats: ...
    def watch(self, *, duration_ms: int = 1000) -> list[dict[str, Any]]: ...
    def __repr__(self) -> str: ...


class AsyncMemory:
    """Async zero-config memory API with pluggable embeddings."""

    @staticmethod
    async def open(
        path: str,
        *,
        embeddings: Optional[EmbeddingFunction] = None,
        agent_id: str = "anonymous",
        token_budget: int = 4096,
        tokenizer_name: Optional[str] = None,
    ) -> AsyncMemory: ...

    async def close(self) -> None: ...
    async def __aenter__(self) -> AsyncMemory: ...
    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool: ...
    async def remember(
        self,
        content: str,
        *,
        agent_id: Optional[str] = None,
        importance: float = 0.5,
    ) -> str: ...
    async def batch_remember(
        self,
        contents: list[str],
        *,
        agent_id: Optional[str] = None,
        importance: float = 0.5,
    ) -> list[str]: ...
    async def recall(
        self,
        query: str,
        *,
        limit: int = 10,
        threshold: Optional[float] = None,
        as_of: Optional[str] = None,
        snapshot_kind: Optional[RecallSnapshotKind] = None,
        agent_id: Optional[str] = None,
    ) -> list[RecallResult]: ...
    async def think(
        self,
        query: str,
        *,
        budget: int = 4096,
        agent_id: Optional[str] = None,
    ) -> Context: ...
    async def correct(
        self,
        memory_id: str,
        *,
        description: Optional[str] = None,
        confidence: Optional[float] = None,
        evidence_count: Optional[int] = None,
        reason: Optional[str] = None,
        observed_at: Optional[str] = None,
        caused_by: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> QueryResult: ...
    async def supersede(
        self,
        memory_id: str,
        *,
        description: Optional[str] = None,
        confidence: Optional[float] = None,
        evidence_count: Optional[int] = None,
        reason: Optional[str] = None,
        observed_at: Optional[str] = None,
        caused_by: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> QueryResult: ...
    async def merge(
        self,
        source_ids: list[str],
        target_id: str,
        *,
        description: Optional[str] = None,
        confidence: Optional[float] = None,
        evidence_count: Optional[int] = None,
        reason: Optional[str] = None,
        observed_at: Optional[str] = None,
        caused_by: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> QueryResult: ...
    async def retract(
        self,
        memory_id: str,
        *,
        reason: Optional[str] = None,
        observed_at: Optional[str] = None,
        caused_by: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> QueryResult: ...
    async def query(self, hirnql: str, *, agent_id: Optional[str] = None) -> QueryResult: ...
    async def forget(self, memory_id: str, *, agent_id: Optional[str] = None) -> None: ...
    async def watch(self) -> WatchStream: ...
    async def stats(self) -> Stats: ...
    def __repr__(self) -> str: ...
