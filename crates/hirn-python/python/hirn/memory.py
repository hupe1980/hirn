"""High-level Memory API with pluggable embeddings.

This module provides the zero-config ``Memory`` and ``AsyncMemory`` classes
that compose the Rust native bridge with a Python-side
:class:`~hirn.embeddings.EmbeddingFunction`.

Usage::

    from hirn import Memory

    mem = Memory.open("./brain")  # auto-detects embeddings from env
    mem.remember("User prefers dark mode")
    ctx = mem.think("What are the user's preferences?", budget=2048)
    print(ctx.context)
"""

from __future__ import annotations

import asyncio
import logging
import math
from typing import TYPE_CHECKING, Any

from hirn._hirn import (
    AsyncHirnBridge,
    Context,
    HirnBridge,
    QueryResult,
    RecallResult,
    Stats,
    WatchStream,
)
from hirn.embeddings import EmbeddingFunction, _detect_embeddings

if TYPE_CHECKING:
    pass

logger = logging.getLogger("hirn")


def _is_already_registered_error(error: Exception) -> bool:
    return "already registered" in str(error).lower()


def _require_non_empty_string(value: str, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string, got {type(value).__name__}")
    if not value.strip():
        raise ValueError(f"{name} must not be empty or whitespace-only")
    return value


def _quote_hirnql_string(value: str) -> str:
    escaped = (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\t", "\\t")
        .replace("\r", "\\r")
    )
    return f'"{escaped}"'


def _append_optional_string_clause(
    parts: list[str],
    clause: str,
    value: str | None,
    *,
    name: str,
) -> None:
    if value is None:
        return
    parts.extend([clause, _quote_hirnql_string(_require_non_empty_string(value, name=name))])


def _format_semantic_assignments(
    *,
    description: str | None = None,
    confidence: float | None = None,
    evidence_count: int | None = None,
    require_any: bool,
) -> list[str]:
    assignments: list[str] = []

    if description is not None:
        assignments.append(
            "description = "
            + _quote_hirnql_string(
                _require_non_empty_string(description, name="description")
            )
        )

    if confidence is not None:
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise TypeError("confidence must be a finite number")
        confidence_value = float(confidence)
        if not math.isfinite(confidence_value):
            raise ValueError("confidence must be a finite number")
        assignments.append(f"confidence = {confidence_value!r}")

    if evidence_count is not None:
        if isinstance(evidence_count, bool) or not isinstance(evidence_count, int):
            raise TypeError("evidence_count must be a non-negative integer")
        if evidence_count < 0:
            raise ValueError("evidence_count must be non-negative")
        assignments.append(f"evidence_count = {evidence_count}")

    if require_any and not assignments:
        raise ValueError(
            "at least one semantic update field must be provided: "
            "description, confidence, or evidence_count"
        )

    return assignments


def _build_semantic_edit_query(
    verb: str,
    memory_id: str,
    *,
    description: str | None = None,
    confidence: float | None = None,
    evidence_count: int | None = None,
    reason: str | None = None,
    observed_at: str | None = None,
    caused_by: str | None = None,
) -> str:
    assignments = _format_semantic_assignments(
        description=description,
        confidence=confidence,
        evidence_count=evidence_count,
        require_any=True,
    )
    parts = [
        verb,
        _quote_hirnql_string(_require_non_empty_string(memory_id, name="memory_id")),
        "SET",
        ", ".join(assignments),
    ]
    _append_optional_string_clause(parts, "REASON", reason, name="reason")
    _append_optional_string_clause(
        parts, "OBSERVED AT", observed_at, name="observed_at"
    )
    _append_optional_string_clause(parts, "CAUSED BY", caused_by, name="caused_by")
    return " ".join(parts)


def _build_semantic_merge_query(
    source_ids: list[str],
    target_id: str,
    *,
    description: str | None = None,
    confidence: float | None = None,
    evidence_count: int | None = None,
    reason: str | None = None,
    observed_at: str | None = None,
    caused_by: str | None = None,
) -> str:
    if not source_ids:
        raise ValueError("source_ids must contain at least one memory ID")

    quoted_sources = [
        _quote_hirnql_string(_require_non_empty_string(source_id, name="source_ids[]"))
        for source_id in source_ids
    ]
    parts = [
        "MERGE",
        "MEMORY",
        ", ".join(quoted_sources),
        "INTO",
        _quote_hirnql_string(_require_non_empty_string(target_id, name="target_id")),
    ]

    assignments = _format_semantic_assignments(
        description=description,
        confidence=confidence,
        evidence_count=evidence_count,
        require_any=False,
    )
    if assignments:
        parts.extend(["SET", ", ".join(assignments)])

    _append_optional_string_clause(parts, "REASON", reason, name="reason")
    _append_optional_string_clause(
        parts, "OBSERVED AT", observed_at, name="observed_at"
    )
    _append_optional_string_clause(parts, "CAUSED BY", caused_by, name="caused_by")
    return " ".join(parts)


def _build_semantic_retract_query(
    memory_id: str,
    *,
    reason: str | None = None,
    observed_at: str | None = None,
    caused_by: str | None = None,
) -> str:
    parts = [
        "RETRACT",
        _quote_hirnql_string(_require_non_empty_string(memory_id, name="memory_id")),
    ]
    _append_optional_string_clause(parts, "REASON", reason, name="reason")
    _append_optional_string_clause(
        parts, "OBSERVED AT", observed_at, name="observed_at"
    )
    _append_optional_string_clause(parts, "CAUSED BY", caused_by, name="caused_by")
    return " ".join(parts)


class Memory:
    """Zero-config memory API with automatic embedding.

    Combines a Rust-backed native bridge with a Python-side
    :class:`~hirn.embeddings.EmbeddingFunction` so that ``remember`` /
    ``recall`` / ``think`` accept plain text.

    Args:
        hirn: An open native bridge instance.
        embeddings: An :class:`~hirn.embeddings.EmbeddingFunction` implementation.
        agent_id: Default agent identifier (default: ``"anonymous"``).
    """

    def __init__(
        self,
        hirn: HirnBridge,
        embeddings: EmbeddingFunction,
        *,
        agent_id: str = "anonymous",
    ) -> None:
        self._hirn: HirnBridge | None = hirn
        self._embeddings = embeddings
        self._agent_id = agent_id
        self._registered_agents: set[str] = set()

    # ── Factory ───────────────────────────────────────────────

    @staticmethod
    def open(
        path: str,
        *,
        embeddings: EmbeddingFunction | None = None,
        agent_id: str = "anonymous",
        token_budget: int = 4096,
        tokenizer_name: str | None = None,
    ) -> Memory:
        """Open (or create) a brain at the given path.

        Embedding provider resolution order:

        1. Explicit ``embeddings`` argument
        2. Auto-detect from environment (``OPENAI_API_KEY``, ``OLLAMA_HOST``)
        3. Fall back to :class:`~hirn.embeddings.fake.FakeEmbeddings`

        Args:
            path: File system path to the brain directory.
            embeddings: Optional embedding function. Auto-detected if *None*.
            agent_id: Default agent identifier (default: ``"anonymous"``).
            token_budget: Token budget for context assembly (default: 4096).
            tokenizer_name: Optional Rust tokenizer registry name. This selects
                an existing Rust tokenizer without routing token budgeting
                through Python.
        """
        if embeddings is None:
            embeddings = _detect_embeddings()
        if embeddings is None:
            from hirn.embeddings.fake import FakeEmbeddings

            embeddings = FakeEmbeddings()

        hirn = HirnBridge.open(
            path,
            embedding_dimensions=embeddings.dimensions,
            token_budget=token_budget,
            tokenizer_name=tokenizer_name,
        )
        return Memory(hirn, embeddings, agent_id=agent_id)

    # ── Lifecycle ─────────────────────────────────────────────

    def close(self) -> None:
        """Close the memory database."""
        if self._hirn is not None:
            self._hirn.close()
            self._hirn = None

    def __enter__(self) -> Memory:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        self.close()
        return False

    # ── Helpers ───────────────────────────────────────────────

    def _check_open(self) -> HirnBridge:
        if self._hirn is None:
            raise RuntimeError("memory is closed")
        return self._hirn

    def _effective_agent(self, per_call: str | None) -> str:
        return per_call if per_call is not None else self._agent_id

    def _ensure_agent(self, agent_id: str) -> None:
        """Register an agent if not already registered in this session."""
        if agent_id in self._registered_agents:
            return
        h = self._check_open()
        try:
            h.register_agent(agent_id, agent_id)
        except Exception as error:
            if not _is_already_registered_error(error):
                raise
        self._registered_agents.add(agent_id)

    # ── Core Operations ──────────────────────────────────────

    def remember(
        self,
        content: str,
        *,
        agent_id: str | None = None,
        importance: float = 0.5,
    ) -> str:
        """Store a text memory with automatic embedding.

        Args:
            content: Text content of the memory.
            agent_id: Optional per-call agent identifier.
            importance: Importance score 0.0–1.0 (default: 0.5).

        Returns:
            The ULID string of the new memory.

        Raises:
            TypeError: If *content* is not a string.
            ValueError: If *content* is empty.
        """
        if not isinstance(content, str):
            raise TypeError(f"content must be a string, got {type(content).__name__}")
        if not content.strip():
            raise ValueError("content must not be empty or whitespace-only")
        h = self._check_open()
        aid = self._effective_agent(agent_id)
        self._ensure_agent(aid)
        embedding = self._embeddings.embed_documents([content])[0]
        logger.debug("remember agent=%s len=%d dims=%d", aid, len(content), len(embedding))
        return h.remember(aid, content, embedding=embedding, importance=importance)

    def batch_remember(
        self,
        contents: list[str],
        *,
        agent_id: str | None = None,
        importance: float = 0.5,
    ) -> list[str]:
        """Store multiple memories with a single batch embedding call.

        This is significantly more efficient than calling :meth:`remember`
        in a loop because embedding API calls are batched.

        Args:
            contents: List of text contents to remember.
            agent_id: Optional per-call agent identifier.
            importance: Importance score 0.0–1.0 (default: 0.5).

        Returns:
            List of ULID strings for the new memories (same order as *contents*).

        Raises:
            TypeError: If any element of *contents* is not a string.
            ValueError: If any element of *contents* is empty.
        """
        if not contents:
            return []
        for i, c in enumerate(contents):
            if not isinstance(c, str):
                raise TypeError(
                    f"contents[{i}] must be a string, got {type(c).__name__}"
                )
            if not c.strip():
                raise ValueError(f"contents[{i}] must not be empty or whitespace-only")
        h = self._check_open()
        aid = self._effective_agent(agent_id)
        self._ensure_agent(aid)
        embeddings = self._embeddings.embed_documents(contents)
        logger.debug("batch_remember agent=%s count=%d", aid, len(contents))
        return [
            h.remember(aid, content, embedding=emb, importance=importance)
            for content, emb in zip(contents, embeddings)
        ]

    def recall(
        self,
        query: str,
        *,
        limit: int = 10,
        threshold: float | None = None,
        as_of: str | None = None,
        snapshot_kind: str | None = None,
        agent_id: str | None = None,
    ) -> list[RecallResult]:
        """Recall memories relevant to a query.

        Args:
            query: Natural language query.
            limit: Maximum number of results (default: 10).
            threshold: Optional minimum similarity threshold.
            as_of: Optional historical snapshot value (``YYYY-MM-DD``, RFC 3339,
                or a revision ULID). Defaults to current-state recall.
            snapshot_kind: Optional snapshot selector: ``observed`` (default when
                ``as_of`` is provided), ``recorded``, or ``revision``.
            agent_id: Optional per-call agent identifier.

        Returns:
            List of :class:`RecallResult` objects. Semantic results include
            ``logical_memory_id``, ``revision_id``, and ``revision_state``.
        """
        h = self._check_open()
        aid = self._effective_agent(agent_id)
        self._ensure_agent(aid)
        query_vec = self._embeddings.embed_query(query)
        return h.recall(
            aid,
            query_vec,
            limit=limit,
            threshold=threshold,
            as_of=as_of,
            snapshot_kind=snapshot_kind,
        )

    def think(
        self,
        query: str,
        *,
        budget: int = 4096,
        agent_id: str | None = None,
    ) -> Context:
        """Assemble optimal LLM context for a query.

        Args:
            query: Natural language query.
            budget: Token budget (default: 4096).
            agent_id: Optional per-call agent identifier.

        Returns:
            :class:`Context` with the assembled context string.
        """
        h = self._check_open()
        aid = self._effective_agent(agent_id)
        self._ensure_agent(aid)
        query_vec = self._embeddings.embed_query(query)
        return h.think(aid, query_vec, budget=budget)

    def correct(
        self,
        memory_id: str,
        *,
        description: str | None = None,
        confidence: float | None = None,
        evidence_count: int | None = None,
        reason: str | None = None,
        observed_at: str | None = None,
        caused_by: str | None = None,
        agent_id: str | None = None,
    ) -> QueryResult:
        """Append a correction revision for a semantic memory."""
        hirnql = _build_semantic_edit_query(
            "CORRECT",
            memory_id,
            description=description,
            confidence=confidence,
            evidence_count=evidence_count,
            reason=reason,
            observed_at=observed_at,
            caused_by=caused_by,
        )
        return self.query(hirnql, agent_id=agent_id)

    def supersede(
        self,
        memory_id: str,
        *,
        description: str | None = None,
        confidence: float | None = None,
        evidence_count: int | None = None,
        reason: str | None = None,
        observed_at: str | None = None,
        caused_by: str | None = None,
        agent_id: str | None = None,
    ) -> QueryResult:
        """Append a new authoritative semantic revision."""
        hirnql = _build_semantic_edit_query(
            "SUPERSEDE",
            memory_id,
            description=description,
            confidence=confidence,
            evidence_count=evidence_count,
            reason=reason,
            observed_at=observed_at,
            caused_by=caused_by,
        )
        return self.query(hirnql, agent_id=agent_id)

    def merge(
        self,
        source_ids: list[str],
        target_id: str,
        *,
        description: str | None = None,
        confidence: float | None = None,
        evidence_count: int | None = None,
        reason: str | None = None,
        observed_at: str | None = None,
        caused_by: str | None = None,
        agent_id: str | None = None,
    ) -> QueryResult:
        """Merge one or more semantic memories into a canonical target."""
        hirnql = _build_semantic_merge_query(
            source_ids,
            target_id,
            description=description,
            confidence=confidence,
            evidence_count=evidence_count,
            reason=reason,
            observed_at=observed_at,
            caused_by=caused_by,
        )
        return self.query(hirnql, agent_id=agent_id)

    def retract(
        self,
        memory_id: str,
        *,
        reason: str | None = None,
        observed_at: str | None = None,
        caused_by: str | None = None,
        agent_id: str | None = None,
    ) -> QueryResult:
        """Append a tombstone revision for a semantic memory."""
        hirnql = _build_semantic_retract_query(
            memory_id,
            reason=reason,
            observed_at=observed_at,
            caused_by=caused_by,
        )
        return self.query(hirnql, agent_id=agent_id)

    def query(
        self,
        hirnql: str,
        *,
        agent_id: str | None = None,
    ) -> QueryResult:
        """Execute a HirnQL query string.

        Args:
            hirnql: HirnQL query string.
            agent_id: Optional per-call agent identifier.

        Returns:
            :class:`QueryResult` with the result as a JSON-accessible dict.

        Use raw HirnQL here when you need exact clause control, revision-aware
        statements not covered by the convenience helpers, or plan/explain
        surfaces. ``correct()``, ``supersede()``, ``merge()``, and
        ``retract()`` cover the common semantic edit flows directly.
        """
        h = self._check_open()
        aid = self._effective_agent(agent_id)
        self._ensure_agent(aid)
        return h.execute(aid, hirnql)

    def forget(
        self,
        memory_id: str,
        *,
        agent_id: str | None = None,
    ) -> None:
        """Forget a memory by its ULID string.

        Args:
            memory_id: ULID string of the memory to forget.
            agent_id: Optional per-call agent identifier.
        """
        h = self._check_open()
        aid = self._effective_agent(agent_id)
        self._ensure_agent(aid)
        h.forget(aid, memory_id)

    def stats(self) -> Stats:
        """Get database statistics."""
        h = self._check_open()
        return h.stats()

    def watch(self, *, duration_ms: int = 1000) -> list[dict[str, Any]]:
        """Watch for memory events.

        Subscribes to the database event stream and collects events for the
        specified duration.

        Args:
            duration_ms: How long to listen for events in milliseconds (default: 1000).

        Returns:
            List of dicts, each with ``type`` and event-specific fields.
        """
        h = self._check_open()
        return h.watch(duration_ms=duration_ms)

    def __repr__(self) -> str:
        if self._hirn is not None:
            emb = type(self._embeddings).__name__
            return f"Memory(open, embeddings={emb}, dims={self._embeddings.dimensions})"
        return "Memory(closed)"


class AsyncMemory:
    """Async zero-config memory API with automatic embedding.

    Combines a Rust-backed native bridge with a Python-side
    :class:`~hirn.embeddings.EmbeddingFunction`.

    Usage::

        import asyncio
        from hirn import AsyncMemory

        async def main():
            mem = await AsyncMemory.open("./brain")
            await mem.remember("User prefers dark mode")
            ctx = await mem.think("preferences?", budget=2048)
            print(ctx.context)

        asyncio.run(main())
    """

    def __init__(
        self,
        hirn: AsyncHirnBridge,
        embeddings: EmbeddingFunction,
        *,
        agent_id: str = "anonymous",
    ) -> None:
        self._hirn: AsyncHirnBridge | None = hirn
        self._embeddings = embeddings
        self._agent_id = agent_id
        self._registered_agents: set[str] = set()

    # ── Factory ───────────────────────────────────────────────

    @staticmethod
    async def open(
        path: str,
        *,
        embeddings: EmbeddingFunction | None = None,
        agent_id: str = "anonymous",
        token_budget: int = 4096,
        tokenizer_name: str | None = None,
    ) -> AsyncMemory:
        """Open (or create) a brain at the given path asynchronously.

        Embedding provider resolution order:

        1. Explicit ``embeddings`` argument
        2. Auto-detect from environment (``OPENAI_API_KEY``, ``OLLAMA_HOST``)
        3. Fall back to :class:`~hirn.embeddings.fake.FakeEmbeddings`

        Args:
            path: File system path to the brain directory.
            embeddings: Optional embedding function. Auto-detected if *None*.
            agent_id: Default agent identifier (default: ``"anonymous"``).
            token_budget: Token budget for context assembly (default: 4096).
            tokenizer_name: Optional Rust tokenizer registry name. This selects
                an existing Rust tokenizer without routing token budgeting
                through Python.
        """
        if embeddings is None:
            embeddings = _detect_embeddings()
        if embeddings is None:
            from hirn.embeddings.fake import FakeEmbeddings

            embeddings = FakeEmbeddings()

        hirn = await AsyncHirnBridge.open(
            path,
            embedding_dimensions=embeddings.dimensions,
            token_budget=token_budget,
            tokenizer_name=tokenizer_name,
        )
        return AsyncMemory(hirn, embeddings, agent_id=agent_id)

    # ── Lifecycle ─────────────────────────────────────────────

    async def close(self) -> None:
        """Close the memory database."""
        if self._hirn is not None:
            await self._hirn.close()
            self._hirn = None

    async def __aenter__(self) -> AsyncMemory:
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        await self.close()
        return False

    # ── Helpers ───────────────────────────────────────────────

    def _check_open(self) -> AsyncHirnBridge:
        if self._hirn is None:
            raise RuntimeError("memory is closed")
        return self._hirn

    def _effective_agent(self, per_call: str | None) -> str:
        return per_call if per_call is not None else self._agent_id

    async def _ensure_agent(self, agent_id: str) -> None:
        """Register an agent if not already registered in this session."""
        if agent_id in self._registered_agents:
            return
        h = self._check_open()
        try:
            await h.register_agent(agent_id, agent_id)
        except Exception as error:
            if not _is_already_registered_error(error):
                raise
        self._registered_agents.add(agent_id)

    async def _embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed documents, offloading to a thread if the embedder is synchronous."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self._embeddings.embed_documents, texts
        )

    async def _embed_query(self, text: str) -> list[float]:
        """Embed a query, offloading to a thread if the embedder is synchronous."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._embeddings.embed_query, text)

    # ── Core Operations ──────────────────────────────────────

    async def remember(
        self,
        content: str,
        *,
        agent_id: str | None = None,
        importance: float = 0.5,
    ) -> str:
        """Store a text memory with automatic embedding (async).

        Args:
            content: Text content of the memory.
            agent_id: Optional per-call agent identifier.
            importance: Importance score 0.0–1.0 (default: 0.5).

        Returns:
            The ULID string of the new memory.

        Raises:
            TypeError: If *content* is not a string.
            ValueError: If *content* is empty.
        """
        if not isinstance(content, str):
            raise TypeError(f"content must be a string, got {type(content).__name__}")
        if not content.strip():
            raise ValueError("content must not be empty or whitespace-only")
        h = self._check_open()
        aid = self._effective_agent(agent_id)
        await self._ensure_agent(aid)
        embeddings = await self._embed_documents([content])
        logger.debug("remember agent=%s len=%d dims=%d", aid, len(content), len(embeddings[0]))
        return await h.remember(aid, content, embedding=embeddings[0], importance=importance)

    async def batch_remember(
        self,
        contents: list[str],
        *,
        agent_id: str | None = None,
        importance: float = 0.5,
    ) -> list[str]:
        """Store multiple memories with a single batch embedding call (async).

        This is significantly more efficient than calling :meth:`remember`
        in a loop because embedding API calls are batched.

        Args:
            contents: List of text contents to remember.
            agent_id: Optional per-call agent identifier.
            importance: Importance score 0.0–1.0 (default: 0.5).

        Returns:
            List of ULID strings for the new memories (same order as *contents*).

        Raises:
            TypeError: If any element of *contents* is not a string.
            ValueError: If any element of *contents* is empty.
        """
        if not contents:
            return []
        for i, c in enumerate(contents):
            if not isinstance(c, str):
                raise TypeError(
                    f"contents[{i}] must be a string, got {type(c).__name__}"
                )
            if not c.strip():
                raise ValueError(f"contents[{i}] must not be empty or whitespace-only")
        h = self._check_open()
        aid = self._effective_agent(agent_id)
        await self._ensure_agent(aid)
        embeddings = await self._embed_documents(contents)
        logger.debug("batch_remember agent=%s count=%d", aid, len(contents))
        ids: list[str] = []
        for content, emb in zip(contents, embeddings):
            mem_id = await h.remember(aid, content, embedding=emb, importance=importance)
            ids.append(mem_id)
        return ids

    async def recall(
        self,
        query: str,
        *,
        limit: int = 10,
        threshold: float | None = None,
        as_of: str | None = None,
        snapshot_kind: str | None = None,
        agent_id: str | None = None,
    ) -> list[RecallResult]:
        """Recall memories relevant to a query (async).

        Args:
            query: Natural language query.
            limit: Maximum number of results (default: 10).
            threshold: Optional minimum similarity threshold.
            as_of: Optional historical snapshot value (``YYYY-MM-DD``, RFC 3339,
                or a revision ULID). Defaults to current-state recall.
            snapshot_kind: Optional snapshot selector: ``observed`` (default when
                ``as_of`` is provided), ``recorded``, or ``revision``.
            agent_id: Optional per-call agent identifier.

        Returns:
            List of :class:`RecallResult` objects. Semantic results include
            ``logical_memory_id``, ``revision_id``, and ``revision_state``.
        """
        h = self._check_open()
        aid = self._effective_agent(agent_id)
        await self._ensure_agent(aid)
        query_vec = await self._embed_query(query)
        return await h.recall(
            aid,
            query_vec,
            limit=limit,
            threshold=threshold,
            as_of=as_of,
            snapshot_kind=snapshot_kind,
        )

    async def think(
        self,
        query: str,
        *,
        budget: int = 4096,
        agent_id: str | None = None,
    ) -> Context:
        """Assemble optimal LLM context for a query (async).

        Args:
            query: Natural language query.
            budget: Token budget (default: 4096).
            agent_id: Optional per-call agent identifier.

        Returns:
            :class:`Context` with the assembled context string.
        """
        h = self._check_open()
        aid = self._effective_agent(agent_id)
        await self._ensure_agent(aid)
        query_vec = await self._embed_query(query)
        return await h.think(aid, query_vec, budget=budget)

    async def correct(
        self,
        memory_id: str,
        *,
        description: str | None = None,
        confidence: float | None = None,
        evidence_count: int | None = None,
        reason: str | None = None,
        observed_at: str | None = None,
        caused_by: str | None = None,
        agent_id: str | None = None,
    ) -> QueryResult:
        """Append a correction revision for a semantic memory (async)."""
        hirnql = _build_semantic_edit_query(
            "CORRECT",
            memory_id,
            description=description,
            confidence=confidence,
            evidence_count=evidence_count,
            reason=reason,
            observed_at=observed_at,
            caused_by=caused_by,
        )
        return await self.query(hirnql, agent_id=agent_id)

    async def supersede(
        self,
        memory_id: str,
        *,
        description: str | None = None,
        confidence: float | None = None,
        evidence_count: int | None = None,
        reason: str | None = None,
        observed_at: str | None = None,
        caused_by: str | None = None,
        agent_id: str | None = None,
    ) -> QueryResult:
        """Append a new authoritative semantic revision (async)."""
        hirnql = _build_semantic_edit_query(
            "SUPERSEDE",
            memory_id,
            description=description,
            confidence=confidence,
            evidence_count=evidence_count,
            reason=reason,
            observed_at=observed_at,
            caused_by=caused_by,
        )
        return await self.query(hirnql, agent_id=agent_id)

    async def merge(
        self,
        source_ids: list[str],
        target_id: str,
        *,
        description: str | None = None,
        confidence: float | None = None,
        evidence_count: int | None = None,
        reason: str | None = None,
        observed_at: str | None = None,
        caused_by: str | None = None,
        agent_id: str | None = None,
    ) -> QueryResult:
        """Merge one or more semantic memories into a canonical target (async)."""
        hirnql = _build_semantic_merge_query(
            source_ids,
            target_id,
            description=description,
            confidence=confidence,
            evidence_count=evidence_count,
            reason=reason,
            observed_at=observed_at,
            caused_by=caused_by,
        )
        return await self.query(hirnql, agent_id=agent_id)

    async def retract(
        self,
        memory_id: str,
        *,
        reason: str | None = None,
        observed_at: str | None = None,
        caused_by: str | None = None,
        agent_id: str | None = None,
    ) -> QueryResult:
        """Append a tombstone revision for a semantic memory (async)."""
        hirnql = _build_semantic_retract_query(
            memory_id,
            reason=reason,
            observed_at=observed_at,
            caused_by=caused_by,
        )
        return await self.query(hirnql, agent_id=agent_id)

    async def query(
        self,
        hirnql: str,
        *,
        agent_id: str | None = None,
    ) -> QueryResult:
        """Execute a HirnQL query string (async).

        Args:
            hirnql: HirnQL query string.
            agent_id: Optional per-call agent identifier.

        Returns:
            :class:`QueryResult` with the result as a JSON-accessible dict.

        Use raw HirnQL here when you need exact clause control, revision-aware
        statements not covered by the convenience helpers, or plan/explain
        surfaces. ``correct()``, ``supersede()``, ``merge()``, and
        ``retract()`` cover the common semantic edit flows directly.
        """
        h = self._check_open()
        aid = self._effective_agent(agent_id)
        await self._ensure_agent(aid)
        return await h.execute(aid, hirnql)

    async def forget(
        self,
        memory_id: str,
        *,
        agent_id: str | None = None,
    ) -> None:
        """Forget a memory by its ULID string (async).

        Args:
            memory_id: ULID string of the memory to forget.
            agent_id: Optional per-call agent identifier.
        """
        h = self._check_open()
        aid = self._effective_agent(agent_id)
        await self._ensure_agent(aid)
        await h.forget(aid, memory_id)

    async def watch(self) -> WatchStream:
        """Subscribe to memory events and return a WatchStream.

        Usage::

            stream = await mem.watch()
            async for event in stream:
                print(event)
            stream.cancel()
        """
        h = self._check_open()
        return await h.watch()

    async def stats(self) -> Stats:
        """Get database statistics (async)."""
        h = self._check_open()
        return await h.stats()

    def __repr__(self) -> str:
        if self._hirn is not None:
            emb = type(self._embeddings).__name__
            return f"AsyncMemory(open, embeddings={emb}, dims={self._embeddings.dimensions})"
        return "AsyncMemory(closed)"


__all__ = ["Memory", "AsyncMemory"]
