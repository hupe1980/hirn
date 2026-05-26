"""Integration tests for the hirn Python bindings."""

import asyncio
import math
import os
import tempfile
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pytest
import hirn
import hirn._hirn as hirn_bridge_module

from hirn import (
    AsyncMemory,
    Context,
    HirnError,
    NotFoundError,
    QueryError,
    QueryResult,
    RecallResult,
    Stats,
)
from hirn._hirn import AsyncHirnBridge as AsyncHirn, HirnBridge as Hirn
from hirn.embeddings.fake import FakeEmbeddings

DIM = 64
TEST_AGENT_ID = "agent-1"
ORIGINAL_ABOUT = "lease authority"
CURRENT_ABOUT = "lease authority v2"


def make_embedding(seed: float = 0.1) -> list[float]:
    """Create a simple embedding vector."""
    return [seed] * DIM


def db_path(tmp: str, name: str = "test.hirn") -> str:
    return os.path.join(tmp, name)


def observed_cutover_after(timestamp: str) -> str:
    observed_at = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    observed_at = observed_at.astimezone(UTC) + timedelta(hours=2)
    return observed_at.isoformat().replace("+00:00", "Z")


def seed_semantic_revision_history(path: str) -> dict[str, Any]:
    embeddings = FakeEmbeddings(dimensions=DIM)
    with Memory.open(path, embeddings=embeddings, agent_id=TEST_AGENT_ID) as mem:
        created = mem.query(f'REMEMBER semantic CONTENT "{ORIGINAL_ABOUT}"')
        assert created.type == "created"
        original_id = created.json["id"]

        original_history = mem.query(f'HISTORY "{original_id}"')
        assert original_history.type == "history"
        original_created_at = original_history.json["semantic_revision"]["revisions"][0][
            "created_at"
        ]

        superseded = mem.query(
            f'SUPERSEDE "{original_id}" SET description = "{CURRENT_ABOUT}" '
            f'REASON "cutover" OBSERVED AT "{observed_cutover_after(original_created_at)}"'
        )
        assert superseded.type == "superseded"

        history = mem.query(f'HISTORY "{original_id}"')
        assert history.type == "history"
        summary = history.json["semantic_revision"]
        revisions = summary["revisions"]

    return {
        "embeddings": embeddings,
        "logical_memory_id": summary["logical_memory_id"],
        "original_revision_id": revisions[0]["revision_id"],
        "historical_cutoff": revisions[0]["created_at"],
        "recorded_cutoff": revisions[-1]["created_at"],
    }


# ─── Package Surface ─────────────────────────────────────────


class TestPackageSurface:
    def test_root_package_does_not_export_low_level_handles(self):
        assert not hasattr(hirn, "Hirn")
        assert not hasattr(hirn, "AsyncHirn")

    def test_internal_bridge_exports_explicit_bridge_names(self):
        assert hasattr(hirn_bridge_module, "HirnBridge")
        assert hasattr(hirn_bridge_module, "AsyncHirnBridge")


# ─── Sync HirnBridge Tests ───────────────────────────────────


class TestHirnOpenClose:
    def test_open_and_close(self, tmp_path):
        h = Hirn.open(str(tmp_path / "test.hirn"), embedding_dimensions=DIM)
        assert h is not None
        h.close()

    def test_context_manager(self, tmp_path):
        with Hirn.open(str(tmp_path / "test.hirn"), embedding_dimensions=DIM) as h:
            s = h.stats()
            assert isinstance(s, Stats)
        # After __exit__, database should be closed
        with pytest.raises(RuntimeError):
            h.stats()

    def test_double_close_is_safe(self, tmp_path):
        h = Hirn.open(str(tmp_path / "test.hirn"), embedding_dimensions=DIM)
        h.close()
        h.close()  # should not raise


class TestRegisterAgent:
    def test_register_agent(self, tmp_path):
        with Hirn.open(str(tmp_path / "test.hirn"), embedding_dimensions=DIM) as h:
            h.register_agent("agent-1", "Test Agent")

    def test_register_agent_empty_id_fails(self, tmp_path):
        with Hirn.open(str(tmp_path / "test.hirn"), embedding_dimensions=DIM) as h:
            with pytest.raises(QueryError):
                h.register_agent("", "Test")


class TestRemember:
    def test_remember_returns_ulid(self, tmp_path):
        with Hirn.open(str(tmp_path / "test.hirn"), embedding_dimensions=DIM) as h:
            h.register_agent("agent-1", "Test Agent")
            mid = h.remember("agent-1", "Hello world", embedding=make_embedding())
            assert isinstance(mid, str)
            assert len(mid) == 26  # ULID length

    def test_remember_with_numpy(self, tmp_path):
        with Hirn.open(str(tmp_path / "test.hirn"), embedding_dimensions=DIM) as h:
            h.register_agent("agent-1", "Test Agent")
            emb = np.array(make_embedding(), dtype=np.float32)
            mid = h.remember("agent-1", "Numpy test", embedding=emb)
            assert len(mid) == 26

    def test_remember_with_importance(self, tmp_path):
        with Hirn.open(str(tmp_path / "test.hirn"), embedding_dimensions=DIM) as h:
            h.register_agent("agent-1", "Test Agent")
            mid = h.remember(
                "agent-1", "Important!", embedding=make_embedding(), importance=0.9
            )
            assert len(mid) == 26

    def test_remember_without_embedding(self, tmp_path):
        with Hirn.open(str(tmp_path / "test.hirn"), embedding_dimensions=DIM) as h:
            h.register_agent("agent-1", "Test Agent")
            mid = h.remember("agent-1", "No embedding")
            assert len(mid) == 26


class TestRecall:
    def test_recall_returns_results(self, tmp_path):
        """Hirn.open() uses NullBackend — vector search returns empty.

        Real recall is tested via Memory class (uses LanceDbBackend).
        """
        with Hirn.open(str(tmp_path / "test.hirn"), embedding_dimensions=DIM) as h:
            h.register_agent("agent-1", "Test Agent")
            emb = make_embedding(0.5)
            h.remember("agent-1", "Memory one", embedding=emb)
            h.remember("agent-1", "Memory two", embedding=emb)

            results = h.recall("agent-1", emb, limit=5)
            assert isinstance(results, list)

    def test_recall_with_threshold(self, tmp_path):
        """Hirn.open() uses NullBackend — threshold is accepted but no results."""
        with Hirn.open(str(tmp_path / "test.hirn"), embedding_dimensions=DIM) as h:
            h.register_agent("agent-1", "Test Agent")
            h.remember("agent-1", "Threshold test", embedding=make_embedding(0.3))
            results = h.recall("agent-1", make_embedding(0.3), limit=5, threshold=0.99)
            assert isinstance(results, list)

    def test_recall_empty_db(self, tmp_path):
        with Hirn.open(str(tmp_path / "test.hirn"), embedding_dimensions=DIM) as h:
            h.register_agent("agent-1", "Test Agent")
            results = h.recall("agent-1", make_embedding(), limit=5)
            assert results == []

    def test_recall_with_numpy_query(self, tmp_path):
        """Hirn.open() uses NullBackend — numpy query accepted but no results."""
        with Hirn.open(str(tmp_path / "test.hirn"), embedding_dimensions=DIM) as h:
            h.register_agent("agent-1", "Test Agent")
            emb = make_embedding(0.7)
            h.remember("agent-1", "Numpy recall", embedding=emb)
            query = np.array(emb, dtype=np.float32)
            results = h.recall("agent-1", query, limit=5)
            assert isinstance(results, list)


class TestThink:
    def test_think_returns_context(self, tmp_path):
        with Hirn.open(str(tmp_path / "test.hirn"), embedding_dimensions=DIM) as h:
            h.register_agent("agent-1", "Test Agent")
            emb = make_embedding(0.4)
            h.remember("agent-1", "Context memory", embedding=emb)

            ctx = h.think("agent-1", emb, budget=4096)
            assert isinstance(ctx, Context)
            assert isinstance(ctx.context, str)
            assert isinstance(ctx.token_count, int)
            assert isinstance(ctx.query_time_ms, float)
            assert isinstance(ctx.records_included, list)

    def test_focus_token_count_hint_does_not_override_rust_tokenizer(self, tmp_path):
        with Hirn.open(
            str(tmp_path / "test.hirn"),
            embedding_dimensions=DIM,
            tokenizer_name="estimating",
        ) as h:
            h.register_agent("agent-1", "Test Agent")
            h.focus(
                "agent-1",
                "Working memory note that should stay under Rust-side token control.",
                token_count=1,
            )

            ctx = h.think("agent-1", make_embedding(0.4), budget=256)
            assert "Working memory note" in ctx.context
            assert ctx.token_count == math.ceil(len(ctx.context.encode("utf-8")) / 4)


class TestForget:
    def test_forget_memory(self, tmp_path):
        with Hirn.open(str(tmp_path / "test.hirn"), embedding_dimensions=DIM) as h:
            h.register_agent("agent-1", "Test Agent")
            mid = h.remember("agent-1", "To forget", embedding=make_embedding())
            h.forget("agent-1", mid)  # should not raise

    def test_forget_invalid_id_fails(self, tmp_path):
        with Hirn.open(str(tmp_path / "test.hirn"), embedding_dimensions=DIM) as h:
            h.register_agent("agent-1", "Test Agent")
            with pytest.raises(QueryError):
                h.forget("agent-1", "not-a-valid-ulid")


class TestExecute:
    def test_execute_recall_ql(self, tmp_path):
        with Hirn.open(str(tmp_path / "test.hirn"), embedding_dimensions=DIM) as h:
            h.register_agent("agent-1", "Test Agent")
            h.remember("agent-1", "QL test memory", embedding=make_embedding())

            result = h.execute("agent-1", 'RECALL episodic ABOUT "test" LIMIT 5')
            assert isinstance(result, QueryResult)
            assert result.type == "records"
            assert isinstance(result.json, dict)

    def test_execute_think_ql(self, tmp_path):
        with Hirn.open(str(tmp_path / "test.hirn"), embedding_dimensions=DIM) as h:
            h.register_agent("agent-1", "Test Agent")
            h.remember("agent-1", "Think QL memory", embedding=make_embedding())

            result = h.execute("agent-1", 'THINK ABOUT "test" BUDGET 4096')
            assert isinstance(result, QueryResult)
            assert result.type == "records"

    def test_execute_invalid_ql_fails(self, tmp_path):
        with Hirn.open(str(tmp_path / "test.hirn"), embedding_dimensions=DIM) as h:
            h.register_agent("agent-1", "Test Agent")
            with pytest.raises((HirnError, QueryError)):
                h.execute("agent-1", "NOT VALID QL")


class TestInspect:
    def test_inspect_memory(self, tmp_path):
        with Hirn.open(str(tmp_path / "test.hirn"), embedding_dimensions=DIM) as h:
            h.register_agent("agent-1", "Test Agent")
            mid = h.remember("agent-1", "Inspectable", embedding=make_embedding())
            result = h.inspect("agent-1", mid)
            assert isinstance(result, QueryResult)
            assert result.type == "inspected"
            data = result.json
            assert data["id"] == mid

    def test_inspect_invalid_id_fails(self, tmp_path):
        with Hirn.open(str(tmp_path / "test.hirn"), embedding_dimensions=DIM) as h:
            h.register_agent("agent-1", "Test Agent")
            with pytest.raises(QueryError):
                h.inspect("agent-1", "bad-id")


class TestTrace:
    def test_trace_memory(self, tmp_path):
        with Hirn.open(str(tmp_path / "test.hirn"), embedding_dimensions=DIM) as h:
            h.register_agent("agent-1", "Test Agent")
            mid = h.remember("agent-1", "Traceable", embedding=make_embedding())
            result = h.trace("agent-1", mid)
            assert isinstance(result, QueryResult)
            assert result.type == "traced"
            data = result.json
            assert data["id"] == mid
            assert "trust_score" in data


class TestStats:
    def test_stats_empty_db(self, tmp_path):
        with Hirn.open(str(tmp_path / "test.hirn"), embedding_dimensions=DIM) as h:
            s = h.stats()
            assert isinstance(s, Stats)
            assert s.total_count == 0
            assert s.episodic_count == 0
            assert s.working_count == 0
            assert s.semantic_count == 0
            assert s.file_size_bytes >= 0

    def test_stats_after_remember(self, tmp_path):
        with Hirn.open(str(tmp_path / "test.hirn"), embedding_dimensions=DIM) as h:
            h.register_agent("agent-1", "Test Agent")
            h.remember("agent-1", "Stats test", embedding=make_embedding())
            s = h.stats()
            assert s.episodic_count == 1
            assert s.total_count >= 1

    def test_stats_repr(self, tmp_path):
        with Hirn.open(str(tmp_path / "test.hirn"), embedding_dimensions=DIM) as h:
            s = h.stats()
            r = repr(s)
            assert "Stats(" in r
            assert "total=" in r


class TestErrorHierarchy:
    def test_not_found_is_hirn_error(self):
        assert issubclass(NotFoundError, HirnError)

    def test_query_error_is_hirn_error(self):
        assert issubclass(QueryError, HirnError)


class TestClosedDbRaises:
    def test_remember_on_closed(self, tmp_path):
        h = Hirn.open(str(tmp_path / "test.hirn"), embedding_dimensions=DIM)
        h.close()
        with pytest.raises(RuntimeError):
            h.remember("agent-1", "test")

    def test_recall_on_closed(self, tmp_path):
        h = Hirn.open(str(tmp_path / "test.hirn"), embedding_dimensions=DIM)
        h.close()
        with pytest.raises(RuntimeError):
            h.recall("agent-1", make_embedding())

    def test_stats_on_closed(self, tmp_path):
        h = Hirn.open(str(tmp_path / "test.hirn"), embedding_dimensions=DIM)
        h.close()
        with pytest.raises(RuntimeError):
            h.stats()


# ─── Async Tests ──────────────────────────────────────────────


class TestAsyncHirn:
    def test_async_open_close(self, tmp_path):
        async def run():
            h = await AsyncHirn.open(str(tmp_path / "test.hirn"), embedding_dimensions=DIM)
            assert h is not None
            await h.close()

        asyncio.run(run())

    def test_async_methods_raise_after_close(self, tmp_path):
        async def run():
            h = await AsyncHirn.open(str(tmp_path / "test.hirn"), embedding_dimensions=DIM)
            await h.close()

            with pytest.raises(RuntimeError, match="closed"):
                await h.stats()

            with pytest.raises(RuntimeError, match="closed"):
                await h.remember("agent-1", "test")

        asyncio.run(run())

    def test_async_context_manager(self, tmp_path):
        async def run():
            async with await AsyncHirn.open(
                str(tmp_path / "test.hirn"), embedding_dimensions=DIM
            ) as h:
                s = await h.stats()
                assert isinstance(s, Stats)

        asyncio.run(run())

    def test_async_remember_recall(self, tmp_path):
        """Hirn.open() uses NullBackend — recall returns empty."""

        async def run():
            async with await AsyncHirn.open(
                str(tmp_path / "test.hirn"), embedding_dimensions=DIM
            ) as h:
                await h.register_agent("agent-1", "Test Agent")
                emb = make_embedding(0.6)
                mid = await h.remember("agent-1", "Async memory", embedding=emb)
                assert len(mid) == 26

                results = await h.recall("agent-1", emb, limit=5)
                assert isinstance(results, list)

        asyncio.run(run())

    def test_async_think(self, tmp_path):
        async def run():
            async with await AsyncHirn.open(
                str(tmp_path / "test.hirn"), embedding_dimensions=DIM
            ) as h:
                await h.register_agent("agent-1", "Test Agent")
                emb = make_embedding(0.4)
                await h.remember("agent-1", "Async think", embedding=emb)
                ctx = await h.think("agent-1", emb, budget=4096)
                assert isinstance(ctx, Context)

        asyncio.run(run())

    def test_async_execute(self, tmp_path):
        async def run():
            async with await AsyncHirn.open(
                str(tmp_path / "test.hirn"), embedding_dimensions=DIM
            ) as h:
                await h.register_agent("agent-1", "Test Agent")
                await h.remember("agent-1", "Async QL", embedding=make_embedding())
                result = await h.execute(
                    "agent-1", 'RECALL episodic ABOUT "test" LIMIT 5'
                )
                assert isinstance(result, QueryResult)

        asyncio.run(run())

    def test_async_inspect_trace(self, tmp_path):
        async def run():
            async with await AsyncHirn.open(
                str(tmp_path / "test.hirn"), embedding_dimensions=DIM
            ) as h:
                await h.register_agent("agent-1", "Test Agent")
                mid = await h.remember(
                    "agent-1", "Async inspect", embedding=make_embedding()
                )

                result = await h.inspect("agent-1", mid)
                assert result.type == "inspected"

                result = await h.trace("agent-1", mid)
                assert result.type == "traced"

        asyncio.run(run())

    def test_async_forget(self, tmp_path):
        async def run():
            async with await AsyncHirn.open(
                str(tmp_path / "test.hirn"), embedding_dimensions=DIM
            ) as h:
                await h.register_agent("agent-1", "Test Agent")
                mid = await h.remember(
                    "agent-1", "Async forget", embedding=make_embedding()
                )
                await h.forget("agent-1", mid)

        asyncio.run(run())

    def test_async_stats(self, tmp_path):
        async def run():
            async with await AsyncHirn.open(
                str(tmp_path / "test.hirn"), embedding_dimensions=DIM
            ) as h:
                await h.register_agent("agent-1", "Test Agent")
                await h.remember("agent-1", "Async stats", embedding=make_embedding())
                s = await h.stats()
                assert s.episodic_count == 1

        asyncio.run(run())

    def test_async_concurrent_remember_operations(self, tmp_path):
        async def run():
            async with await AsyncHirn.open(
                str(tmp_path / "test.hirn"), embedding_dimensions=DIM
            ) as h:
                await h.register_agent("agent-1", "Test Agent")
                await h.remember("agent-1", "Seed entry", embedding=make_embedding(0.01))

                async def remember_one(i: int) -> str:
                    return await h.remember(
                        "agent-1",
                        f"Async concurrent bridge memory #{i}",
                        embedding=make_embedding(0.02 + (i * 0.01)),
                    )

                ids = await asyncio.gather(*(remember_one(i) for i in range(20)))

                assert len(ids) == 20
                assert len(set(ids)) == 20

                s = await h.stats()
                assert s.total_count >= 20

        asyncio.run(run())


# ─── Memory (Level 1 Zero-Config) Tests ──────────────────────

from hirn import Memory, AsyncMemory, WatchStream


class TestMemoryOpenRememberThink:
    """Story 4.4: open → remember → think → correct context."""

    def test_open_remember_think(self, tmp_path):
        mem = Memory.open(str(tmp_path / "brain"))
        mid = mem.remember("User prefers dark mode")
        assert isinstance(mid, str)
        assert len(mid) == 26  # ULID

        ctx = mem.think("What are the user's preferences?", budget=2048)
        assert isinstance(ctx, Context)
        assert isinstance(ctx.context, str)
        assert isinstance(ctx.token_count, int)
        assert isinstance(ctx.query_time_ms, float)
        assert isinstance(ctx.records_included, list)
        mem.close()

    def test_context_manager(self, tmp_path):
        with Memory.open(str(tmp_path / "brain")) as mem:
            mem.remember("Test content")
            ctx = mem.think("content", budget=1024)
            assert isinstance(ctx, Context)

    def test_closed_raises(self, tmp_path):
        mem = Memory.open(str(tmp_path / "brain"))
        mem.close()
        with pytest.raises(RuntimeError):
            mem.remember("should fail")


class TestMemoryHirnQL:
    """Story 4.4: query with HirnQL → results match Rust API."""

    def test_query_remember_and_recall(self, tmp_path):
        mem = Memory.open(str(tmp_path / "brain"))
        # Use remember() to store, then recall via HirnQL
        mem.remember("The meeting is at 3pm")

        result = mem.query('RECALL episodic ABOUT "meeting" LIMIT 5')
        assert isinstance(result, QueryResult)
        assert result.type == "records"
        data = result.json
        assert isinstance(data, dict)
        mem.close()


class TestMemoryEditing:
    """High-level semantic edit helpers mirror revision-native HirnQL."""

    def test_edit_helpers_cover_correct_supersede_and_retract(self, tmp_path):
        mem = Memory.open(str(tmp_path / "brain"), agent_id=TEST_AGENT_ID)
        try:
            created_target = mem.query('REMEMBER semantic CONTENT "lease authority"')
            target_id = created_target.json["id"]

            corrected = mem.correct(
                target_id,
                description="canonical lease authority clarified",
                confidence=0.7,
                evidence_count=2,
                reason="clarified wording",
            )
            assert corrected.type == "corrected"

            superseded = mem.supersede(
                target_id,
                description="canonical lease authority v2",
                reason="authoritative cutover",
            )
            assert superseded.type == "superseded"

            retracted = mem.retract(target_id, reason="obsolete")
            assert retracted.type == "retracted"
        finally:
            mem.close()

    def test_merge_helper_builds_expected_query(self):
        mem = object.__new__(Memory)
        captured: dict[str, object] = {}
        sentinel = object()

        def fake_query(hirnql: str, *, agent_id: str | None = None):
            captured["hirnql"] = hirnql
            captured["agent_id"] = agent_id
            return sentinel

        mem.query = fake_query  # type: ignore[attr-defined]

        result = mem.merge(
            ["01HSRCA", "01HSRCB"],
            "01HTARGET",
            description="canonical lease authority",
            confidence=0.95,
            evidence_count=3,
            reason="deduplicate agents",
            observed_at="2026-03-01T00:00:00Z",
            caused_by="01HCAUSE",
            agent_id="agent-merge",
        )

        assert result is sentinel
        assert captured == {
            "hirnql": (
                'MERGE MEMORY "01HSRCA", "01HSRCB" INTO "01HTARGET" '
                'SET description = "canonical lease authority", confidence = 0.95, '
                'evidence_count = 3 REASON "deduplicate agents" '
                'OBSERVED AT "2026-03-01T00:00:00Z" CAUSED BY "01HCAUSE"'
            ),
            "agent_id": "agent-merge",
        }

    def test_correct_requires_at_least_one_update_field(self, tmp_path):
        mem = Memory.open(str(tmp_path / "brain"), agent_id=TEST_AGENT_ID)
        try:
            with pytest.raises(ValueError, match="at least one semantic update field"):
                mem.correct("01HXYZ")
        finally:
            mem.close()

    def test_merge_requires_at_least_one_source_id(self, tmp_path):
        mem = Memory.open(str(tmp_path / "brain"), agent_id=TEST_AGENT_ID)
        try:
            with pytest.raises(
                ValueError, match="source_ids must contain at least one memory ID"
            ):
                mem.merge([], "01HXYZ")
        finally:
            mem.close()


class TestMemoryRecall:
    """Story 4.4: remember image bytes → retrieve by text query."""

    def test_remember_and_recall_text(self, tmp_path):
        mem = Memory.open(str(tmp_path / "brain"))
        mem.remember("Cats are beloved pets worldwide")
        mem.remember("Dogs are loyal companions")

        results = mem.recall("pets", limit=5)
        assert isinstance(results, list)
        assert len(results) >= 1
        for r in results:
            assert isinstance(r, RecallResult)
            assert isinstance(r.similarity, float)
        mem.close()

    def test_recall_supports_historical_snapshots_and_revision_metadata(self, tmp_path):
        path = str(tmp_path / "brain")
        seeded = seed_semantic_revision_history(path)

        mem = Memory.open(
            path,
            embeddings=seeded["embeddings"],
            agent_id=TEST_AGENT_ID,
        )
        try:
            current = mem.recall(CURRENT_ABOUT, limit=10, threshold=0.0)
            assert len(current) == 1
            assert current[0].logical_memory_id == seeded["logical_memory_id"]
            assert current[0].revision_state == "Active"
            assert current[0].revision_id != seeded["original_revision_id"]

            historical = mem.recall(
                ORIGINAL_ABOUT,
                limit=10,
                as_of=seeded["historical_cutoff"],
            )
            assert len(historical) == 1
            assert historical[0].revision_id == seeded["original_revision_id"]
            assert historical[0].revision_state == "Active"

            recorded = mem.recall(
                CURRENT_ABOUT,
                limit=10,
                as_of=seeded["recorded_cutoff"],
                snapshot_kind="recorded",
            )
            assert len(recorded) == 1
            assert recorded[0].revision_id != seeded["original_revision_id"]

            revision = mem.recall(
                ORIGINAL_ABOUT,
                limit=10,
                as_of=seeded["original_revision_id"],
                snapshot_kind="revision",
            )
            assert len(revision) == 1
            assert revision[0].revision_id == seeded["original_revision_id"]
        finally:
            mem.close()


class TestMemoryRepr:
    """Story 4.4: __repr__ on results shows useful information."""

    def test_memory_repr_open(self, tmp_path):
        mem = Memory.open(str(tmp_path / "brain"))
        r = repr(mem)
        assert r.startswith("Memory(open")
        assert "embeddings=" in r
        assert "dims=" in r
        mem.close()
        assert repr(mem) == "Memory(closed)"

    def test_context_repr(self, tmp_path):
        mem = Memory.open(str(tmp_path / "brain"))
        mem.remember("Some content for repr test")
        ctx = mem.think("content", budget=1024)
        r = repr(ctx)
        assert "Context(" in r
        assert "tokens=" in r
        mem.close()

    def test_recall_result_repr(self, tmp_path):
        mem = Memory.open(str(tmp_path / "brain"))
        mem.remember("Data for recall repr")
        results = mem.recall("recall", limit=5)
        if results:
            r = repr(results[0])
            assert "RecallResult(" in r
        mem.close()

    def test_query_result_repr(self, tmp_path):
        mem = Memory.open(str(tmp_path / "brain"))
        mem.remember("Data for query repr")
        result = mem.query('RECALL episodic ABOUT "data" LIMIT 5')
        r = repr(result)
        assert "QueryResult(" in r
        mem.close()


class TestAsyncMemory:
    """Story 4.4: async support (asyncio.run)."""

    def test_async_open_remember_think(self, tmp_path):
        async def run():
            mem = await AsyncMemory.open(str(tmp_path / "brain"))
            mid = await mem.remember("Async dark mode preference")
            assert isinstance(mid, str)
            assert len(mid) == 26

            ctx = await mem.think("preferences", budget=2048)
            assert isinstance(ctx, Context)
            assert isinstance(ctx.context, str)
            await mem.close()

        asyncio.run(run())


class TestAsyncMemoryEditing:
    """Async semantic edit helpers mirror the sync API."""

    def test_async_edit_helpers_cover_correct_supersede_and_retract(self, tmp_path):
        async def run():
            mem = await AsyncMemory.open(str(tmp_path / "brain"), agent_id=TEST_AGENT_ID)
            try:
                created_target = await mem.query('REMEMBER semantic CONTENT "lease authority"')
                target_id = created_target.json["id"]

                corrected = await mem.correct(
                    target_id,
                    description="canonical lease authority clarified",
                    confidence=0.7,
                    evidence_count=2,
                    reason="clarified wording",
                )
                assert corrected.type == "corrected"

                superseded = await mem.supersede(
                    target_id,
                    description="canonical lease authority v2",
                    reason="authoritative cutover",
                )
                assert superseded.type == "superseded"

                retracted = await mem.retract(target_id, reason="obsolete")
                assert retracted.type == "retracted"
            finally:
                await mem.close()

        asyncio.run(run())

    def test_async_merge_helper_builds_expected_query(self):
        async def run():
            mem = object.__new__(AsyncMemory)
            captured: dict[str, object] = {}
            sentinel = object()

            async def fake_query(hirnql: str, *, agent_id: str | None = None):
                captured["hirnql"] = hirnql
                captured["agent_id"] = agent_id
                return sentinel

            mem.query = fake_query  # type: ignore[attr-defined]

            result = await mem.merge(
                ["01HSRCA", "01HSRCB"],
                "01HTARGET",
                description="canonical lease authority",
                confidence=0.95,
                evidence_count=3,
                reason="deduplicate agents",
                observed_at="2026-03-01T00:00:00Z",
                caused_by="01HCAUSE",
                agent_id="agent-merge",
            )

            assert result is sentinel
            assert captured == {
                "hirnql": (
                    'MERGE MEMORY "01HSRCA", "01HSRCB" INTO "01HTARGET" '
                    'SET description = "canonical lease authority", confidence = 0.95, '
                    'evidence_count = 3 REASON "deduplicate agents" '
                    'OBSERVED AT "2026-03-01T00:00:00Z" CAUSED BY "01HCAUSE"'
                ),
                "agent_id": "agent-merge",
            }

        asyncio.run(run())

    def test_async_recall_supports_historical_snapshots(self, tmp_path):
        path = str(tmp_path / "brain")
        seeded = seed_semantic_revision_history(path)

        async def run():
            mem = await AsyncMemory.open(
                path,
                embeddings=seeded["embeddings"],
                agent_id=TEST_AGENT_ID,
            )
            current = await mem.recall(CURRENT_ABOUT, limit=10, threshold=0.0)
            assert len(current) == 1
            assert current[0].logical_memory_id == seeded["logical_memory_id"]
            assert current[0].revision_id != seeded["original_revision_id"]

            revision = await mem.recall(
                ORIGINAL_ABOUT,
                limit=10,
                as_of=seeded["original_revision_id"],
                snapshot_kind="revision",
            )
            assert len(revision) == 1
            assert revision[0].revision_id == seeded["original_revision_id"]

            recorded = await mem.recall(
                CURRENT_ABOUT,
                limit=10,
                as_of=seeded["recorded_cutoff"],
                snapshot_kind="recorded",
            )
            assert len(recorded) == 1
            assert recorded[0].revision_id != seeded["original_revision_id"]
            await mem.close()

        asyncio.run(run())


# ─── Memory batch_remember + forget ──────────────────────────


class TestMemoryBatchRemember:
    """Story 4.3: batch_remember stores multiple memories efficiently."""

    def test_batch_remember_returns_ids(self, tmp_path):
        mem = Memory.open(str(tmp_path / "brain"))
        ids = mem.batch_remember([
            "Alpha particles consist of two protons and two neutrons bound together",
            "Beta decay converts a neutron into a proton electron and antineutrino",
            "Gamma rays are high-energy electromagnetic radiation from nuclear transitions",
        ])
        assert len(ids) == 3
        for mid in ids:
            assert isinstance(mid, str)
            assert len(mid) == 26
        # All IDs unique
        assert len(set(ids)) == 3
        mem.close()

    def test_batch_remember_empty(self, tmp_path):
        mem = Memory.open(str(tmp_path / "brain"))
        ids = mem.batch_remember([])
        assert ids == []
        mem.close()

    def test_batch_remember_rejects_non_string(self, tmp_path):
        mem = Memory.open(str(tmp_path / "brain"))
        with pytest.raises(TypeError, match="must be a string"):
            mem.batch_remember([42])
        mem.close()

    def test_batch_remember_rejects_empty_string(self, tmp_path):
        mem = Memory.open(str(tmp_path / "brain"))
        with pytest.raises(ValueError, match="empty"):
            mem.batch_remember(["valid", "   "])
        mem.close()


class TestMemoryForget:
    """Story 4.3: Memory.forget archives a memory by ID."""

    def test_forget_remembered(self, tmp_path):
        mem = Memory.open(str(tmp_path / "brain"))
        mid = mem.remember("Temporary fact to be forgotten soon after creation")
        mem.forget(mid)  # should not raise
        mem.close()

    def test_forget_with_agent_id(self, tmp_path):
        mem = Memory.open(str(tmp_path / "brain"), agent_id="agent-f")
        mid = mem.remember("Agent-scoped fact for forget test")
        mem.forget(mid, agent_id="agent-f")  # should not raise
        mem.close()


class TestAsyncMemoryBatchRemember:
    """Story 4.3: async batch_remember."""

    def test_async_batch_remember(self, tmp_path):
        async def run():
            mem = await AsyncMemory.open(str(tmp_path / "brain"))
            ids = await mem.batch_remember([
                "Neutron stars are incredibly dense remnants of massive supernovae",
                "Black holes warp spacetime so strongly that light cannot escape",
            ])
            assert len(ids) == 2
            for mid in ids:
                assert isinstance(mid, str)
                assert len(mid) == 26
            await mem.close()

        asyncio.run(run())


class TestAsyncMemoryForget:
    """Story 4.3: async forget."""

    def test_async_forget(self, tmp_path):
        async def run():
            mem = await AsyncMemory.open(str(tmp_path / "brain"))
            mid = await mem.remember("Async temporary fact to forget")
            await mem.forget(mid)  # should not raise
            await mem.close()

        asyncio.run(run())


# ─── Story 5.3: SDK End-to-End (Python) ──────────────────────


MEMORIES = [
    "Kubernetes horizontal pod autoscaler adjusts replica count based on CPU utilization",
    "Docker multi-stage builds reduce final image size by discarding build dependencies",
    "PostgreSQL MVCC provides snapshot isolation without read locks",
    "Redis sorted sets maintain elements with scores for leaderboard patterns",
    "JWT refresh tokens rotate on each use to prevent replay attacks",
    "OAuth2 authorization code flow with PKCE prevents interception attacks",
    "TLS 1.3 handshake completes in a single round trip for improved latency",
    "Prometheus PromQL supports rate functions for counter metric analysis",
    "Grafana alerting evaluates rules at configurable intervals with notification channels",
    "Terraform state files track resource identity for idempotent infrastructure changes",
    "Helm chart values files override default template variables per environment",
    "gRPC bidirectional streaming enables real-time communication between services",
    "Elasticsearch inverted index maps terms to document IDs for fast full-text search",
    "WebAssembly linear memory model provides sandboxed execution environment",
    "Istio service mesh injects Envoy sidecar proxies for mutual TLS encryption",
    "Apache Kafka partitions distribute message load across consumer group members",
    "GitHub Actions workflow files define CI/CD pipelines with reusable composite actions",
    "SQLite WAL mode allows concurrent readers alongside a single writer process",
    "Nginx reverse proxy handles SSL termination and upstream load balancing",
    "RabbitMQ dead letter exchanges capture unprocessable messages for later analysis",
]


class TestPythonSdkE2E:
    """Story 5.3: Python SDK end-to-end — remember 20 memories → think → recall."""

    def test_remember_20_then_think(self, tmp_path):
        mem = Memory.open(str(tmp_path / "brain"))
        for text in MEMORIES:
            mid = mem.remember(text)
            assert isinstance(mid, str)
            assert len(mid) == 26

        ctx = mem.think("authentication security", budget=4096)
        assert isinstance(ctx, Context)
        assert len(ctx.context) > 0
        assert ctx.token_count > 0
        assert len(ctx.records_included) >= 1
        mem.close()

    def test_remember_20_then_recall(self, tmp_path):
        mem = Memory.open(str(tmp_path / "brain"))
        for text in MEMORIES:
            mem.remember(text)

        results = mem.recall("database performance", limit=5)
        assert isinstance(results, list)
        assert len(results) >= 1
        assert len(results) <= 5
        for r in results:
            assert isinstance(r, RecallResult)
            assert isinstance(r.similarity, float)
        mem.close()

    def test_hirnql_remember_and_recall(self, tmp_path):
        mem = Memory.open(str(tmp_path / "brain"))
        for text in MEMORIES:
            result = mem.query(f'REMEMBER episode CONTENT "{text}"')
            assert result.type == "created"

        result = mem.query('RECALL episodic ABOUT "message queue" LIMIT 5')
        assert result.type == "records"
        mem.close()

    def test_hirnql_think(self, tmp_path):
        mem = Memory.open(str(tmp_path / "brain"))
        for text in MEMORIES:
            mem.remember(text)

        result = mem.query('THINK ABOUT "infrastructure provisioning"')
        assert result.type == "records"
        mem.close()

    def test_hirnql_inspect_and_forget(self, tmp_path):
        mem = Memory.open(str(tmp_path / "brain"))
        mid = mem.remember("Temporary note to be forgotten")

        result = mem.query(f'INSPECT "{mid}"')
        assert result.type == "inspected"

        result = mem.query(f'FORGET "{mid}" ARCHIVE')
        assert result.type == "forgotten"
        mem.close()

    def test_hirnql_connect_and_trace(self, tmp_path):
        mem = Memory.open(str(tmp_path / "brain"))
        id1 = mem.remember("gRPC uses Protocol Buffers for efficient serialization")
        id2 = mem.remember("Protocol Buffers define strongly typed message schemas")

        result = mem.query(f'CONNECT "{id1}" TO "{id2}" AS related_to WEIGHT 0.9')
        assert result.type == "connected"

        result = mem.query(f'TRACE "{id1}"')
        assert result.type == "traced"
        mem.close()

    def test_explain_plan(self, tmp_path):
        mem = Memory.open(str(tmp_path / "brain"))
        for text in MEMORIES[:5]:
            mem.remember(text)

        result = mem.query('EXPLAIN RECALL episodic ABOUT "database"')
        assert result.type == "explain"
        mem.close()

    def test_context_manager_full_lifecycle(self, tmp_path):
        with Memory.open(str(tmp_path / "brain")) as mem:
            for text in MEMORIES:
                mem.remember(text)
            ctx = mem.think("SSL encryption", budget=2048)
            assert len(ctx.context) > 0
            results = mem.recall("kubernetes", limit=10)
            assert len(results) >= 1


# ─── Story 5.2: Python Binding Extensions ────────────────────


class TestAsyncMemoryTrulyAsync:
    """Story 5.2: AsyncMemory methods are truly async (await-able)."""

    def test_async_open_remember_recall(self, tmp_path):
        """Async open → await remember → await recall → correct results."""
        async def run():
            mem = await AsyncMemory.open(str(tmp_path / "brain"))
            mid = await mem.remember("Kubernetes uses etcd for cluster state")
            assert isinstance(mid, str)
            assert len(mid) == 26

            results = await mem.recall("cluster state management", limit=5)
            assert isinstance(results, list)
            assert len(results) >= 1
            assert results[0].id == mid

            await mem.close()

        asyncio.run(run())

    def test_async_think(self, tmp_path):
        """Async think returns proper Context."""
        async def run():
            mem = await AsyncMemory.open(str(tmp_path / "brain"))
            await mem.remember("Redis sorted sets for leaderboards")
            ctx = await mem.think("data structures", budget=2048)
            assert isinstance(ctx, Context)
            assert len(ctx.context) > 0
            assert ctx.token_count > 0
            await mem.close()

        asyncio.run(run())

    def test_async_query(self, tmp_path):
        """Async HirnQL query."""
        async def run():
            mem = await AsyncMemory.open(str(tmp_path / "brain"))
            await mem.remember("Docker multi-stage builds reduce image size")
            result = await mem.query('RECALL episodic ABOUT "docker" LIMIT 5')
            assert result.type == "records"
            await mem.close()

        asyncio.run(run())

    def test_async_stats(self, tmp_path):
        """Async stats returns Stats."""
        async def run():
            mem = await AsyncMemory.open(str(tmp_path / "brain"))
            await mem.remember("Test content")
            s = await mem.stats()
            assert isinstance(s, Stats)
            assert s.total_count >= 1
            await mem.close()

        asyncio.run(run())

    def test_async_context_manager(self, tmp_path):
        """Async context manager open/close lifecycle."""
        async def run():
            async with await AsyncMemory.open(str(tmp_path / "brain")) as mem:
                mid = await mem.remember("Context manager test")
                assert len(mid) == 26

        asyncio.run(run())

    def test_async_memory_methods_raise_after_close(self, tmp_path):
        """Closed AsyncMemory surfaces return a structured runtime failure."""

        async def run():
            mem = await AsyncMemory.open(str(tmp_path / "brain"))
            await mem.close()

            with pytest.raises(RuntimeError, match="closed"):
                await mem.remember("should fail")

            with pytest.raises(RuntimeError, match="closed"):
                await mem.query('RECALL episodic ABOUT "closed" LIMIT 1')

        asyncio.run(run())


class TestAsyncMemoryWatch:
    """Story 5.2: watch → receive events."""

    def test_sync_memory_watch(self, tmp_path):
        """Sync Memory.watch() collects events over duration."""
        import threading, time

        mem = Memory.open(str(tmp_path / "brain"))

        # Subscribe and produce events in a thread
        def produce():
            time.sleep(0.1)
            mem.remember("Event source content")

        t = threading.Thread(target=produce)
        t.start()
        events = mem.watch(duration_ms=2000)
        t.join()
        # At minimum, the channel was subscribed; events may or may not
        # arrive depending on timing. Verify the return is a list.
        assert isinstance(events, list)
        mem.close()

    def test_async_watch_stream(self, tmp_path):
        """AsyncMemory.watch() returns a WatchStream with cancel/is_done."""
        async def run():
            mem = await AsyncMemory.open(str(tmp_path / "brain"))
            stream = await mem.watch()
            assert isinstance(stream, WatchStream)
            assert not stream.is_done()
            stream.cancel()
            assert stream.is_done()
            await mem.close()

        asyncio.run(run())

    def test_watch_next_event_timeout(self, tmp_path):
        """WatchStream.next_event returns None on timeout."""
        async def run():
            mem = await AsyncMemory.open(str(tmp_path / "brain"))
            stream = await mem.watch()
            ev = stream.next_event(timeout_ms=50)
            assert ev is None  # no events produced yet
            stream.cancel()
            await mem.close()

        asyncio.run(run())


class TestMemoryAgentId:
    """Story 5.2: agent_id parameter on Memory operations."""

    def test_memory_open_with_agent_id(self, tmp_path):
        """Memory.open accepts agent_id keyword."""
        mem = Memory.open(str(tmp_path / "brain"), agent_id="test_agent")
        mid = mem.remember("Content from test_agent")
        assert isinstance(mid, str)
        assert len(mid) == 26
        mem.close()

    def test_memory_remember_per_call_agent_id(self, tmp_path):
        """remember() accepts per-call agent_id override."""
        mem = Memory.open(str(tmp_path / "brain"))
        mid = mem.remember("Custom agent content", agent_id="custom_agent")
        assert isinstance(mid, str)
        assert len(mid) == 26
        mem.close()

    def test_memory_recall_with_agent_id(self, tmp_path):
        """recall() accepts agent_id parameter."""
        mem = Memory.open(str(tmp_path / "brain"))
        mem.remember("Test content for recall")
        results = mem.recall("test content", limit=5, agent_id="reader_agent")
        assert isinstance(results, list)
        mem.close()

    def test_memory_think_with_agent_id(self, tmp_path):
        """think() accepts agent_id parameter."""
        mem = Memory.open(str(tmp_path / "brain"))
        mem.remember("Context assembly test data")
        ctx = mem.think("context", budget=2048, agent_id="thinker_agent")
        assert isinstance(ctx, Context)
        mem.close()

    def test_memory_query_with_agent_id(self, tmp_path):
        """query() accepts agent_id parameter."""
        mem = Memory.open(str(tmp_path / "brain"))
        mem.remember("HirnQL agent test")
        result = mem.query(
            'RECALL episodic ABOUT "agent" LIMIT 5',
            agent_id="querier",
        )
        assert result.type == "records"
        mem.close()

    def test_async_memory_open_with_agent_id(self, tmp_path):
        """AsyncMemory.open accepts agent_id keyword."""
        async def run():
            mem = await AsyncMemory.open(
                str(tmp_path / "brain"), agent_id="async_agent"
            )
            mid = await mem.remember("Async agent content")
            assert len(mid) == 26
            await mem.close()

        asyncio.run(run())

    def test_async_memory_per_call_agent_id(self, tmp_path):
        """AsyncMemory operations accept per-call agent_id."""
        async def run():
            mem = await AsyncMemory.open(str(tmp_path / "brain"))
            mid = await mem.remember("content", agent_id="caller_agent")
            assert len(mid) == 26
            results = await mem.recall("content", limit=5, agent_id="r")
            assert isinstance(results, list)
            ctx = await mem.think("topic", budget=1024, agent_id="t")
            assert isinstance(ctx, Context)
            await mem.close()

        asyncio.run(run())


class TestConcurrentAsyncOperations:
    """Story 5.2: 50 concurrent async operations → all complete."""

    def test_50_concurrent_async_operations(self, tmp_path):
        async def run():
            mem = await AsyncMemory.open(str(tmp_path / "brain"))

            # Prime the database so the episodic table exists before concurrency.
            await mem.remember("Seed entry to initialise tables")

            async def remember_one(i: int) -> str:
                return await mem.remember(f"Concurrent memory entry #{i}")

            # Launch 50 concurrent remember operations
            tasks = [remember_one(i) for i in range(50)]
            results = await asyncio.gather(*tasks)

            assert len(results) == 50
            for mid in results:
                assert isinstance(mid, str)
                assert len(mid) == 26

            # Verify all 50 entries are stored
            s = await mem.stats()
            assert s.total_count >= 50

            await mem.close()

        asyncio.run(run())

