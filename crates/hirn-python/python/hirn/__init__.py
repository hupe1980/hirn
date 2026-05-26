"""hirn — Brain-inspired cognitive memory database for LLMs.

Level 1 (Zero-config) API::

    from hirn import Memory

    mem = Memory.open("./brain")
    mem.remember("User prefers dark mode")
    ctx = mem.think("What are the user's preferences?", budget=2048)
    print(ctx.context)

Level 1 with explicit embeddings::

    from hirn import Memory
    from hirn.embeddings.openai import OpenAIEmbeddings

    mem = Memory.open("./brain", embeddings=OpenAIEmbeddings())
    mem.remember("User prefers dark mode")
    ctx = mem.think("What are the user's preferences?", budget=2048)
    print(ctx.context)

The package root intentionally exports only the high-level ``Memory`` and
``AsyncMemory`` APIs. The native PyO3 bridge remains internal to the binding.
"""

from hirn._hirn import (
    WatchStream,
    RecallResult,
    Context,
    QueryResult,
    Stats,
    HirnError,
    NotFoundError,
    QueryError,
)
from hirn.embeddings import EmbeddingFunction
from hirn.memory import AsyncMemory, Memory

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "Memory",
    "AsyncMemory",
    "EmbeddingFunction",
    "WatchStream",
    "RecallResult",
    "Context",
    "QueryResult",
    "Stats",
    "HirnError",
    "NotFoundError",
    "QueryError",
]
