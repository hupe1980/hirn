---
description: "Use when working on hirn-python (PyO3) or hirn-node (napi-rs): FFI bindings, tokio runtime bridging, GIL handling, or view API delegation."
applyTo: crates/hirn-python/**, crates/hirn-node/**
---
# Language Bindings

## Python (hirn-python)

**Async bridging:**
- The public Python package exports both high-level `Memory` and `AsyncMemory`; prefer `AsyncMemory` for long-running or concurrent application calls.
- The low-level sync bridge still uses `pyo3_async_runtimes::tokio::get_runtime().block_on(f)` and therefore keeps the GIL while the Rust call runs.
- Async native methods use `pyo3_async_runtimes::tokio::future_into_py(...)`; keep the PyO3 bridge internal and put Python-facing ergonomics on the pure-Python `Memory` / `AsyncMemory` layer.

**numpy integration:**
- f32 arrays: zero-copy via `try_readonly()`
- f64 arrays: conversion with NaN/Inf/overflow checks
- Python lists: fallback `Vec<f32>` conversion

**Exception hierarchy:**
- `NotFoundError` ⊂ `QueryError` ⊂ `HirnError` ⊂ `PyException`
- Catch specific exceptions first (`except NotFoundError` before `except HirnError`)

## Node.js (hirn-node)

- napi-rs bindings; async methods return JS Promises
- Similar runtime bridging pattern as Python
