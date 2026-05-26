# Contributing to hirn

Thank you for considering contributing to hirn! This guide will help you get set up and productive.

## Development Environment

### Prerequisites

- **Rust 1.91+** (edition 2024) — install via [rustup](https://rustup.rs/)
- **cargo-fuzz** — for fuzz testing: `cargo install cargo-fuzz`
- **cargo-llvm-cov** — for coverage: `cargo install cargo-llvm-cov`

### Clone and Build

```bash
git clone https://github.com/hupe1980/hirn.git
cd hirn
cargo build --workspace
```

## Running Tests

```bash
# Run all workspace tests
cargo test --workspace

# Run tests for a specific crate
cargo test -p hirn-engine

# Run a specific test by name
cargo test -p hirn-engine "consolidation::raptor"
```

## Running Benchmarks

```bash
# Run all benchmarks
cargo bench

# Run benchmarks for a specific crate
cargo bench -p hirn-bench
```

## Running Fuzz Tests

Fuzz targets are in the `fuzz/` directory:

```bash
# List available fuzz targets
cargo fuzz list

# Run a fuzz target (e.g., HirnQL parser)
cargo fuzz run hirnql_parse -- -max_total_time=60

# Available targets: hirnql_parse, bincode_snapshot, lance_filter
```

## Coding Standards

### Formatting

All code must pass `cargo fmt`:

```bash
cargo fmt --all        # Format everything
cargo fmt --check --all  # Check without modifying (used in CI)
```

### Linting

Clippy warnings are treated as errors in CI:

```bash
RUSTFLAGS="-Dwarnings" cargo clippy --workspace --all-targets
```

### Rules

- **No `unwrap()` in production code.** Use `?`, `expect()` with a message, or proper error handling.
- **No `unsafe` without a `// SAFETY:` comment** explaining why it is sound.
- **Typed errors everywhere.** Use the crate-level error types (`HirnError`, `EmbedError`, `LlmError`, `StorageError`).
- **Tests for every feature.** Every new function, bug fix, or behavior change must ship with tests.

## Architecture

See [docs/architecture.md](docs/architecture.md) for the full system architecture.

### Crate Structure

```
crates/
├── hirn           # Umbrella crate (re-exports)
├── hirn-core      # Core types, config, error definitions
├── hirn-graph     # Property graph, spreading activation, Hebbian learning
├── hirn-query     # HirnQL parser, typed AST, compiler pipeline
├── hirn-storage   # Lance 4.0 storage engine and PhysicalStore
├── hirn-provider  # Embedders, LLMs, tokenizers, rerankers
├── hirn-exec      # DataFusion operators, UDFs, optimizer rules
├── hirn-policy    # Cedar authorization and audit helpers
├── hirn-engine    # Main engine: HirnDB, recall, consolidation
├── hirn-python    # Python (PyO3) bindings
├── hirn-node      # Node.js (napi-rs) bindings
├── hirn-bench     # Cognitive benchmarks
└── hirnd          # Standalone gRPC/HTTP/MCP server with Raft coordination
```

## PR Process

1. **Fork** the repository and create a feature branch.
2. **Write tests** before or alongside your implementation.
3. **Run the full test suite** locally: `cargo test --workspace`.
4. **Run clippy and fmt**: `cargo fmt --all && RUSTFLAGS="-Dwarnings" cargo clippy --workspace --all-targets`.
5. **Open a PR** with a clear description of what changed and why.
6. CI will run markdown link checks, workspace build/test, Linux fmt/clippy, and `cargo deny` advisory/license/source checks.

## License

By contributing, you agree that your contributions will be licensed under the [Apache-2.0 License](LICENSE).
