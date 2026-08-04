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

## Documentation Site

The landing page and documentation live in `site/` and are built with
[Zola](https://www.getzola.org/), a single static binary — no package manager, no lockfile.

```bash
# Install (macOS); see getzola.org for other platforms.
brew install zola

cd site
zola serve                      # live reload on http://127.0.0.1:1111
zola check --skip-external-links  # what CI enforces
zola build                      # output in site/public/ (gitignored)
```

Content is Markdown under `site/content/docs/`, one directory per section. A few conventions
matter:

- **Cross-references use Zola's `@/` syntax**, written as
  `` [Architecture](@/docs/concepts/architecture.md) ``. These are resolved at build time, so a
  renamed or deleted page fails the build instead of shipping a dead link. Plain relative paths
  are not checked; do not use them.
- **Callouts are shortcodes**, not blockquotes: `{% note() %}…{% end %}`. Available kinds are
  `note`, `tip`, `important`, `warning`, `danger`, and `experimental`.
- **Links to repository files** (benchmark artifacts, source) must be absolute GitHub URLs.
  The published site does not contain the repo, so a relative path would 404.
- **Nav order** comes from each page's `weight`; the sidebar and prev/next links are generated
  from it, so there is no separate navigation file to keep in sync.

Some documentation is compiled: `docs_smoke` tests in `hirn-core`, `hirn-query`, and
`hirn-policy` `include_str!` pages from `site/content/` and assert the examples still match the
code. Moving one of those pages breaks the build by design.

The front end is deliberately small — one stylesheet (`site/sass/main.scss`) and one deferred
script (`site/static/site.js`), no framework and no build step beyond Zola. A page costs about
13 KB gzipped in total; the search index is larger but is fetched only when someone actually
searches. Keep it that way: if a change needs a bundler, it probably needs rethinking.

## Architecture

See [Architecture](https://hupe1980.github.io/hirn/docs/concepts/architecture/) for the full system architecture.

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
