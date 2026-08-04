# Contributing to hirn

Thank you for considering contributing to hirn! This guide will help you get set up and productive.

## Development Environment

### Prerequisites

- **Rust 1.91.1+** (edition 2024) — install via [rustup](https://rustup.rs/)
- **protobuf-compiler** — `brew install protobuf` / `apt install protobuf-compiler`
- **[just](https://github.com/casey/just)** — task runner: `brew install just`
- **[zola](https://www.getzola.org/)** — documentation site: `brew install zola`

Optional: `cargo-fuzz` (fuzzing), `cargo-llvm-cov` (coverage), `cargo-deny`
(supply chain), `maturin` + `pytest` (Python bindings), Node 20+ (Node bindings).
`just setup` prints the full list.

### Clone and Build

```bash
git clone https://github.com/hupe1980/hirn.git
cd hirn
cargo build --workspace
```

### Before you push

```bash
just ci
```

That runs **exactly** what `.github/workflows/ci.yml` gates on — formatting, clippy with
warnings denied, workspace tests, doctests, the feature-gated `hirn-provider` build, and
both link checks. If it passes locally it passes in CI; when the two drift, the justfile is
wrong and should be corrected against the workflow.

`just` on its own lists every recipe. The ones you will reach for most:

| Recipe | What it does |
|---|---|
| `just ci` | The full pre-push gate |
| `just dev` | Format, then type-check — fast feedback while editing |
| `just t <pattern>` | Run one test by name across the workspace |
| `just test-crate <crate>` | Tests for a single crate |
| `just site` | Serve the docs with live reload |
| `just links` | Both link checks |
| `just clean-incremental` | Reclaim disk when `target/` gets large |

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
├── hirn-storage   # Lance 9 storage engine and PhysicalStore
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
3. **Run `just ci`** — the same gates CI enforces. Do this before pushing, not after a red build.
4. **Open a PR** with a clear description of what changed and why.
5. CI re-runs the `just ci` gates plus `cargo deny` advisory/license/source checks, the
   MSRV floor, and the Python/Node bindings.

## License

By contributing, you agree that your contributions will be licensed under the [Apache-2.0 License](LICENSE).
