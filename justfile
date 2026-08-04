# hirn task runner — https://github.com/casey/just
#
# `just ci` runs exactly what .github/workflows/ci.yml gates on. If it passes
# here it passes there; when the two drift, this file is wrong and should be
# corrected against the workflow rather than the other way round.

set shell := ["bash", "-uc"]

# Feature set for hirn-provider. Its default is `[]`, so the provider modules
# are invisible to workspace-wide commands and need an explicit pass.
# cross-encoder is excluded: it pulls the ONNX runtime, too heavy for CI.
provider_features := "openai,anthropic,cohere,voyage,ollama"

zola_version := "0.22.1"

# Show available recipes.
default:
    @just --list --unsorted

# ── The gate ────────────────────────────────────────────────────────────────

# Everything CI checks. Run this before pushing.
ci: fmt-check lint test test-doc lint-provider test-provider links
    @echo ""
    @echo "✅ all CI gates passed"

# Fast feedback while editing: format, then compile-check the workspace.
dev: fmt check
    @echo "✅ formatted and type-checked"

# ── Rust ────────────────────────────────────────────────────────────────────

# Format the workspace in place.
fmt:
    cargo fmt --all

# Fail if anything is unformatted (CI gate).
fmt-check:
    cargo fmt --check --all

# Type-check without building test binaries — the quickest correctness signal.
check:
    cargo check --workspace --all-targets

# Clippy with warnings denied, as CI runs it.
lint:
    RUSTFLAGS="-Dwarnings" cargo clippy --workspace --all-targets

# hirn-provider's feature-gated modules, invisible to the workspace run above.
lint-provider:
    RUSTFLAGS="-Dwarnings" cargo clippy -p hirn-provider --all-targets --features {{provider_features}}

# Unit, integration, and binary tests (nextest when available, as CI runs it).
test:
    #!/usr/bin/env bash
    set -euo pipefail
    if command -v cargo-nextest >/dev/null 2>&1; then
        cargo nextest run --workspace --profile ci
    else
        echo "note: cargo-nextest not installed (cargo binstall cargo-nextest); using cargo test"
        cargo test --workspace --lib --bins --tests
    fi

# Doctests, which the flags above deliberately skip.
test-doc:
    cargo test --workspace --doc

test-provider:
    cargo test -p hirn-provider --features {{provider_features}}

# Install the tools CI uses, via prebuilt binaries.
install-tools:
    cargo binstall -y cargo-nextest cargo-deny || \
        cargo install cargo-nextest cargo-deny --locked

# Run one test by name across the workspace: `just t normalize_preference`
t pattern:
    cargo test --workspace {{pattern}}

# Tests for a single crate: `just test-crate hirn-engine`
test-crate crate:
    cargo test -p {{crate}}

# Compile against the MSRV floor declared in rust-toolchain.toml.
msrv:
    cargo +1.91.1 check --workspace

# ── Documentation ───────────────────────────────────────────────────────────

# Every link check CI runs: cross-file links, then site anchors.
links: links-markdown links-site

# Cross-file markdown links (fails loudly on a path that does not exist).
links-markdown:
    python3 scripts/check_markdown_links.py \
        README.md CONTRIBUTING.md FINDINGS.md \
        site bench-results crates .github

# Site links and heading anchors, slugified exactly as the renderer does.
links-site:
    cd site && zola check --skip-external-links

# Serve the docs site with live reload on http://127.0.0.1:1111
site:
    cd site && zola serve

# Build the site into site/public/
site-build:
    cd site && zola build

# Regenerate the social preview card from its SVG source.
og-card:
    rsvg-convert -w 1200 -h 630 site/static/og-card.svg -o site/static/og-card.png
    @echo "✅ site/static/og-card.png regenerated"

# Rust API docs for the workspace.
doc:
    cargo doc --workspace --no-deps --open

# ── Bindings ────────────────────────────────────────────────────────────────

# Build and test the Python bindings. Needs an active virtualenv.
python:
    maturin develop --manifest-path crates/hirn-python/Cargo.toml
    pytest crates/hirn-python/tests

# Build and test the Node bindings (debug build, as CI runs it).
node:
    cd crates/hirn-node && npm install && npm test

bindings: python node

# ── Benchmarks ──────────────────────────────────────────────────────────────

# HIRN-Bench suites (pseudo-embeddings — a smoke test, not evidence).
bench-smoke:
    cargo run --release -p hirn-bench -- cognitive --benchmark all

# Full LongMemEval oracle run — needs OPENAI_API_KEY, takes ~30 min.
bench-lme data_dir output:
    # --reader-answers caches generated answers, so a judge-stage failure does
    # not force paying for them a second time.
    cargo run --release -p hirn-bench -- external \
        --format-name longmemeval --data-dir {{data_dir}} \
        --embeddings embeddings/longmemeval_oracle_embeddings.json \
        --embedding-model-label text-embedding-3-small \
        --dims 1536 --full-corpus --runs 1 --no-baselines \
        --reader gpt-4o --judge gpt-4o --reader-temperature 0.0 --seed 0 \
        --reader-answers /tmp/lme-answers.json \
        --json-output bench-results/{{output}}.json \
        --markdown-output bench-results/{{output}}.md

# Query-intent routing evidence.
bench-routing label:
    cargo run --release -p hirn-bench -- nlu-routing \
        --environment-label "{{label}}" \
        --json-output bench-results/nlu-routing.json \
        --markdown-output bench-results/nlu-routing.md

# ── Supply chain ────────────────────────────────────────────────────────────

audit:
    cargo deny check advisories

deny:
    cargo deny check licenses
    cargo deny check bans
    cargo deny check sources

# ── Housekeeping ────────────────────────────────────────────────────────────

# Reclaim disk: drop incremental artifacts (the bulk of a large target/).
clean-incremental:
    rm -rf target/debug/incremental
    @du -sh target 2>/dev/null || true

clean:
    cargo clean
    rm -rf site/public

# What a fresh checkout needs beyond a Rust toolchain.
setup:
    @echo "Required:"
    @echo "  protobuf-compiler   (brew install protobuf / apt install protobuf-compiler)"
    @echo "  zola {{zola_version}}          (brew install zola) — documentation site"
    @echo "Optional:"
    @echo "  maturin, pytest, numpy   — Python bindings"
    @echo "  node 20+                 — Node bindings"
    @echo "  librsvg                  — regenerating the OG card"
    @echo "  cargo-deny               — supply-chain checks"
