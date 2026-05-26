#!/usr/bin/env bash
# check.sh — run the same checks as CI locally
# Usage: ./check.sh [--skip-audit]
set -euo pipefail

SKIP_AUDIT=false
for arg in "$@"; do
  [[ "$arg" == "--skip-audit" ]] && SKIP_AUDIT=true
done

echo "==> fmt"
cargo fmt --all -- --check

echo "==> build"
RUSTFLAGS="-Dwarnings" cargo build --workspace --all-targets

echo "==> clippy"
RUSTFLAGS="-Dwarnings" cargo clippy --workspace --all-targets

echo "==> test"
RUSTFLAGS="-Dwarnings" cargo test --workspace

if [[ "$SKIP_AUDIT" == false ]]; then
  echo "==> deny (advisories, licenses, bans, sources)"
  cargo deny check advisories
  cargo deny check licenses
  cargo deny check bans
  cargo deny check sources
else
  echo "==> deny skipped (--skip-audit)"
fi

echo ""
echo "All checks passed."
