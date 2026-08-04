# 🔍 hirn-query

> [!WARNING]
> **Experimental.** APIs, on-disk formats, and behaviour may change without notice.
> Not recommended for production use.

HirnQL parser, typed AST, and DataFusion plan compiler for the hirn cognitive memory database.

## Overview

This crate provides the complete query pipeline for HirnQL — hirn's declarative query language. It contains the parser, typed AST analysis, and multi-stage compilation pipeline that transforms HirnQL text into DataFusion `LogicalPlan` trees.

## Architecture

### 7-Stage Query Pipeline

```
┌─────────┐   ┌────────┐   ┌─────────┐   ┌─────────┐   ┌───────┐   ┌─────────┐   ┌─────────┐
│  Parse  │ → │ Limits │ → │ Analyze │ → │ Compile │ → │ Cache │ → │ Explain │ → │ Execute │
└─────────┘   └────────┘   └─────────┘   └─────────┘   └───────┘   └─────────┘   └─────────┘
```

1. **Parse** — Pest PEG grammar → raw AST (`Statement`)
2. **Limits** — DoS protection (max query length, expand depth, limit)
3. **Analyze** — Namespace resolution, type validation → `TypedStatement`
4. **Compile** — `TypedStatement` → DataFusion `LogicalPlan` (tree of `HirnPlanNode` extension nodes)
5. **Cache** — DashMap-based `PlanCache` with LRU eviction
6. **Explain** — Optional `EXPLAIN [ANALYZE]` formatting
7. **Execute** — Handed to `hirn-exec` for physical plan conversion

### Plan Compilation

Each statement compiles to a pipeline of hirn extension nodes:

| Statement | Pipeline |
|-----------|----------|
| **RECALL** | [QueryComplexity] → HybridSearch → [GraphActivation] → [CausalChain] → HebbianBuffer → [ContextBudget] |
| **THINK** | [QueryComplexity] → HybridSearch → [GraphActivation] → [IterativeRetrieval] → QualityGate → HebbianBuffer → ContextBudget |
| **REMEMBER** | RpeScore → ProspectiveIndexing → SvoExtraction → InterferenceDetector → Remember |
| **CONSOLIDATE** | ImperativeBoundary(Consolidate) |
| **TRAVERSE** | Traverse (leaf) |

Brackets indicate conditionally emitted operators based on query clauses.

### Module Layout

```
src/
├── parser/
│   ├── hirnql.pest      # PEG grammar (Pest) — all clause orderings defined here
│   ├── parse.rs          # Parser: text → raw AST, DoS limits enforcement
│   └── ast.rs            # Raw AST types (Statement, RecallStmt, ThinkStmt, etc.)
├── compiler/
│   ├── typed_ast.rs      # Analyze: raw AST → TypedStatement (resolve namespaces, validate)
│   ├── plan_compiler.rs  # Compile: TypedStatement → DataFusion LogicalPlan
│   └── pipeline.rs       # QueryPipeline: orchestrates all stages, PlanCache
└── lib.rs                # Public re-exports
```

## Usage

```rust
use hirn_query::{parse, analyze, compile, AnalyzeContext};

// Parse
let stmt = parse(r#"RECALL episodic ABOUT "deployment" DEPTH AUTO LIMIT 10"#)?;

// Analyze (resolve types)
let ctx = AnalyzeContext::default();
let typed = analyze(&stmt, &ctx)?;

// Compile to DataFusion plan
let plan = compile(&typed)?;
```

Or use the full pipeline:

```rust
use hirn_query::{QueryPipeline, PlanCache};

let cache = PlanCache::new(128);
let pipeline = QueryPipeline::new(cache);
let compiled = pipeline.compile(r#"RECALL episodic ABOUT "test""#)?;
```

## Key Concepts

### Depth Scheduling
- `DEPTH AUTO` — classify query complexity automatically (default)
- `DEPTH FULL` — always run full pipeline
- `DEPTH SUMMARY` — skip graph activation for faster results

### Cognitive Operators
- **QueryComplexity** — classifies query as Simple/Medium/Complex
- **QualityGate** — confidence-based fallback to LLM deliberation
- **IterativeRetrieval** — multi-hop retrieve→reformulate→retrieve loop
- **InterferenceDetector** — duplicate/conflict detection before writes

### Grammar Clause Ordering
Clauses must appear in grammar-defined order. Misordering causes parse errors. See `src/parser/hirnql.pest` for exact ordering.

## 📚 Documentation

- [HirnQL Reference](https://hupe1980.github.io/hirn/docs/hirnql-reference/) — this crate's concepts, explained
- [Full documentation](https://hupe1980.github.io/hirn/)
