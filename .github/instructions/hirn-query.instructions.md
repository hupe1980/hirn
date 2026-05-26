---
description: "Use when working on hirn-query: HirnQL parser, Pest grammar, AST nodes, typed AST, plan compiler, query pipeline, query limits, or query compilation."
applyTo: "crates/hirn-query/**"
---
# hirn-query

HirnQL query language — Pest grammar parser + typed AST + DataFusion plan compiler.

Absorbed `hirn-ql` (parser/AST) and added compiler pipeline.

## Module Structure

```
hirn-query/src/
  parser/
    hirnql.pest    — PEG grammar (Pest)
    parse.rs       — Parser → raw AST
    ast.rs         — Raw AST types
  compiler/
    typed_ast.rs   — Analyze: raw AST → TypedStatement (namespace resolution, validation)
    plan_compiler.rs — Compile: TypedStatement → DataFusion LogicalPlan (HirnPlanNode extension nodes)
    pipeline.rs    — 7-stage QueryPipeline: parse → limits → analyze → compile → cache → explain → execute
```

## Grammar

Pest grammar in `src/parser/hirnql.pest`. Recursive descent, linear time, no backtracking.

**Clause ordering is grammar-enforced** — clauses must appear in the order defined in the grammar. Misordering causes parse errors.

## Statement Types

Data ops: `RECALL`, `RECALL EVENTS`, `THINK`, `REMEMBER`, `FORGET`, `CONNECT`
Inspection: `INSPECT`, `TRACE`, `EXPLAIN [ANALYZE]`
Lifecycle: `CONSOLIDATE`, `WATCH`, `TRAVERSE`
Policy: `GRANT`, `REVOKE`, `SHOW POLICIES`, `EXPLAIN POLICY`, `CREATE REALM`, `DROP REALM`, `SHOW CLUSTER`

## RECALL Clause Order

```
RECALL [layer] ABOUT <query>
  [INVOLVING <entities>]
  [AFTER/BEFORE/BETWEEN temporal]
  [AS OF <timestamp>]
  [EXPAND GRAPH DEPTH <d> [MIN_WEIGHT <w>] [ACTIVATION <mode>]]
  [FOLLOW CAUSES DEPTH <d>]
  [WHERE <conditions>]*
  [MODALITY <type>]
  [DEPTH AUTO|FULL|SUMMARY]
  [TOPIC <topic>]
  [WITH PROSPECTIVE ON|OFF]
  [WITH MCFA_DEFENSE ON|OFF]
  [WITH CONFLICTS]
  [GROUP BY <field>]
  [SELECT <fields>]
  [AS JSON|TABLE|MARKDOWN]
  [FORMAT JSON|TABLE|MARKDOWN]
  [BUDGET <tokens>]
  [NAMESPACE <ns>]
  [CONSISTENCY STRONG|EVENTUAL]
  [LIMIT <n>]
  [HYBRID]
```

## THINK Clause Order

```
THINK [GLOBAL] ABOUT <query>
  [INVOLVING <entities>]
  [AFTER/BEFORE/BETWEEN temporal]
  [EXPAND GRAPH DEPTH <d>]
  [FOLLOW CAUSES DEPTH <d>]
  [WHERE <conditions>]*
  [DEPTH AUTO|FULL|SUMMARY]
  [WITH PROSPECTIVE ON|OFF]
  [WITH MCFA_DEFENSE ON|OFF]
  [AS JSON|TABLE|MARKDOWN]
  [BUDGET <tokens>]
  [NAMESPACE <ns>]
  [CONSISTENCY STRONG|EVENTUAL]
  [LIMIT <n>]
  [MODE LOCAL|GLOBAL|HYBRID|RAPTOR|ADAPTIVE|ITERATIVE [MAX_HOPS <n>]]
  [COMMUNITY_DEPTH <n>]
  [HYBRID]
```

## THINK Modes

`LOCAL` (HNSW + spreading activation), `GLOBAL` (community summaries), `HYBRID` (both), `RAPTOR` (hierarchical), `ADAPTIVE` (query-complexity routing), `ITERATIVE` (multi-hop retrieval with MAX_HOPS).

Standalone THINK `HYBRID` is separate from `MODE HYBRID`: the clause enables BM25 + vector fusion on the local branch, while the mode merges local and global/community retrieval.

## Depth Scheduling

- `DEPTH AUTO` — classify query complexity via `QueryComplexityExec`, route pipeline depth automatically (default)
- `DEPTH FULL` — always run full pipeline (all operators)
- `DEPTH SUMMARY` — skip graph activation for faster summary-only results

## Plan Compilation

`TypedStatement` → DataFusion `LogicalPlan` trees of `HirnPlanNode` extension nodes.

- **RECALL:** [QueryComplexity] → HybridSearch → [GraphActivation] → [CausalChain] → HebbianBuffer → [ContextBudget]
- **THINK:** [QueryComplexity] → HybridSearch → [GraphActivation] → [IterativeRetrieval] → QualityGate → HebbianBuffer → ContextBudget
- **REMEMBER:** RpeScore → ProspectiveIndexing → SvoExtraction → InterferenceDetector → Remember
- **CONSOLIDATE:** ImperativeBoundary(Consolidate)

## DoS Protection (QueryLimits)

| Limit | Default | Enforced at |
|-------|---------|-------------|
| `max_query_length` | 1 MB | Parse time |
| `max_expand_depth` | 10 | Parse time |
| `max_limit` | 10,000 | Parse time |

Enforced by `parse_with_limits()`. No runtime protection — queries that pass parse execute fully.

## Keywords

Case-insensitive. First-word typos get suggestions ("did you mean 'RECALL'?").
