//! Output formatters: JSON, CSV, and Markdown.

use std::io::{self, Write};

use crate::advanced::{
    AdvancedResult, AdvancedSuiteResult, DEFAULT_STRATEGY_NAME as DEFAULT_ADVANCED_STRATEGY_NAME,
};
use crate::cognitive::{
    ActiveRetrievalSurfaces, BaselineStrategy, CognitiveResult, CognitiveSuiteResult,
    DEFAULT_STRATEGY_NAME as DEFAULT_COGNITIVE_STRATEGY_NAME,
};
use crate::metrics::BenchmarkResult;

/// Output format.
#[derive(Debug, Clone, Copy)]
pub enum OutputFormat {
    Json,
    Csv,
    Markdown,
}

impl std::str::FromStr for OutputFormat {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "json" => Ok(Self::Json),
            "csv" => Ok(Self::Csv),
            "md" | "markdown" => Ok(Self::Markdown),
            _ => Err(format!(
                "unknown format: {s} (expected json, csv, markdown)"
            )),
        }
    }
}

/// Write benchmark result in the given format.
pub fn write_result(
    result: &BenchmarkResult,
    format: OutputFormat,
    w: &mut dyn Write,
) -> io::Result<()> {
    match format {
        OutputFormat::Json => write_json(result, w),
        OutputFormat::Csv => write_csv(result, w),
        OutputFormat::Markdown => write_markdown(result, w),
    }
}

fn write_json(result: &BenchmarkResult, w: &mut dyn Write) -> io::Result<()> {
    let json = serde_json::to_string_pretty(result)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    writeln!(w, "{json}")
}

fn write_csv(result: &BenchmarkResult, w: &mut dyn Write) -> io::Result<()> {
    writeln!(
        w,
        "suite,run_id,precision,recall,f1,mrr,ndcg,\
         remember_p50_us,remember_p95_us,remember_p99_us,\
         recall_p50_us,recall_p95_us,recall_p99_us,\
         think_p50_us,think_p95_us,think_p99_us,\
         remember_ops_sec,recall_ops_sec,think_ops_sec,\
         peak_memory_bytes,db_file_size_bytes,total_time_us"
    )?;
    let a = &result.aggregate;
    writeln!(
        w,
        "{},{},{:.4},{:.4},{:.4},{:.4},{:.4},\
         {:.1},{:.1},{:.1},\
         {:.1},{:.1},{:.1},\
         {:.1},{:.1},{:.1},\
         {:.1},{:.1},{:.1},\
         {},{},{:.0}",
        result.suite_name,
        result.run_id,
        a.mean_precision,
        a.mean_recall,
        a.mean_f1,
        a.mean_mrr,
        a.mean_ndcg,
        result.remember_latency.p50.as_secs_f64() * 1e6,
        result.remember_latency.p95.as_secs_f64() * 1e6,
        result.remember_latency.p99.as_secs_f64() * 1e6,
        result.recall_latency.p50.as_secs_f64() * 1e6,
        result.recall_latency.p95.as_secs_f64() * 1e6,
        result.recall_latency.p99.as_secs_f64() * 1e6,
        result.think_latency.p50.as_secs_f64() * 1e6,
        result.think_latency.p95.as_secs_f64() * 1e6,
        result.think_latency.p99.as_secs_f64() * 1e6,
        result.throughput.remember_ops_per_sec,
        result.throughput.recall_ops_per_sec,
        result.throughput.think_ops_per_sec,
        result.peak_memory_bytes,
        result.db_file_size_bytes,
        result.total_time.as_secs_f64() * 1e6,
    )
}

fn write_markdown(result: &BenchmarkResult, w: &mut dyn Write) -> io::Result<()> {
    writeln!(w, "# Benchmark Report: {}", result.suite_name)?;
    writeln!(w)?;
    writeln!(w, "**Run ID:** {}", result.run_id)?;
    writeln!(w, "**Records:** {}", result.config.num_records)?;
    writeln!(w, "**Embedding dims:** {}", result.config.embedding_dims)?;
    writeln!(w, "**K:** {}", result.config.k)?;
    writeln!(w, "**Token budget:** {}", result.config.token_budget)?;
    writeln!(
        w,
        "**Runs:** {} warmup + {} measured",
        result.config.warmup_runs, result.config.measured_runs
    )?;
    writeln!(w)?;

    // Quality metrics.
    writeln!(w, "## Retrieval Quality")?;
    writeln!(w)?;
    writeln!(w, "| Metric | Value |")?;
    writeln!(w, "|--------|------:|")?;
    let a = &result.aggregate;
    writeln!(
        w,
        "| Precision@{} | {:.4} |",
        result.config.k, a.mean_precision
    )?;
    writeln!(w, "| Recall@{} | {:.4} |", result.config.k, a.mean_recall)?;
    writeln!(w, "| F1 | {:.4} |", a.mean_f1)?;
    writeln!(w, "| MRR | {:.4} |", a.mean_mrr)?;
    writeln!(w, "| NDCG@{} | {:.4} |", result.config.k, a.mean_ndcg)?;
    writeln!(w)?;

    // Latency.
    writeln!(w, "## Latency (µs)")?;
    writeln!(w)?;
    writeln!(w, "| Operation | p50 | p95 | p99 | min | max | mean |")?;
    writeln!(w, "|-----------|----:|----:|----:|----:|----:|-----:|")?;
    for (name, stats) in [
        ("remember", &result.remember_latency),
        ("recall", &result.recall_latency),
        ("think", &result.think_latency),
    ] {
        writeln!(
            w,
            "| {} | {:.1} | {:.1} | {:.1} | {:.1} | {:.1} | {:.1} |",
            name,
            stats.p50.as_secs_f64() * 1e6,
            stats.p95.as_secs_f64() * 1e6,
            stats.p99.as_secs_f64() * 1e6,
            stats.min.as_secs_f64() * 1e6,
            stats.max.as_secs_f64() * 1e6,
            stats.mean.as_secs_f64() * 1e6,
        )?;
    }
    writeln!(w)?;

    // Throughput.
    writeln!(w, "## Throughput")?;
    writeln!(w)?;
    writeln!(w, "| Operation | ops/sec |")?;
    writeln!(w, "|-----------|--------:|")?;
    writeln!(
        w,
        "| remember | {:.1} |",
        result.throughput.remember_ops_per_sec
    )?;
    writeln!(
        w,
        "| recall | {:.1} |",
        result.throughput.recall_ops_per_sec
    )?;
    writeln!(w, "| think | {:.1} |", result.throughput.think_ops_per_sec)?;
    writeln!(w)?;

    // Resource usage.
    writeln!(w, "## Resource Usage")?;
    writeln!(w)?;
    writeln!(w, "| Metric | Value |")?;
    writeln!(w, "|--------|------:|")?;
    writeln!(
        w,
        "| Peak RSS | {:.2} MB |",
        result.peak_memory_bytes as f64 / 1_048_576.0
    )?;
    writeln!(
        w,
        "| DB file size | {:.2} MB |",
        result.db_file_size_bytes as f64 / 1_048_576.0
    )?;
    writeln!(
        w,
        "| Total time | {:.2} s |",
        result.total_time.as_secs_f64()
    )?;
    writeln!(w)
}

// ─── Cognitive Benchmark Output ──────────────────────────────

/// Write cognitive benchmark suite results in the given format.
pub fn write_cognitive_result(
    result: &CognitiveSuiteResult,
    format: OutputFormat,
    w: &mut dyn Write,
) -> io::Result<()> {
    match format {
        OutputFormat::Json => write_cognitive_json(result, w),
        OutputFormat::Csv => write_cognitive_csv(result, w),
        OutputFormat::Markdown => write_cognitive_markdown(result, w),
    }
}

/// Write advanced benchmark suite results in the given format.
pub fn write_advanced_result(
    result: &AdvancedSuiteResult,
    format: OutputFormat,
    w: &mut dyn Write,
) -> io::Result<()> {
    match format {
        OutputFormat::Json => write_advanced_json(result, w),
        OutputFormat::Csv => write_advanced_csv(result, w),
        OutputFormat::Markdown => write_advanced_markdown(result, w),
    }
}

fn write_advanced_json(result: &AdvancedSuiteResult, w: &mut dyn Write) -> io::Result<()> {
    let json = serde_json::to_string_pretty(result)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    writeln!(w, "{json}")
}

fn write_advanced_csv(result: &AdvancedSuiteResult, w: &mut dyn Write) -> io::Result<()> {
    let summary_strategy = result
        .results
        .first()
        .map(|entry| entry.strategy.as_str())
        .unwrap_or(DEFAULT_ADVANCED_STRATEGY_NAME);
    writeln!(
        w,
        "benchmark,strategy,run_id,runs,offline_wait_ms,repro_threshold,os,arch,logical_cpus,\
         primary_score,precision,recall,accuracy,usefulness,latency_p50_us,latency_p95_us,latency_p99_us,\
         context_tokens,prompt_tokens,completion_tokens,total_tokens,estimated_spend_usd,total_cases,total_time_s,\
         repro_runs,repro_max_relative_delta,repro_materially_similar"
    )?;
    for benchmark in &result.results {
        write_advanced_csv_row(benchmark, result, w)?;
    }
    writeln!(
        w,
        "TOTAL,{},{},{},{},{},{},{},{},\
         {:.4},0.0000,0.0000,0.0000,0.0000,0.0,0.0,0.0,0,0,0,0,0.000000,{}, {:.3},0,0.0000,false",
        summary_strategy,
        result.run_id,
        result.metadata.runs,
        result.metadata.offline_wait_ms,
        result.metadata.repro_threshold,
        result.metadata.environment.os,
        result.metadata.environment.arch,
        result.metadata.environment.logical_cpus,
        result.overall_primary_score,
        result
            .results
            .iter()
            .map(|entry| entry.total_cases)
            .sum::<usize>(),
        result.total_time_secs,
    )?;
    Ok(())
}

fn write_advanced_markdown(result: &AdvancedSuiteResult, w: &mut dyn Write) -> io::Result<()> {
    writeln!(w, "# Advanced Offline Cognition Benchmark Report")?;
    writeln!(w)?;
    writeln!(w, "**Run ID:** {}", result.run_id)?;
    writeln!(w, "**Total time:** {:.2}s", result.total_time_secs)?;
    writeln!(
        w,
        "**Overall Primary Score:** {:.1}%",
        result.overall_primary_score * 100.0
    )?;
    writeln!(w)?;

    writeln!(w, "## Run Metadata")?;
    writeln!(w)?;
    writeln!(w, "| Field | Value |")?;
    writeln!(w, "|-------|:------|")?;
    writeln!(
        w,
        "| Generated at | {} |",
        result.metadata.generated_at_rfc3339
    )?;
    writeln!(w, "| Runs | {} |", result.metadata.runs)?;
    writeln!(
        w,
        "| Offline wait budget | {} ms |",
        result.metadata.offline_wait_ms
    )?;
    writeln!(
        w,
        "| Repro threshold | {:.1}% |",
        result.metadata.repro_threshold * 100.0
    )?;
    let environment_label = result.metadata.environment.label.as_deref().unwrap_or("-");
    let environment_image = result.metadata.environment.image.as_deref().unwrap_or("-");
    writeln!(w, "| Environment label | {} |", environment_label)?;
    writeln!(w, "| Environment image | {} |", environment_image)?;
    writeln!(
        w,
        "| Platform | {}/{} |",
        result.metadata.environment.os, result.metadata.environment.arch
    )?;
    writeln!(
        w,
        "| Logical CPUs | {} |",
        result.metadata.environment.logical_cpus
    )?;
    let git_commit_sha = result
        .metadata
        .environment
        .git_commit_sha
        .as_deref()
        .unwrap_or("-");
    let cargo_lock_blake3 = result
        .metadata
        .environment
        .cargo_lock_blake3
        .as_deref()
        .unwrap_or("-");
    writeln!(w, "| Git commit | {} |", git_commit_sha)?;
    writeln!(w, "| Cargo.lock blake3 | {} |", cargo_lock_blake3)?;
    writeln!(w)?;

    writeln!(w, "## Overview")?;
    writeln!(w)?;
    writeln!(
        w,
        "| Benchmark | Primary | Precision | Recall | Accuracy | Usefulness | p95 | Tokens | Spend |"
    )?;
    writeln!(
        w,
        "|-----------|--------:|----------:|-------:|---------:|-----------:|----:|-------:|------:|"
    )?;
    for benchmark in &result.results {
        writeln!(
            w,
            "| {} | {:.1}% | {:.1}% | {:.1}% | {:.1}% | {:.1}% | {:.2} ms | {} | ${:.4} |",
            benchmark.benchmark,
            benchmark.quality.primary_score * 100.0,
            benchmark.quality.precision * 100.0,
            benchmark.quality.recall * 100.0,
            benchmark.quality.accuracy * 100.0,
            benchmark.quality.usefulness * 100.0,
            benchmark.latency.p95.as_secs_f64() * 1_000.0,
            benchmark.cost.total_tokens,
            benchmark.cost.estimated_spend_usd,
        )?;
    }
    writeln!(w)?;

    for benchmark in &result.results {
        writeln!(w, "## {}", benchmark.benchmark)?;
        writeln!(w)?;
        writeln!(w, "| Metric | Value |")?;
        writeln!(w, "|--------|------:|")?;
        writeln!(
            w,
            "| Primary score | {:.1}% |",
            benchmark.quality.primary_score * 100.0
        )?;
        writeln!(
            w,
            "| Precision | {:.1}% |",
            benchmark.quality.precision * 100.0
        )?;
        writeln!(w, "| Recall | {:.1}% |", benchmark.quality.recall * 100.0)?;
        writeln!(
            w,
            "| Accuracy | {:.1}% |",
            benchmark.quality.accuracy * 100.0
        )?;
        writeln!(
            w,
            "| Usefulness | {:.1}% |",
            benchmark.quality.usefulness * 100.0
        )?;
        writeln!(w, "| Cases | {} |", benchmark.total_cases)?;
        writeln!(w, "| Total time | {:.3}s |", benchmark.total_time_secs)?;
        writeln!(w)?;

        writeln!(w, "### Latency")?;
        writeln!(w)?;
        writeln!(w, "| p50 | p95 | p99 | mean |")?;
        writeln!(w, "|----:|----:|----:|-----:|")?;
        writeln!(
            w,
            "| {:.2} ms | {:.2} ms | {:.2} ms | {:.2} ms |",
            benchmark.latency.p50.as_secs_f64() * 1_000.0,
            benchmark.latency.p95.as_secs_f64() * 1_000.0,
            benchmark.latency.p99.as_secs_f64() * 1_000.0,
            benchmark.latency.mean.as_secs_f64() * 1_000.0,
        )?;
        writeln!(w)?;

        writeln!(w, "### Cost Envelope")?;
        writeln!(w)?;
        writeln!(w, "| Context | Prompt | Completion | Total | Spend |")?;
        writeln!(w, "|--------:|-------:|-----------:|------:|------:|")?;
        writeln!(
            w,
            "| {} | {} | {} | {} | ${:.4} |",
            benchmark.cost.context_tokens,
            benchmark.cost.prompt_tokens,
            benchmark.cost.completion_tokens,
            benchmark.cost.total_tokens,
            benchmark.cost.estimated_spend_usd,
        )?;
        writeln!(w)?;
    }

    if result
        .results
        .iter()
        .any(|benchmark| benchmark.reproducibility.is_some())
    {
        writeln!(w, "## Reproducibility")?;
        writeln!(w)?;
        writeln!(
            w,
            "| Benchmark | Runs | Max drift | Threshold | Materially similar |"
        )?;
        writeln!(
            w,
            "|-----------|-----:|----------:|----------:|:-------------------|"
        )?;
        for benchmark in &result.results {
            if let Some(summary) = &benchmark.reproducibility {
                writeln!(
                    w,
                    "| {} | {} | {:.2}% | {:.2}% | {} |",
                    benchmark.benchmark,
                    summary.runs,
                    summary.max_relative_delta * 100.0,
                    summary.threshold * 100.0,
                    if summary.materially_similar {
                        "yes"
                    } else {
                        "no"
                    },
                )?;
            }
        }
        writeln!(w)?;
    }

    Ok(())
}

fn write_advanced_csv_row(
    benchmark: &AdvancedResult,
    suite: &AdvancedSuiteResult,
    w: &mut dyn Write,
) -> io::Result<()> {
    let reproducibility = benchmark.reproducibility.as_ref();
    writeln!(
        w,
        "{},{},{},{},{},{:.4},{},{},{},\
         {:.4},{:.4},{:.4},{:.4},{:.4},{:.1},{:.1},{:.1},\
         {},{},{},{},{:.6},{},{:.3},{},{:.4},{}",
        benchmark.benchmark,
        benchmark.strategy,
        benchmark.run_id,
        suite.metadata.runs,
        suite.metadata.offline_wait_ms,
        suite.metadata.repro_threshold,
        suite.metadata.environment.os,
        suite.metadata.environment.arch,
        suite.metadata.environment.logical_cpus,
        benchmark.quality.primary_score,
        benchmark.quality.precision,
        benchmark.quality.recall,
        benchmark.quality.accuracy,
        benchmark.quality.usefulness,
        benchmark.latency.p50.as_secs_f64() * 1_000_000.0,
        benchmark.latency.p95.as_secs_f64() * 1_000_000.0,
        benchmark.latency.p99.as_secs_f64() * 1_000_000.0,
        benchmark.cost.context_tokens,
        benchmark.cost.prompt_tokens,
        benchmark.cost.completion_tokens,
        benchmark.cost.total_tokens,
        benchmark.cost.estimated_spend_usd,
        benchmark.total_cases,
        benchmark.total_time_secs,
        reproducibility.map_or(0, |summary| summary.runs),
        reproducibility.map_or(0.0, |summary| summary.max_relative_delta),
        reproducibility.is_some_and(|summary| summary.materially_similar),
    )
}

fn write_cognitive_json(result: &CognitiveSuiteResult, w: &mut dyn Write) -> io::Result<()> {
    let json = serde_json::to_string_pretty(result)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    writeln!(w, "{json}")
}

fn write_cognitive_csv(result: &CognitiveSuiteResult, w: &mut dyn Write) -> io::Result<()> {
    let summary_strategy = result
        .results
        .first()
        .map(|entry| entry.strategy.as_str())
        .unwrap_or(DEFAULT_COGNITIVE_STRATEGY_NAME);
    writeln!(
        w,
        "benchmark,strategy,run_id,dataset_source,corpus_embedding_source,corpus_embedding_model,query_embedding_source,query_embedding_model,retrieval_profile,execution_surface,active_surfaces,disabled_surfaces,os,arch,logical_cpus,\
            containment,token_f1,recall_accuracy,mrr,ndcg,fpr,exec_p50_us,exec_p95_us,exec_p99_us,eval_p50_us,eval_p95_us,eval_p99_us,e2e_p50_us,e2e_p95_us,e2e_p99_us,\
            context_tokens,prompt_tokens,total_tokens,total_queries,ingest_time_s,query_time_s,total_time_s,\
            repro_runs,repro_threshold,repro_max_relative_delta,repro_materially_similar"
    )?;
    for r in &result.results {
        write_cognitive_csv_row(r, result, w)?;
        for baseline in &r.baselines {
            write_cognitive_csv_row(baseline, result, w)?;
        }
    }
    // Aggregate summary row.
    writeln!(
        w,
        "TOTAL,{},{},{},{},{},{},{},{},{},{},{},{},{},{},\\
         {:.4},{:.4},{:.4},0.0000,0.0000,0.0000,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0,0,0,{},0.000,0.000,{:.3},0,0.0000,0.0000,false",
        summary_strategy,
        result.run_id,
        result.metadata.dataset_source,
        result.metadata.corpus_embedding_source,
        result.metadata.embedding_model_label,
        result.metadata.query_embedding_source,
        result.metadata.query_embedding_model_label,
        result.metadata.retrieval_profile,
        result.metadata.execution_surface,
        format_surface_labels(
            result.metadata.active_retrieval_surfaces.enabled_labels(),
            "+"
        ),
        format_surface_labels(
            result.metadata.active_retrieval_surfaces.disabled_labels(),
            "+"
        ),
        result.metadata.environment.os,
        result.metadata.environment.arch,
        result.metadata.environment.logical_cpus,
        result.final_score,
        result.geometric_mean,
        result.min_suite_score,
        result
            .results
            .iter()
            .map(|r| r.total_queries)
            .sum::<usize>(),
        result.total_time_secs,
    )?;
    Ok(())
}

fn write_cognitive_markdown(result: &CognitiveSuiteResult, w: &mut dyn Write) -> io::Result<()> {
    use crate::cognitive::Benchmark;
    use crate::cognitive::baselines;

    writeln!(w, "# Cognitive Memory Benchmark Report")?;
    writeln!(w)?;
    writeln!(w, "**Run ID:** {}", result.run_id)?;
    writeln!(w, "**Total time:** {:.2}s", result.total_time_secs)?;
    writeln!(w, "**Final Score:** {:.1}%", result.final_score * 100.0)?;
    writeln!(
        w,
        "**Geometric Mean:** {:.1}%",
        result.geometric_mean * 100.0
    )?;
    writeln!(
        w,
        "**Min Suite Score:** {:.1}%",
        result.min_suite_score * 100.0
    )?;
    writeln!(
        w,
        "**All Competitive:** {}",
        if result.all_competitive { "✓" } else { "✗" }
    )?;
    writeln!(w)?;

    writeln!(w, "## Run Metadata")?;
    writeln!(w)?;
    writeln!(w, "| Field | Value |")?;
    writeln!(w, "|-------|:------|")?;
    writeln!(
        w,
        "| Generated at | {} |",
        result.metadata.generated_at_rfc3339
    )?;
    writeln!(w, "| Dataset source | {} |", result.metadata.dataset_source)?;
    writeln!(
        w,
        "| Corpus embedding source | {} |",
        result.metadata.corpus_embedding_source
    )?;
    writeln!(
        w,
        "| Corpus embedding model | {} |",
        result.metadata.embedding_model_label
    )?;
    writeln!(
        w,
        "| Query embedding source | {} |",
        result.metadata.query_embedding_source
    )?;
    writeln!(
        w,
        "| Query embedding model | {} |",
        result.metadata.query_embedding_model_label
    )?;
    writeln!(w, "| Embedding dims | {} |", result.metadata.embedding_dims)?;
    writeln!(w, "| Token budget | {} |", result.metadata.token_budget)?;
    writeln!(w, "| Top-K | {} |", result.metadata.k)?;
    writeln!(
        w,
        "| Retrieval profile | {} |",
        result.metadata.retrieval_profile
    )?;
    writeln!(
        w,
        "| Execution surface | {} |",
        result.metadata.execution_surface
    )?;
    writeln!(
        w,
        "| Query-text hybrid | {} |",
        if result.metadata.query_text_hybrid {
            "enabled"
        } else {
            "disabled"
        }
    )?;
    writeln!(
        w,
        "| Active retrieval surfaces | {} |",
        format_active_retrieval_surfaces(&result.metadata.active_retrieval_surfaces)
    )?;
    writeln!(w, "| Runs | {} |", result.metadata.runs)?;
    if let Some(scale) = result.metadata.synthetic_scale {
        writeln!(w, "| Synthetic scale | {} |", scale)?;
    }
    let environment_label = result.metadata.environment.label.as_deref().unwrap_or("-");
    let environment_image = result.metadata.environment.image.as_deref().unwrap_or("-");
    writeln!(w, "| Environment label | {} |", environment_label)?;
    writeln!(w, "| Environment image | {} |", environment_image)?;
    writeln!(
        w,
        "| Platform | {}/{} |",
        result.metadata.environment.os, result.metadata.environment.arch
    )?;
    writeln!(
        w,
        "| Logical CPUs | {} |",
        result.metadata.environment.logical_cpus
    )?;
    let git_commit_sha = result
        .metadata
        .environment
        .git_commit_sha
        .as_deref()
        .unwrap_or("-");
    let cargo_lock_blake3 = result
        .metadata
        .environment
        .cargo_lock_blake3
        .as_deref()
        .unwrap_or("-");
    writeln!(w, "| Git commit | {} |", git_commit_sha)?;
    writeln!(w, "| Cargo.lock blake3 | {} |", cargo_lock_blake3)?;
    writeln!(
        w,
        "| Baseline strategies | {} |",
        if result.metadata.baseline_strategies.is_empty() {
            "disabled".to_string()
        } else {
            result.metadata.baseline_strategies.join(", ")
        }
    )?;
    writeln!(w)?;

    // Summary table with SOTA comparison.
    writeln!(w, "## Summary")?;
    writeln!(w)?;
    writeln!(
        w,
        "| Benchmark | Containment | Recall Acc. | MRR | nDCG | FPR | Exec p50 | Exec p95 | Exec p99 | Total tokens | SOTA Target | Status |"
    )?;
    writeln!(
        w,
        "|-----------|------------:|------------:|----:|-----:|----:|----:|----:|----:|-------------:|:------------|:-------|"
    )?;
    for r in &result.results {
        let bench: Result<Benchmark, _> = r
            .benchmark
            .split(" (")
            .next()
            .unwrap_or(&r.benchmark)
            .parse();
        let (target_desc, status) = if let Ok(b) = bench {
            let t = baselines::target(b);
            let competitive = baselines::is_competitive(b, r.overall_containment);
            let status = if competitive { "✓" } else { "✗" };
            (t.description.to_string(), status.to_string())
        } else {
            ("-".to_string(), "-".to_string())
        };
        writeln!(
            w,
            "| {} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {:.1} ms | {:.1} ms | {:.1} ms | {} | {} | {} |",
            r.benchmark,
            r.overall_containment,
            r.overall_recall_accuracy,
            r.overall_mrr,
            r.overall_ndcg,
            r.false_positive_rate,
            r.execution_latency.p50.as_secs_f64() * 1_000.0,
            r.execution_latency.p95.as_secs_f64() * 1_000.0,
            r.execution_latency.p99.as_secs_f64() * 1_000.0,
            r.token_cost.total_tokens,
            target_desc,
            status,
        )?;
    }
    writeln!(w)?;

    writeln!(w, "## Strategy Comparisons")?;
    writeln!(w)?;
    for r in &result.results {
        writeln!(w, "### {}", r.benchmark)?;
        writeln!(w)?;
        writeln!(
            w,
            "| Strategy | Containment | Token F1 | Recall Acc. | MRR | nDCG | FPR | Exec p50 | Exec p95 | Exec p99 | Total tokens | Delta containment | Delta Exec p95 | Delta tokens | Reproducibility |"
        )?;
        writeln!(
            w,
            "|----------|------------:|---------:|------------:|----:|-----:|----:|----:|----:|----:|-------------:|------------------:|----------:|-------------:|:----------------|"
        )?;
        writeln!(
            w,
            "| {} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {:.1} ms | {:.1} ms | {:.1} ms | {} | - | - | - | {} |",
            r.strategy,
            r.overall_containment,
            r.overall_token_f1,
            r.overall_recall_accuracy,
            r.overall_mrr,
            r.overall_ndcg,
            r.false_positive_rate,
            r.execution_latency.p50.as_secs_f64() * 1_000.0,
            r.execution_latency.p95.as_secs_f64() * 1_000.0,
            r.execution_latency.p99.as_secs_f64() * 1_000.0,
            r.token_cost.total_tokens,
            format_reproducibility(r.reproducibility.as_ref()),
        )?;
        for baseline in &r.baselines {
            writeln!(
                w,
                "| {} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {:.1} ms | {:.1} ms | {:.1} ms | {} | {:+.4} | {:+.1} ms | {:+} | {} |",
                baseline.strategy,
                baseline.overall_containment,
                baseline.overall_token_f1,
                baseline.overall_recall_accuracy,
                baseline.overall_mrr,
                baseline.overall_ndcg,
                baseline.false_positive_rate,
                baseline.execution_latency.p50.as_secs_f64() * 1_000.0,
                baseline.execution_latency.p95.as_secs_f64() * 1_000.0,
                baseline.execution_latency.p99.as_secs_f64() * 1_000.0,
                baseline.token_cost.total_tokens,
                r.overall_containment - baseline.overall_containment,
                (r.execution_latency.p95.as_secs_f64()
                    - baseline.execution_latency.p95.as_secs_f64())
                    * 1_000.0,
                r.token_cost.total_tokens as i64 - baseline.token_cost.total_tokens as i64,
                format_reproducibility(baseline.reproducibility.as_ref()),
            )?;
        }
        writeln!(w)?;
        for baseline in &r.baselines {
            writeln!(
                w,
                "Strategy note ({}): {}",
                baseline.strategy,
                baseline_description(&baseline.strategy),
            )?;
        }
        if !r.baselines.is_empty() {
            writeln!(w)?;
        }
        if r.baselines.is_empty() {
            writeln!(w, "Executable baselines were disabled for this run.")?;
            writeln!(w)?;
        }
        writeln!(w, "Benchmark latencies:")?;
        writeln!(w, "| Component | p50 | p95 | p99 | mean |")?;
        writeln!(w, "|-----------|----:|----:|----:|-----:|")?;
        for (label, stats) in [
            ("execution", &r.execution_latency),
            ("evaluation", &r.evaluation_latency),
            ("end-to-end", &r.end_to_end_latency),
        ] {
            writeln!(
                w,
                "| {} | {:.1} ms | {:.1} ms | {:.1} ms | {:.1} ms |",
                label,
                stats.p50.as_secs_f64() * 1_000.0,
                stats.p95.as_secs_f64() * 1_000.0,
                stats.p99.as_secs_f64() * 1_000.0,
                stats.mean.as_secs_f64() * 1_000.0,
            )?;
        }
        writeln!(w)?;
        if let Some(compiled_phase_timings) = r.compiled_phase_timings.as_ref() {
            writeln!(w, "Compiled phase timings:")?;
            writeln!(w, "| Phase | p50 | p95 | p99 | mean |")?;
            writeln!(w, "|-------|----:|----:|----:|-----:|")?;
            for (label, stats) in [
                ("embed", &compiled_phase_timings.embed),
                ("optimize", &compiled_phase_timings.optimize),
                ("physical-plan", &compiled_phase_timings.physical_plan),
                ("execute-plan", &compiled_phase_timings.execute_plan),
                ("decode", &compiled_phase_timings.decode),
                ("assemble", &compiled_phase_timings.assemble),
                ("total", &compiled_phase_timings.total),
            ] {
                writeln!(
                    w,
                    "| {} | {:.1} ms | {:.1} ms | {:.1} ms | {:.1} ms |",
                    label,
                    stats.p50.as_secs_f64() * 1_000.0,
                    stats.p95.as_secs_f64() * 1_000.0,
                    stats.p99.as_secs_f64() * 1_000.0,
                    stats.mean.as_secs_f64() * 1_000.0,
                )?;
            }
            writeln!(w)?;
        }
    }

    let reproducibility_rows: Vec<(&str, &str, &crate::cognitive::ReproducibilitySummary)> = result
        .results
        .iter()
        .flat_map(|r| {
            let mut rows = Vec::new();
            if let Some(repro) = r.reproducibility.as_ref() {
                rows.push((r.benchmark.as_str(), r.strategy.as_str(), repro));
            }
            for baseline in &r.baselines {
                if let Some(repro) = baseline.reproducibility.as_ref() {
                    rows.push((r.benchmark.as_str(), baseline.strategy.as_str(), repro));
                }
            }
            rows
        })
        .collect();
    if !reproducibility_rows.is_empty() {
        writeln!(w, "## Reproducibility")?;
        writeln!(w)?;
        writeln!(
            w,
            "| Benchmark | Strategy | Runs | Threshold | Max drift | Mean drift | Status |"
        )?;
        writeln!(
            w,
            "|-----------|----------|-----:|----------:|----------:|-----------:|:-------|"
        )?;
        for (benchmark, strategy, repro) in reproducibility_rows {
            writeln!(
                w,
                "| {} | {} | {} | {:.2}% | {:.2}% | {:.2}% | {} |",
                benchmark,
                strategy,
                repro.runs,
                repro.threshold * 100.0,
                repro.max_relative_delta * 100.0,
                repro.mean_relative_delta * 100.0,
                if repro.materially_similar {
                    "materially similar"
                } else {
                    "drift exceeds threshold"
                },
            )?;
        }
        writeln!(w)?;
    }

    // SOTA baselines reference.
    writeln!(w, "## Reference Baselines (RFC §10)")?;
    writeln!(w)?;
    writeln!(w, "| Benchmark | System | Score | Source |")?;
    writeln!(w, "|-----------|--------|------:|--------|")?;
    for &b in Benchmark::all() {
        for bl in baselines::baselines(b) {
            if bl.score > 0.0 {
                writeln!(
                    w,
                    "| {} | {} | {:.1}% | {} |",
                    b.name(),
                    bl.system,
                    bl.score * 100.0,
                    bl.source,
                )?;
            }
        }
    }
    writeln!(w)?;

    // Per-benchmark category breakdown.
    for r in &result.results {
        writeln!(w, "## {}", r.benchmark)?;
        writeln!(w)?;
        writeln!(
            w,
            "| Category | Containment | Token F1 | Recall Acc. | MRR | nDCG | FPR | Queries |"
        )?;
        writeln!(
            w,
            "|----------|------------:|---------:|------------:|----:|-----:|----:|--------:|"
        )?;
        for cat in &r.categories {
            writeln!(
                w,
                "| {} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {} |",
                cat.name,
                cat.containment,
                cat.token_f1,
                cat.recall_accuracy,
                cat.mrr,
                cat.ndcg,
                cat.false_positive_rate,
                cat.total,
            )?;
        }
        writeln!(w)?;
    }

    Ok(())
}

fn write_cognitive_csv_row(
    result: &CognitiveResult,
    suite: &CognitiveSuiteResult,
    w: &mut dyn Write,
) -> io::Result<()> {
    let repro = result.reproducibility.as_ref();
    writeln!(
        w,
        "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{},{},{},{},{:.3},{:.3},{:.3},{},{:.4},{:.4},{}",
        result.benchmark,
        result.strategy,
        result.run_id,
        suite.metadata.dataset_source,
        suite.metadata.corpus_embedding_source,
        suite.metadata.embedding_model_label,
        suite.metadata.query_embedding_source,
        suite.metadata.query_embedding_model_label,
        suite.metadata.retrieval_profile,
        suite.metadata.execution_surface,
        format_surface_labels(
            suite.metadata.active_retrieval_surfaces.enabled_labels(),
            "+"
        ),
        format_surface_labels(
            suite.metadata.active_retrieval_surfaces.disabled_labels(),
            "+"
        ),
        suite.metadata.environment.os,
        suite.metadata.environment.arch,
        suite.metadata.environment.logical_cpus,
        result.overall_containment,
        result.overall_token_f1,
        result.overall_recall_accuracy,
        result.overall_mrr,
        result.overall_ndcg,
        result.false_positive_rate,
        result.execution_latency.p50.as_secs_f64() * 1e6,
        result.execution_latency.p95.as_secs_f64() * 1e6,
        result.execution_latency.p99.as_secs_f64() * 1e6,
        result.evaluation_latency.p50.as_secs_f64() * 1e6,
        result.evaluation_latency.p95.as_secs_f64() * 1e6,
        result.evaluation_latency.p99.as_secs_f64() * 1e6,
        result.end_to_end_latency.p50.as_secs_f64() * 1e6,
        result.end_to_end_latency.p95.as_secs_f64() * 1e6,
        result.end_to_end_latency.p99.as_secs_f64() * 1e6,
        result.token_cost.context_tokens,
        result.token_cost.prompt_tokens,
        result.token_cost.total_tokens,
        result.total_queries,
        result.ingest_time_secs,
        result.query_time_secs,
        result.total_time_secs,
        repro.map_or(0, |summary| summary.runs),
        repro.map_or(0.0, |summary| summary.threshold),
        repro.map_or(0.0, |summary| summary.max_relative_delta),
        repro.is_some_and(|summary| summary.materially_similar),
    )
}

fn format_active_retrieval_surfaces(surfaces: &ActiveRetrievalSurfaces) -> String {
    let enabled = format_surface_labels(surfaces.enabled_labels(), ", ");
    let disabled = format_surface_labels(surfaces.disabled_labels(), ", ");

    if surfaces.notes.is_empty() {
        format!("enabled: {enabled}; disabled: {disabled}")
    } else {
        format!(
            "enabled: {enabled}; disabled: {disabled}; notes: {}",
            surfaces.notes.join(", ")
        )
    }
}

fn format_surface_labels(labels: Vec<&'static str>, separator: &str) -> String {
    if labels.is_empty() {
        "none".to_string()
    } else {
        labels.join(separator)
    }
}

fn baseline_description(strategy: &str) -> &'static str {
    match strategy {
        "full-context" => BaselineStrategy::FullContext.description(),
        "iterative-retrieval" => BaselineStrategy::IterativeRetrieval.description(),
        _ => "Primary hirn retrieval path",
    }
}

fn format_reproducibility(repro: Option<&crate::cognitive::ReproducibilitySummary>) -> String {
    match repro {
        Some(summary) => {
            let status = if summary.materially_similar {
                "similar"
            } else {
                "drift"
            };
            format!(
                "{} runs, max {:.2}% ({})",
                summary.runs,
                summary.max_relative_delta * 100.0,
                status,
            )
        }
        None => "single run".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::advanced::{
        AdvancedCostEnvelope, AdvancedMetadata, AdvancedQualityMetrics, AdvancedResult,
        AdvancedSuiteResult,
    };
    use crate::cognitive::{
        ActiveRetrievalSurfaces, BenchmarkExecutionSurface, BenchmarkRetrievalProfile,
        CognitiveResult, CognitiveSuiteResult, EnvironmentInfo, ReproducibilitySummary,
        SuiteMetadata, TokenCostEstimate,
    };
    use crate::metrics::*;
    use std::time::Duration;

    fn sample_result() -> BenchmarkResult {
        BenchmarkResult {
            suite_name: "test".to_string(),
            run_id: "run-001".to_string(),
            config: BenchmarkConfig::default(),
            query_metrics: vec![QueryMetrics {
                precision_at_k: 0.8,
                recall_at_k: 0.6,
                f1: 0.6857,
                mrr: 1.0,
                ndcg_at_k: 0.9,
            }],
            aggregate: AggregateQuality {
                mean_precision: 0.8,
                mean_recall: 0.6,
                mean_f1: 0.6857,
                mean_mrr: 1.0,
                mean_ndcg: 0.9,
            },
            remember_latency: LatencyStats {
                p50: Duration::from_micros(100),
                p95: Duration::from_micros(500),
                p99: Duration::from_millis(1),
                min: Duration::from_micros(50),
                max: Duration::from_millis(2),
                mean: Duration::from_micros(150),
            },
            recall_latency: LatencyStats::default(),
            think_latency: LatencyStats::default(),
            throughput: ThroughputStats {
                remember_ops_per_sec: 5000.0,
                recall_ops_per_sec: 2000.0,
                think_ops_per_sec: 500.0,
            },
            peak_memory_bytes: 10_485_760,
            db_file_size_bytes: 1_048_576,
            total_time: Duration::from_secs(5),
        }
    }

    fn sample_latency() -> LatencyStats {
        LatencyStats {
            p50: Duration::from_millis(2),
            p95: Duration::from_millis(5),
            p99: Duration::from_millis(8),
            min: Duration::from_millis(1),
            max: Duration::from_millis(9),
            mean: Duration::from_millis(4),
        }
    }

    fn sample_cognitive_result(strategy: &str, total_tokens: usize) -> CognitiveResult {
        CognitiveResult {
            benchmark: "h1-retrieval".to_string(),
            strategy: strategy.to_string(),
            run_id: "cog-run-001".to_string(),
            categories: vec![crate::cognitive::CategoryScore {
                name: "retrieval".to_string(),
                containment: 0.85,
                token_f1: 0.44,
                recall_accuracy: 0.9,
                mrr: 0.81,
                ndcg: 0.84,
                semantic_similarity: 0.73,
                false_positive_rate: 0.0,
                total: 10,
            }],
            overall_containment: 0.85,
            overall_token_f1: 0.44,
            overall_recall_accuracy: 0.9,
            overall_mrr: 0.81,
            overall_ndcg: 0.84,
            overall_semantic_similarity: 0.73,
            false_positive_rate: 0.0,
            execution_latency: sample_latency(),
            evaluation_latency: sample_latency(),
            end_to_end_latency: sample_latency(),
            token_cost: TokenCostEstimate::from_totals(
                total_tokens.saturating_sub(100),
                total_tokens,
                0,
                10,
            ),
            total_queries: 10,
            ingest_time_secs: 0.2,
            query_time_secs: 0.4,
            total_time_secs: 0.6,
            compiled_phase_timings: None,
            baselines: vec![],
            reproducibility: Some(ReproducibilitySummary {
                runs: 2,
                threshold: 0.05,
                materially_similar: true,
                max_relative_delta: 0.01,
                mean_relative_delta: 0.005,
                metrics: vec![],
            }),
            embedding_cache_miss_count: 0,
        }
    }

    fn sample_cognitive_suite() -> CognitiveSuiteResult {
        let mut primary = sample_cognitive_result("hirn", 640);
        primary.compiled_phase_timings = Some(crate::cognitive::CompiledPhaseTimingSummary {
            optimize: sample_latency(),
            physical_plan: sample_latency(),
            execute_plan: sample_latency(),
            embed: sample_latency(),
            decode: sample_latency(),
            assemble: sample_latency(),
            total: sample_latency(),
        });
        primary.baselines = vec![
            sample_cognitive_result("full-context", 2_560),
            sample_cognitive_result("iterative-retrieval", 1_120),
        ];

        CognitiveSuiteResult {
            run_id: "cog-run-001".to_string(),
            metadata: SuiteMetadata {
                generated_at_rfc3339: "2026-05-04T12:00:00+00:00".to_string(),
                dataset_source: "synthetic".to_string(),
                corpus_embedding_source: "cache:embeddings/all_embeddings.json".to_string(),
                embedding_model_label: "text-embedding-3-small".to_string(),
                query_embedding_source: crate::cognitive::QueryEmbeddingSource::Cache,
                query_embedding_model_label: "text-embedding-3-small".to_string(),
                embedding_dims: 1536,
                token_budget: 4096,
                k: 10,
                retrieval_profile: BenchmarkRetrievalProfile::NormalFullStack,
                execution_surface: BenchmarkExecutionSurface::DirectBuilders,
                query_text_hybrid: true,
                active_retrieval_surfaces: ActiveRetrievalSurfaces {
                    query_text_hybrid: true,
                    graph_routing: true,
                    multivector: true,
                    reranker: true,
                    tokenizer: true,
                    compiled_hirnql: false,
                    quality_gate: true,
                    iterative_retrieval: false,
                    notes: vec!["benchmarks use direct recall/think builders".to_string()],
                },
                runs: 2,
                synthetic_scale: Some(1),
                baseline_strategies: vec![
                    "full-context".to_string(),
                    "iterative-retrieval".to_string(),
                ],
                environment: EnvironmentInfo {
                    label: Some("ci-runner".to_string()),
                    image: Some("ubuntu-24.04".to_string()),
                    os: "linux".to_string(),
                    arch: "x86_64".to_string(),
                    logical_cpus: 8,
                    git_commit_sha: Some("abc123".to_string()),
                    cargo_lock_blake3: Some("lockhash".to_string()),
                },
            },
            results: vec![primary],
            total_time_secs: 1.2,
            final_score: 0.85,
            geometric_mean: 0.84,
            min_suite_score: 0.85,
            all_competitive: true,
        }
    }

    fn sample_advanced_result(benchmark: &str, total_tokens: usize) -> AdvancedResult {
        AdvancedResult {
            benchmark: benchmark.to_string(),
            strategy: "hirn-advanced".to_string(),
            run_id: "advanced-run-001".to_string(),
            quality: AdvancedQualityMetrics {
                primary_score: 0.92,
                precision: 0.95,
                recall: 0.90,
                accuracy: 0.94,
                usefulness: 0.89,
            },
            latency: sample_latency(),
            cost: AdvancedCostEnvelope {
                context_tokens: total_tokens / 2,
                prompt_tokens: total_tokens / 2,
                completion_tokens: 0,
                total_tokens,
                estimated_spend_usd: 0.0,
            },
            total_cases: 3,
            total_time_secs: 0.4,
            reproducibility: Some(ReproducibilitySummary {
                runs: 2,
                threshold: 0.05,
                materially_similar: true,
                max_relative_delta: 0.02,
                mean_relative_delta: 0.01,
                metrics: vec![],
            }),
        }
    }

    fn sample_advanced_suite() -> AdvancedSuiteResult {
        AdvancedSuiteResult {
            run_id: "advanced-run-001".to_string(),
            metadata: AdvancedMetadata {
                generated_at_rfc3339: "2026-05-04T12:00:00+00:00".to_string(),
                runs: 2,
                offline_wait_ms: 5_000,
                repro_threshold: 0.15,
                environment: EnvironmentInfo {
                    label: Some("ci-runner".to_string()),
                    image: Some("ubuntu-24.04".to_string()),
                    os: "linux".to_string(),
                    arch: "x86_64".to_string(),
                    logical_cpus: 8,
                    git_commit_sha: Some("abc123".to_string()),
                    cargo_lock_blake3: Some("lockhash".to_string()),
                },
            },
            results: vec![
                sample_advanced_result("explanation-quality", 240),
                sample_advanced_result("planning-usefulness", 320),
            ],
            total_time_secs: 0.8,
            overall_primary_score: 0.92,
        }
    }

    #[test]
    fn json_output_parseable() {
        let result = sample_result();
        let mut buf = Vec::new();
        write_result(&result, OutputFormat::Json, &mut buf).unwrap();
        let s = String::from_utf8(buf).unwrap();
        let _: serde_json::Value = serde_json::from_str(&s).unwrap();
    }

    #[test]
    fn csv_has_header_and_row() {
        let result = sample_result();
        let mut buf = Vec::new();
        write_result(&result, OutputFormat::Csv, &mut buf).unwrap();
        let s = String::from_utf8(buf).unwrap();
        let lines: Vec<&str> = s.lines().collect();
        assert_eq!(lines.len(), 2);
        assert!(lines[0].starts_with("suite,"));
        assert!(lines[1].starts_with("test,"));
    }

    #[test]
    fn csv_field_count_matches() {
        let result = sample_result();
        let mut buf = Vec::new();
        write_result(&result, OutputFormat::Csv, &mut buf).unwrap();
        let s = String::from_utf8(buf).unwrap();
        let lines: Vec<&str> = s.lines().collect();
        let header_fields = lines[0].split(',').count();
        let data_fields = lines[1].split(',').count();
        assert_eq!(header_fields, data_fields);
    }

    #[test]
    fn markdown_has_headers() {
        let result = sample_result();
        let mut buf = Vec::new();
        write_result(&result, OutputFormat::Markdown, &mut buf).unwrap();
        let s = String::from_utf8(buf).unwrap();
        assert!(s.contains("# Benchmark Report:"));
        assert!(s.contains("## Retrieval Quality"));
        assert!(s.contains("## Latency"));
        assert!(s.contains("## Throughput"));
        assert!(s.contains("## Resource Usage"));
    }

    #[test]
    fn markdown_contains_metrics() {
        let result = sample_result();
        let mut buf = Vec::new();
        write_result(&result, OutputFormat::Markdown, &mut buf).unwrap();
        let s = String::from_utf8(buf).unwrap();
        assert!(s.contains("0.8000"));
        assert!(s.contains("0.6000"));
        assert!(s.contains("Peak RSS"));
    }

    #[test]
    fn output_format_parse() {
        assert!(matches!(
            "json".parse::<OutputFormat>(),
            Ok(OutputFormat::Json)
        ));
        assert!(matches!(
            "CSV".parse::<OutputFormat>(),
            Ok(OutputFormat::Csv)
        ));
        assert!(matches!(
            "markdown".parse::<OutputFormat>(),
            Ok(OutputFormat::Markdown)
        ));
        assert!(matches!(
            "md".parse::<OutputFormat>(),
            Ok(OutputFormat::Markdown)
        ));
        assert!("xml".parse::<OutputFormat>().is_err());
    }

    #[test]
    fn cognitive_markdown_includes_metadata_and_baselines() {
        let result = sample_cognitive_suite();
        let mut buf = Vec::new();
        write_cognitive_result(&result, OutputFormat::Markdown, &mut buf).unwrap();
        let markdown = String::from_utf8(buf).unwrap();
        assert!(markdown.contains("## Run Metadata"));
        assert!(markdown.contains("## Strategy Comparisons"));
        assert!(markdown.contains("full-context"));
        assert!(markdown.contains("iterative-retrieval"));
        assert!(markdown.contains("## Reproducibility"));
        assert!(markdown.contains("text-embedding-3-small"));
        assert!(markdown.contains("Query embedding source"));
        assert!(markdown.contains("Query embedding model"));
        assert!(markdown.contains("Execution surface"));
        assert!(markdown.contains("direct-builders"));
        assert!(markdown.contains("Query-text hybrid"));
        assert!(markdown.contains("Git commit"));
        assert!(markdown.contains("Cargo.lock blake3"));
        assert!(markdown.contains("Benchmark latencies:"));
        assert!(markdown.contains("end-to-end"));
        assert!(markdown.contains("Compiled phase timings:"));
        assert!(markdown.contains("physical-plan"));
        assert!(markdown.contains("execute-plan"));
        assert!(markdown.contains("assemble"));
        assert!(markdown.contains("total"));
    }

    #[test]
    fn cognitive_json_includes_compiled_phase_timings() {
        let result = sample_cognitive_suite();
        let mut buf = Vec::new();
        write_cognitive_result(&result, OutputFormat::Json, &mut buf).unwrap();
        let json = String::from_utf8(buf).unwrap();
        assert!(json.contains("\"compiled_phase_timings\""));
        assert!(json.contains("\"execution_latency\""));
        assert!(json.contains("\"evaluation_latency\""));
        assert!(json.contains("\"end_to_end_latency\""));
        assert!(json.contains("\"optimize\""));
        assert!(json.contains("\"physical_plan\""));
        assert!(json.contains("\"execute_plan\""));
        assert!(json.contains("\"assemble\""));
        assert!(json.contains("\"total\""));
    }

    #[test]
    fn cognitive_csv_emits_strategy_rows() {
        let result = sample_cognitive_suite();
        let mut buf = Vec::new();
        write_cognitive_result(&result, OutputFormat::Csv, &mut buf).unwrap();
        let csv = String::from_utf8(buf).unwrap();
        assert!(csv.lines().next().unwrap().contains("strategy"));
        assert!(
            csv.lines()
                .next()
                .unwrap()
                .contains("query_embedding_source")
        );
        assert!(csv.contains("normal-full-stack"));
        assert!(csv.contains("direct-builders"));
        assert!(csv.contains("hybrid+graph+multivector+reranker+tokenizer+quality-gate"));
        assert!(csv.contains("full-context"));
        assert!(csv.contains("iterative-retrieval"));
        assert!(csv.lines().any(|line| line.starts_with("TOTAL,")));
    }

    #[test]
    fn advanced_markdown_includes_sections() {
        let result = sample_advanced_suite();
        let mut buf = Vec::new();
        write_advanced_result(&result, OutputFormat::Markdown, &mut buf).unwrap();
        let markdown = String::from_utf8(buf).unwrap();
        assert!(markdown.contains("# Advanced Offline Cognition Benchmark Report"));
        assert!(markdown.contains("## Run Metadata"));
        assert!(markdown.contains("## Overview"));
        assert!(markdown.contains("## explanation-quality"));
        assert!(markdown.contains("### Cost Envelope"));
        assert!(markdown.contains("## Reproducibility"));
        assert!(markdown.contains("Generated at"));
        assert!(markdown.contains("Git commit"));
    }

    #[test]
    fn advanced_csv_emits_rows() {
        let result = sample_advanced_suite();
        let mut buf = Vec::new();
        write_advanced_result(&result, OutputFormat::Csv, &mut buf).unwrap();
        let csv = String::from_utf8(buf).unwrap();
        assert!(csv.lines().next().unwrap().contains("primary_score"));
        assert!(csv.contains("explanation-quality"));
        assert!(csv.contains("planning-usefulness"));
        assert_eq!(csv.lines().count(), 4);
    }
}
