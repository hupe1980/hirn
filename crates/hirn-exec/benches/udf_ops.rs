use std::sync::Arc;

use arrow_array::{Float32Array, Float64Array, StringArray, UInt32Array, UInt64Array};
use arrow_schema::{DataType, Field};
use criterion::{Criterion, black_box, criterion_group, criterion_main};
use datafusion_common::config::ConfigOptions;
use datafusion_expr::{ColumnarValue, ScalarFunctionArgs, ScalarUDFImpl};

use hirn_exec::udfs::{
    CausalRelevanceUdf, CompositeScoreUdf, FadeMemDecayUdf, RpeScoreUdf, SourceReliabilityUdf,
    SurpriseScoreUdf, TemporalDecayUdf, TokenCountUdf,
};

const ROWS: usize = 100_000;

fn config_options() -> Arc<ConfigOptions> {
    Arc::new(ConfigOptions::new())
}

// ---------------------------------------------------------------------------
// composite_score(similarity, importance, age_hours, activation, causal, surprise) → Float32
// ---------------------------------------------------------------------------
fn bench_composite_score(c: &mut Criterion) {
    let udf = CompositeScoreUdf::new();

    let sim: Vec<f32> = (0..ROWS).map(|i| i as f32 / ROWS as f32).collect();
    let imp: Vec<f32> = (0..ROWS)
        .map(|i| 0.1 + 0.8 * (i as f32 / ROWS as f32))
        .collect();
    let age: Vec<f64> = (0..ROWS).map(|i| (i as f64) * 0.5).collect();
    let act: Vec<f32> = (0..ROWS).map(|i| (i % 100) as f32 / 100.0).collect();
    let caus: Vec<f32> = (0..ROWS).map(|i| (i % 50) as f32 / 50.0).collect();
    let surp: Vec<f32> = (0..ROWS).map(|i| (i % 200) as f32 / 200.0).collect();

    let args = ScalarFunctionArgs {
        args: vec![
            ColumnarValue::Array(Arc::new(Float32Array::from(sim))),
            ColumnarValue::Array(Arc::new(Float32Array::from(imp))),
            ColumnarValue::Array(Arc::new(Float64Array::from(age))),
            ColumnarValue::Array(Arc::new(Float32Array::from(act))),
            ColumnarValue::Array(Arc::new(Float32Array::from(caus))),
            ColumnarValue::Array(Arc::new(Float32Array::from(surp))),
        ],
        number_rows: ROWS,
        return_field: Arc::new(Field::new("r", DataType::Float32, true)),
        arg_fields: vec![],
        config_options: config_options(),
    };

    c.bench_function("composite_score_100k", |b| {
        b.iter(|| {
            let a = args.clone();
            black_box(udf.invoke_with_args(a).unwrap());
        });
    });
}

// ---------------------------------------------------------------------------
// temporal_decay(age_hours, base_rate, access_freq, importance) → Float32
// ---------------------------------------------------------------------------
fn bench_temporal_decay(c: &mut Criterion) {
    let udf = TemporalDecayUdf::new();

    let age: Vec<f64> = (0..ROWS).map(|i| (i as f64) * 0.5).collect();
    let base: Vec<f64> = vec![0.004; ROWS];
    let freq: Vec<u64> = (0..ROWS).map(|i| (i % 20) as u64).collect();
    let imp: Vec<f32> = (0..ROWS).map(|i| (i % 100) as f32 / 100.0).collect();

    let args = ScalarFunctionArgs {
        args: vec![
            ColumnarValue::Array(Arc::new(Float64Array::from(age))),
            ColumnarValue::Array(Arc::new(Float64Array::from(base))),
            ColumnarValue::Array(Arc::new(UInt64Array::from(freq))),
            ColumnarValue::Array(Arc::new(Float32Array::from(imp))),
        ],
        number_rows: ROWS,
        return_field: Arc::new(Field::new("r", DataType::Float32, true)),
        arg_fields: vec![],
        config_options: config_options(),
    };

    c.bench_function("temporal_decay_100k", |b| {
        b.iter(|| {
            let a = args.clone();
            black_box(udf.invoke_with_args(a).unwrap());
        });
    });
}

// ---------------------------------------------------------------------------
// rpe_score(max_similarity, novelty_zscore) → Float32
// ---------------------------------------------------------------------------
fn bench_rpe_score(c: &mut Criterion) {
    let udf = RpeScoreUdf::new();

    let sim: Vec<f32> = (0..ROWS).map(|i| i as f32 / ROWS as f32).collect();
    let zscore: Vec<f32> = (0..ROWS)
        .map(|i| -2.0 + 4.0 * (i as f32 / ROWS as f32))
        .collect();

    let args = ScalarFunctionArgs {
        args: vec![
            ColumnarValue::Array(Arc::new(Float32Array::from(sim))),
            ColumnarValue::Array(Arc::new(Float32Array::from(zscore))),
        ],
        number_rows: ROWS,
        return_field: Arc::new(Field::new("r", DataType::Float32, true)),
        arg_fields: vec![],
        config_options: config_options(),
    };

    c.bench_function("rpe_score_100k", |b| {
        b.iter(|| {
            let a = args.clone();
            black_box(udf.invoke_with_args(a).unwrap());
        });
    });
}

// ---------------------------------------------------------------------------
// source_reliability(source_type) → Float32
// ---------------------------------------------------------------------------
fn bench_source_reliability(c: &mut Criterion) {
    let udf = SourceReliabilityUdf::new();

    let sources = [
        "direct_observation",
        "agent_generated",
        "inferred",
        "cross_agent",
        "unknown",
    ];
    let vals: Vec<&str> = (0..ROWS).map(|i| sources[i % sources.len()]).collect();

    let args = ScalarFunctionArgs {
        args: vec![ColumnarValue::Array(Arc::new(StringArray::from(vals)))],
        number_rows: ROWS,
        return_field: Arc::new(Field::new("r", DataType::Float32, true)),
        arg_fields: vec![],
        config_options: config_options(),
    };

    c.bench_function("source_reliability_100k", |b| {
        b.iter(|| {
            let a = args.clone();
            black_box(udf.invoke_with_args(a).unwrap());
        });
    });
}

// ---------------------------------------------------------------------------
// surprise_score(kl_divergence) → Float32
// ---------------------------------------------------------------------------
fn bench_surprise_score(c: &mut Criterion) {
    let udf = SurpriseScoreUdf::new();

    let kl: Vec<f32> = (0..ROWS)
        .map(|i| -5.0 + 10.0 * (i as f32 / ROWS as f32))
        .collect();

    let args = ScalarFunctionArgs {
        args: vec![ColumnarValue::Array(Arc::new(Float32Array::from(kl)))],
        number_rows: ROWS,
        return_field: Arc::new(Field::new("r", DataType::Float32, true)),
        arg_fields: vec![],
        config_options: config_options(),
    };

    c.bench_function("surprise_score_100k", |b| {
        b.iter(|| {
            let a = args.clone();
            black_box(udf.invoke_with_args(a).unwrap());
        });
    });
}

// ---------------------------------------------------------------------------
// fade_mem_decay(base, importance, access_freq) → Float32
// ---------------------------------------------------------------------------
fn bench_fade_mem_decay(c: &mut Criterion) {
    let udf = FadeMemDecayUdf::new();

    let base: Vec<f64> = vec![0.004; ROWS];
    let imp: Vec<f32> = (0..ROWS).map(|i| (i % 100) as f32 / 100.0).collect();
    let freq: Vec<u64> = (0..ROWS).map(|i| (i % 20) as u64).collect();

    let args = ScalarFunctionArgs {
        args: vec![
            ColumnarValue::Array(Arc::new(Float64Array::from(base))),
            ColumnarValue::Array(Arc::new(Float32Array::from(imp))),
            ColumnarValue::Array(Arc::new(UInt64Array::from(freq))),
        ],
        number_rows: ROWS,
        return_field: Arc::new(Field::new("r", DataType::Float32, true)),
        arg_fields: vec![],
        config_options: config_options(),
    };

    c.bench_function("fade_mem_decay_100k", |b| {
        b.iter(|| {
            let a = args.clone();
            black_box(udf.invoke_with_args(a).unwrap());
        });
    });
}

// ---------------------------------------------------------------------------
// causal_relevance(strength, confidence, evidence_count) → Float32
// ---------------------------------------------------------------------------
fn bench_causal_relevance(c: &mut Criterion) {
    let udf = CausalRelevanceUdf::new();

    let str_vals: Vec<f32> = (0..ROWS)
        .map(|i| 0.1 + 0.9 * (i as f32 / ROWS as f32))
        .collect();
    let conf: Vec<f32> = (0..ROWS)
        .map(|i| 0.5 + 0.5 * (i as f32 / ROWS as f32))
        .collect();
    let evid: Vec<u32> = (0..ROWS).map(|i| (i % 50) as u32).collect();

    let args = ScalarFunctionArgs {
        args: vec![
            ColumnarValue::Array(Arc::new(Float32Array::from(str_vals))),
            ColumnarValue::Array(Arc::new(Float32Array::from(conf))),
            ColumnarValue::Array(Arc::new(UInt32Array::from(evid))),
        ],
        number_rows: ROWS,
        return_field: Arc::new(Field::new("r", DataType::Float32, true)),
        arg_fields: vec![],
        config_options: config_options(),
    };

    c.bench_function("causal_relevance_100k", |b| {
        b.iter(|| {
            let a = args.clone();
            black_box(udf.invoke_with_args(a).unwrap());
        });
    });
}

// ---------------------------------------------------------------------------
// token_count(text) → UInt32
// ---------------------------------------------------------------------------
fn bench_token_count(c: &mut Criterion) {
    let udf = TokenCountUdf::new();

    let texts: Vec<&str> = (0..ROWS)
        .map(|i| match i % 3 {
            0 => "The quick brown fox jumps over the lazy dog near the river bank.",
            1 => "Artificial intelligence is transforming how we approach complex problems.",
            _ => "Short.",
        })
        .collect();

    let args = ScalarFunctionArgs {
        args: vec![ColumnarValue::Array(Arc::new(StringArray::from(texts)))],
        number_rows: ROWS,
        return_field: Arc::new(Field::new("r", DataType::UInt32, true)),
        arg_fields: vec![],
        config_options: config_options(),
    };

    c.bench_function("token_count_100k", |b| {
        b.iter(|| {
            let a = args.clone();
            black_box(udf.invoke_with_args(a).unwrap());
        });
    });
}

criterion_group!(
    benches,
    bench_composite_score,
    bench_temporal_decay,
    bench_rpe_score,
    bench_source_reliability,
    bench_surprise_score,
    bench_fade_mem_decay,
    bench_causal_relevance,
    bench_token_count,
);
criterion_main!(benches);
