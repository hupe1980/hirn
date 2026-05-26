use criterion::{Criterion, black_box, criterion_group, criterion_main};

use hirn_query::parse;

fn bench_parse_simple_recall(c: &mut Criterion) {
    let query = r#"RECALL episodic ABOUT "deployment strategies" LIMIT 10"#;
    c.bench_function("parse_simple_recall", |b| {
        b.iter(|| parse(black_box(query)).unwrap());
    });
}

fn bench_parse_complex_think(c: &mut Criterion) {
    let query = r#"THINK ABOUT "deployment strategies for microservices" WHERE importance > 0.5 EXPAND GRAPH DEPTH 2 ACTIVATION spreading FOLLOW CAUSES DEPTH 3 BUDGET 4096 LIMIT 50"#;
    c.bench_function("parse_complex_think", |b| {
        b.iter(|| parse(black_box(query)).unwrap());
    });
}

fn bench_parse_remember(c: &mut Criterion) {
    let query = r#"REMEMBER episode CONTENT "Today we deployed the new caching layer to production. Response times dropped by 40% and cache hit rates reached 95% within the first hour." TYPE observation IMPORTANCE 0.9"#;
    c.bench_function("parse_remember", |b| {
        b.iter(|| parse(black_box(query)).unwrap());
    });
}

fn bench_parse_batch_100(c: &mut Criterion) {
    let queries: Vec<String> = (0..100)
        .map(|i| format!(r#"RECALL episodic ABOUT "topic {i}" LIMIT 10"#))
        .collect();

    c.bench_function("parse_batch_100", |b| {
        b.iter(|| {
            for q in &queries {
                let _ = parse(black_box(q)).unwrap();
            }
        });
    });
}

criterion_group!(
    benches,
    bench_parse_simple_recall,
    bench_parse_complex_think,
    bench_parse_remember,
    bench_parse_batch_100,
);
criterion_main!(benches);
