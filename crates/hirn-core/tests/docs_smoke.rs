use hirn_core::HirnConfig;

const PERFORMANCE_TUNING_DOC: &str = include_str!("../../../docs/performance-tuning.md");

fn fenced_blocks(markdown: &str, language: &str) -> Vec<String> {
    let fence = format!("```{language}");
    let mut blocks = Vec::new();
    let mut current: Option<String> = None;

    for line in markdown.lines() {
        let trimmed = line.trim();
        if current.is_none() && trimmed == fence {
            current = Some(String::new());
            continue;
        }

        if let Some(block) = &mut current {
            if trimmed == "```" {
                blocks.push(std::mem::take(block));
                current = None;
            } else {
                block.push_str(line);
                block.push('\n');
            }
        }
    }

    blocks
}

#[test]
fn performance_tuning_toml_examples_deserialize_with_live_config() {
    let snippets = fenced_blocks(PERFORMANCE_TUNING_DOC, "toml");
    assert_eq!(
        snippets.len(),
        1,
        "expected one TOML example in performance-tuning doc"
    );

    let config: HirnConfig =
        toml::from_str(&snippets[0]).expect("performance-tuning example should deserialize");

    assert!(config.rpe_enabled);
    assert!((config.rpe_fast_path_threshold - 0.35).abs() < f32::EPSILON);
    assert!((config.quality_gate_threshold - 0.45).abs() < f32::EPSILON);
    assert!((config.interference_consolidation_threshold - 0.25).abs() < f32::EPSILON);
    assert_eq!(config.activation_max_depth, 4);
    assert_eq!(config.activation_max_iterations, 12);

    let runtime = config.embedder_runtime;
    assert_eq!(runtime.batch_size, Some(32));

    let retry = runtime.retry.expect("retry config should be present");
    assert_eq!(retry.max_retries, 3);
    assert_eq!(retry.base_backoff_ms, 500);
    assert_eq!(retry.max_cumulative_timeout_ms, 10_000);

    let circuit_breaker = runtime
        .circuit_breaker
        .expect("circuit breaker config should be present");
    assert_eq!(circuit_breaker.failure_threshold, 5);
    assert_eq!(circuit_breaker.recovery_timeout_ms, 30_000);
    assert_eq!(circuit_breaker.success_threshold, 2);
}
