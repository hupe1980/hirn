//! Optional LLM QA reader and judge for external benchmarks.
//!
//! Opt-in via `--reader` / `--judge` on the `external` subcommand. The reader
//! answers each benchmark question from the SAME assembled retrieval context
//! that the harness scores with containment/token-F1, calling an
//! OpenAI-compatible chat-completions endpoint (default model `gpt-4o`). The
//! judge then scores the generated answers:
//!
//! - **LongMemEval** uses the official question-type-aware yes/no judge
//!   prompts from the LongMemEval repository (`evaluation/evaluate_qa.py`,
//!   `get_anscheck_prompt`), including the abstention variant for `_abs`
//!   question ids. The resulting metric is published as
//!   `official_reader_accuracy` and must never be conflated with the
//!   retrieval-only `containment` metric.
//! - **BEAM** answers are judged with a faithful gold-answer-cited yes/no
//!   correctness prompt (BEAM's official evaluation judges generated
//!   answers); results are labeled `beam-reader-judged`.
//!
//! Token accounting records the EXACT `prompt_tokens` / `completion_tokens`
//! from the API `usage` field per query — no estimator is involved.
//!
//! Requests are concurrency-limited and retried with exponential backoff,
//! mirroring the auth/env conventions of the embedding client in
//! [`super::openai`] (`OPENAI_API_KEY`, optional `OPENAI_BASE_URL`).

use std::sync::Mutex;
use std::time::Duration;

use serde::{Deserialize, Serialize};

/// Default reader model (official LongMemEval reader).
pub const DEFAULT_READER_MODEL: &str = "gpt-4o";
/// Default judge model (official LongMemEval judge).
pub const DEFAULT_JUDGE_MODEL: &str = "gpt-4o";
/// Default OpenAI-compatible endpoint base URL.
pub const DEFAULT_OPENAI_BASE_URL: &str = "https://api.openai.com/v1";
/// Default number of concurrent reader/judge requests.
pub const DEFAULT_READER_CONCURRENCY: usize = 4;
/// Default retry attempts after the first failure.
pub const DEFAULT_MAX_RETRIES: usize = 3;
/// Default initial retry backoff; doubles per attempt.
pub const DEFAULT_INITIAL_BACKOFF: Duration = Duration::from_millis(500);

/// Configuration for an OpenAI-compatible chat-completions client.
#[derive(Debug, Clone)]
pub struct ChatClientConfig {
    pub base_url: String,
    pub api_key: String,
    pub model: String,
    /// Sampling temperature; defaults to 0.0 for determinism.
    pub temperature: f64,
    pub max_retries: usize,
    pub initial_backoff: Duration,
}

impl ChatClientConfig {
    pub fn new(base_url: String, api_key: String, model: String, temperature: f64) -> Self {
        Self {
            base_url,
            api_key,
            model,
            temperature,
            max_retries: DEFAULT_MAX_RETRIES,
            initial_backoff: DEFAULT_INITIAL_BACKOFF,
        }
    }
}

/// Exact token usage reported by the API for one call.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatUsage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
}

/// One chat completion: generated text plus exact API usage.
#[derive(Debug, Clone)]
pub struct ChatCompletion {
    pub content: String,
    pub usage: ChatUsage,
}

/// Parse an OpenAI chat-completions response body.
///
/// Fails when the `usage` field is missing: reader/judge runs require EXACT
/// token accounting and must not silently fall back to estimates.
pub fn parse_chat_completion(response: &serde_json::Value) -> Result<ChatCompletion, String> {
    if let Some(error) = response.get("error") {
        return Err(format!("chat API error: {error}"));
    }

    let content = response["choices"]
        .get(0)
        .and_then(|choice| choice["message"]["content"].as_str())
        .ok_or("missing 'choices[0].message.content' in chat response")?
        .to_string();

    let usage = response
        .get("usage")
        .ok_or("missing 'usage' in chat response; exact token accounting is required")?;
    let prompt_tokens = usage["prompt_tokens"]
        .as_u64()
        .ok_or("missing 'usage.prompt_tokens' in chat response")? as usize;
    let completion_tokens = usage["completion_tokens"]
        .as_u64()
        .ok_or("missing 'usage.completion_tokens' in chat response")?
        as usize;

    Ok(ChatCompletion {
        content,
        usage: ChatUsage {
            prompt_tokens,
            completion_tokens,
        },
    })
}

/// Call the chat-completions endpoint with retry + exponential backoff.
///
/// Retries transport errors, HTTP 429, and 5xx responses. Other HTTP statuses
/// and API-level errors fail immediately.
pub fn chat_completion(
    config: &ChatClientConfig,
    system_prompt: &str,
    user_prompt: &str,
) -> Result<ChatCompletion, String> {
    let url = format!("{}/chat/completions", config.base_url.trim_end_matches('/'));
    let body = serde_json::json!({
        "model": config.model,
        "temperature": config.temperature,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    });

    let mut backoff = config.initial_backoff;
    let mut last_error = String::new();

    for attempt in 0..=config.max_retries {
        if attempt > 0 {
            std::thread::sleep(backoff);
            backoff = backoff.saturating_mul(2);
        }

        match ureq::post(&url)
            .header("Authorization", &format!("Bearer {}", config.api_key))
            .send_json(&body)
        {
            Ok(mut response) => {
                let parsed: serde_json::Value = response
                    .body_mut()
                    .read_json()
                    .map_err(|error| format!("failed to parse chat response: {error}"))?;
                return parse_chat_completion(&parsed);
            }
            Err(ureq::Error::StatusCode(status)) if status == 429 || status >= 500 => {
                last_error = format!("http status {status}");
            }
            Err(ureq::Error::StatusCode(status)) => {
                return Err(format!(
                    "chat API request rejected with http status {status} (not retryable); \
                     check OPENAI_API_KEY / model / base URL"
                ));
            }
            Err(other) => {
                last_error = format!("transport error: {other}");
            }
        }
    }

    Err(format!(
        "chat completion failed after {} attempt(s): {last_error}",
        config.max_retries + 1
    ))
}

// ─── Reader ──────────────────────────────────────────────────

/// System prompt for the QA reader.
pub const READER_SYSTEM_PROMPT: &str = "You are a helpful assistant that answers questions \
     about a user's long-term conversation history using only the provided excerpts.";

/// Build the reader user prompt: retrieved context + question.
///
/// The reader is explicitly allowed to abstain so that LongMemEval/BEAM
/// abstention questions can be answered correctly.
pub fn build_reader_prompt(question: &str, context: &str) -> String {
    format!(
        "You are given excerpts retrieved from a long conversation history, followed by a question.\n\
         Answer the question based only on the excerpts. Keep the answer short and factual.\n\
         If the excerpts do not contain the information needed to answer, reply exactly: I don't know.\n\n\
         Retrieved excerpts:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    )
}

/// One reader task: the question plus the assembled retrieval context the
/// harness already scored with containment metrics.
#[derive(Debug, Clone)]
pub struct ReaderInput {
    pub query_id: String,
    pub question: String,
    pub context: String,
    pub expected_answers: Vec<String>,
    pub category: String,
    /// Abstention/negative query: correct behavior is declining to answer.
    pub negative: bool,
}

/// Reader output with EXACT per-query API usage.
#[derive(Debug, Clone)]
pub struct ReaderAnswer {
    pub query_id: String,
    pub answer: String,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
}

/// Generate answers for all inputs, concurrency-limited.
pub fn generate_answers(
    inputs: &[ReaderInput],
    config: &ChatClientConfig,
    concurrency: usize,
) -> Result<Vec<ReaderAnswer>, String> {
    run_concurrently(inputs, concurrency, |input| {
        let completion = chat_completion(
            config,
            READER_SYSTEM_PROMPT,
            &build_reader_prompt(&input.question, &input.context),
        )
        .map_err(|error| format!("reader call failed for query `{}`: {error}", input.query_id))?;

        Ok(ReaderAnswer {
            query_id: input.query_id.clone(),
            answer: completion.content,
            prompt_tokens: completion.usage.prompt_tokens,
            completion_tokens: completion.usage.completion_tokens,
        })
    })
}

// ─── Judge ───────────────────────────────────────────────────

/// Which judge prompt family scores the generated answers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JudgeProtocol {
    /// Official LongMemEval GPT-4o judge prompts (question-type-aware).
    LongMemEvalOfficial,
    /// BEAM answer evaluation: LLM judge citing the gold answer.
    BeamReaderJudged,
    /// Generic gold-answer-cited yes/no judge for other adapters.
    GenericReaderJudged,
}

impl JudgeProtocol {
    pub fn label(self) -> &'static str {
        match self {
            Self::LongMemEvalOfficial => "longmemeval-official",
            Self::BeamReaderJudged => "beam-reader-judged",
            Self::GenericReaderJudged => "generic-reader-judged",
        }
    }

    /// Map an external `--format-name` to its judge protocol.
    pub fn for_format_name(format_name: &str) -> Self {
        match format_name.to_ascii_lowercase().as_str() {
            "longmemeval" | "lme" => Self::LongMemEvalOfficial,
            "beam" => Self::BeamReaderJudged,
            _ => Self::GenericReaderJudged,
        }
    }
}

/// System prompt for the yes/no correctness judge.
pub const JUDGE_SYSTEM_PROMPT: &str = "You are a strict evaluator. Answer yes or no only.";

const LME_ANSCHECK_DEFAULT: &str = "I will give you a question, a correct answer, and a response \
     from a model. Please answer yes if the response contains the correct answer. Otherwise, \
     answer no. If the response is equivalent to the correct answer or contains all the \
     intermediate steps to get the correct answer, you should also answer yes. If the response \
     only contains a subset of the information required by the answer, answer no.";

const LME_ANSCHECK_TEMPORAL_SUFFIX: &str = " In addition, do not penalize off-by-one errors for \
     the number of days. If the question asks for the number of days/weeks/months, etc., and the \
     model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's \
     response is still correct.";

const LME_ANSCHECK_KNOWLEDGE_UPDATE: &str = "I will give you a question, a correct answer, and a \
     response from a model. Please answer yes if the response contains the correct answer. \
     Otherwise, answer no. If the response contains some previous information along with an \
     updated answer, the response should be considered as correct as long as the updated answer \
     is the required answer.";

const LME_ANSCHECK_PREFERENCE: &str = "I will give you a question, a rubric for desired \
     personalized response, and a response from a model. Please answer yes if the response \
     satisfies the desired response. Otherwise, answer no. The model does not need to reflect \
     all the points in the rubric. The response is correct as long as it recalls and utilizes \
     the user's personal information correctly.";

const LME_ANSCHECK_ABSTENTION: &str = "I will give you an unanswerable question, an incorrect \
     answer, and a response from a model. Please answer yes if the model correctly identifies \
     the question as unanswerable. The model could say that the information is incomplete, or \
     the question is unanswerable, or it does not know the answer. Otherwise, answer no.";

/// Whether a LongMemEval query is an abstention case.
///
/// The official dataset marks these with a `_abs` suffix on the question id;
/// the adapter additionally flags them as negative queries.
pub fn is_longmemeval_abstention(query_id: &str, negative: bool) -> bool {
    negative || query_id.contains("_abs")
}

/// Build the official LongMemEval judge prompt for one answer.
///
/// Follows the prompt structure of `get_anscheck_prompt` in the LongMemEval
/// repository (question-type-aware templates, abstention variant for `_abs`
/// question ids). Unknown task types fall back to the default template.
pub fn longmemeval_judge_prompt(
    question_type: &str,
    question: &str,
    gold_answer: &str,
    model_answer: &str,
    is_abstention: bool,
) -> String {
    if is_abstention {
        return format!(
            "{LME_ANSCHECK_ABSTENTION}\n\nQuestion: {question}\n\nIncorrect Answer: {gold_answer}\n\n\
             Model Response: {model_answer}\n\nDoes the model correctly identify the question as \
             unanswerable? Answer yes or no only."
        );
    }

    match question_type {
        "temporal-reasoning" => format!(
            "{LME_ANSCHECK_DEFAULT}{LME_ANSCHECK_TEMPORAL_SUFFIX}\n\nQuestion: {question}\n\n\
             Correct Answer: {gold_answer}\n\nModel Response: {model_answer}\n\n\
             Is the model response correct? Answer yes or no only."
        ),
        "knowledge-update" => format!(
            "{LME_ANSCHECK_KNOWLEDGE_UPDATE}\n\nQuestion: {question}\n\nCorrect Answer: \
             {gold_answer}\n\nModel Response: {model_answer}\n\nIs the model response correct? \
             Answer yes or no only."
        ),
        "single-session-preference" => format!(
            "{LME_ANSCHECK_PREFERENCE}\n\nQuestion: {question}\n\nRubric: {gold_answer}\n\n\
             Model Response: {model_answer}\n\nIs the model response correct? Answer yes or no only."
        ),
        // single-session-user, single-session-assistant, multi-session, and
        // any unknown task type use the default template.
        _ => format!(
            "{LME_ANSCHECK_DEFAULT}\n\nQuestion: {question}\n\nCorrect Answer: {gold_answer}\n\n\
             Model Response: {model_answer}\n\nIs the model response correct? Answer yes or no only."
        ),
    }
}

/// Build the BEAM (or generic) judge prompt: yes/no correctness judgment
/// citing the gold answer, with an abstention variant for BEAM's abstention
/// dimension.
pub fn beam_judge_prompt(
    question: &str,
    gold_answer: &str,
    model_answer: &str,
    is_abstention: bool,
) -> String {
    if is_abstention {
        return format!(
            "You are evaluating a memory-augmented assistant on a probing question about a long \
             conversation. The question is unanswerable from the conversation, so the correct \
             behavior is to abstain (decline to answer or state the information is unavailable).\n\n\
             Question: {question}\n\nReference note: {gold_answer}\n\nModel Answer: {model_answer}\n\n\
             Does the model correctly abstain instead of fabricating an answer? Answer yes or no only."
        );
    }

    format!(
        "You are evaluating a memory-augmented assistant's answer to a probing question about a \
         long conversation. Compare the model answer against the gold answer.\n\n\
         Question: {question}\n\nGold Answer: {gold_answer}\n\nModel Answer: {model_answer}\n\n\
         Answer yes if the model answer is factually consistent with the gold answer and contains \
         the key information it requires; answer no otherwise. Partial answers that omit required \
         information count as no. Answer yes or no only."
    )
}

/// Parse a yes/no verdict from a judge response. Returns `None` when the
/// response contains neither.
pub fn parse_judge_verdict(response: &str) -> Option<bool> {
    let normalized = response.trim().to_ascii_lowercase();
    let first_word = normalized
        .split(|c: char| !c.is_ascii_alphabetic())
        .find(|token| !token.is_empty())?;
    match first_word {
        "yes" => Some(true),
        "no" => Some(false),
        _ => None,
    }
}

/// One judged answer with the judge call's exact usage.
#[derive(Debug, Clone)]
pub struct JudgedAnswer {
    pub query_id: String,
    pub category: String,
    pub abstention: bool,
    pub correct: bool,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
}

/// Judge all generated answers, concurrency-limited.
///
/// `answers` must be positionally aligned with `inputs` (as produced by
/// [`generate_answers`]).
pub fn judge_answers(
    protocol: JudgeProtocol,
    inputs: &[ReaderInput],
    answers: &[ReaderAnswer],
    config: &ChatClientConfig,
    concurrency: usize,
) -> Result<Vec<JudgedAnswer>, String> {
    if inputs.len() != answers.len() {
        return Err(format!(
            "judge alignment error: {} inputs vs {} answers",
            inputs.len(),
            answers.len()
        ));
    }

    let pairs: Vec<(&ReaderInput, &ReaderAnswer)> = inputs.iter().zip(answers.iter()).collect();

    run_concurrently(&pairs, concurrency, |(input, answer)| {
        debug_assert_eq!(input.query_id, answer.query_id);
        let gold_answer = input.expected_answers.join("; ");
        let abstention = match protocol {
            JudgeProtocol::LongMemEvalOfficial => {
                is_longmemeval_abstention(&input.query_id, input.negative)
            }
            JudgeProtocol::BeamReaderJudged | JudgeProtocol::GenericReaderJudged => input.negative,
        };
        let prompt = match protocol {
            JudgeProtocol::LongMemEvalOfficial => longmemeval_judge_prompt(
                &input.category,
                &input.question,
                &gold_answer,
                &answer.answer,
                abstention,
            ),
            JudgeProtocol::BeamReaderJudged | JudgeProtocol::GenericReaderJudged => {
                beam_judge_prompt(&input.question, &gold_answer, &answer.answer, abstention)
            }
        };

        let completion =
            chat_completion(config, JUDGE_SYSTEM_PROMPT, &prompt).map_err(|error| {
                format!("judge call failed for query `{}`: {error}", input.query_id)
            })?;
        let correct = parse_judge_verdict(&completion.content).ok_or_else(|| {
            format!(
                "judge returned neither yes nor no for query `{}`: {}",
                input.query_id,
                completion.content.trim()
            )
        })?;

        Ok(JudgedAnswer {
            query_id: input.query_id.clone(),
            category: input.category.clone(),
            abstention,
            correct,
            prompt_tokens: completion.usage.prompt_tokens,
            completion_tokens: completion.usage.completion_tokens,
        })
    })
}

// ─── Concurrency ─────────────────────────────────────────────

/// Run `task` over all items with at most `concurrency` worker threads,
/// preserving input order. Fails with the first error encountered.
fn run_concurrently<T, R, F>(items: &[T], concurrency: usize, task: F) -> Result<Vec<R>, String>
where
    T: Sync,
    R: Send,
    F: Fn(&T) -> Result<R, String> + Sync,
{
    let concurrency = concurrency.max(1);
    let next_index = std::sync::atomic::AtomicUsize::new(0);
    let results: Mutex<Vec<Option<R>>> = Mutex::new((0..items.len()).map(|_| None).collect());
    let first_error: Mutex<Option<String>> = Mutex::new(None);

    std::thread::scope(|scope| {
        for _ in 0..concurrency.min(items.len().max(1)) {
            scope.spawn(|| {
                loop {
                    if first_error.lock().expect("reader error lock").is_some() {
                        return;
                    }
                    let index = next_index.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    if index >= items.len() {
                        return;
                    }
                    match task(&items[index]) {
                        Ok(result) => {
                            results.lock().expect("reader results lock")[index] = Some(result);
                        }
                        Err(error) => {
                            let mut slot = first_error.lock().expect("reader error lock");
                            if slot.is_none() {
                                *slot = Some(error);
                            }
                            return;
                        }
                    }
                }
            });
        }
    });

    if let Some(error) = first_error.into_inner().expect("reader error lock") {
        return Err(error);
    }

    results
        .into_inner()
        .expect("reader results lock")
        .into_iter()
        .map(|slot| slot.ok_or_else(|| "reader worker dropped a result".to_string()))
        .collect()
}

// ─── Report ──────────────────────────────────────────────────

/// Per-category reader-judged accuracy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReaderCategoryAccuracy {
    pub name: String,
    pub accuracy: f64,
    pub total: usize,
}

/// Published reader/judge results for one benchmark run.
///
/// `official_reader_accuracy` is the LLM-judged QA accuracy over generated
/// answers — a different measurement from the retrieval-only `containment`
/// metric. The two must never be conflated in published tables.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReaderJudgeReport {
    pub reader_model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub judge_model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub judge_protocol: Option<String>,
    pub reader_temperature: f64,
    /// LLM-judged QA accuracy; `None` when answers were generated but not judged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub official_reader_accuracy: Option<f64>,
    pub answered_queries: usize,
    pub judged_queries: usize,
    pub abstention_queries: usize,
    pub abstention_correct: usize,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub category_accuracy: Vec<ReaderCategoryAccuracy>,
    pub reader_prompt_tokens_total: usize,
    pub reader_completion_tokens_total: usize,
    pub reader_prompt_tokens_per_query_mean: f64,
    pub reader_prompt_tokens_per_query_p50: usize,
    pub reader_prompt_tokens_per_query_p95: usize,
    pub reader_completion_tokens_per_query_mean: f64,
    pub reader_completion_tokens_per_query_p50: usize,
    pub reader_completion_tokens_per_query_p95: usize,
    pub judge_prompt_tokens_total: usize,
    pub judge_completion_tokens_total: usize,
}

fn usage_percentile(sorted: &[usize], percentile: usize) -> usize {
    if sorted.is_empty() {
        return 0;
    }
    let n = sorted.len();
    sorted[(n * percentile / 100).min(n - 1)]
}

fn usage_mean(values: &[usize]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<usize>() as f64 / values.len() as f64
    }
}

/// Aggregate reader answers and (optional) judge verdicts into the published
/// report, including exact per-query token distributions.
pub fn summarize(
    reader_model: &str,
    judge_model: Option<&str>,
    protocol: Option<JudgeProtocol>,
    reader_temperature: f64,
    answers: &[ReaderAnswer],
    judged: Option<&[JudgedAnswer]>,
) -> ReaderJudgeReport {
    let mut prompt_tokens: Vec<usize> = answers.iter().map(|a| a.prompt_tokens).collect();
    let mut completion_tokens: Vec<usize> = answers.iter().map(|a| a.completion_tokens).collect();
    let reader_prompt_tokens_total: usize = prompt_tokens.iter().sum();
    let reader_completion_tokens_total: usize = completion_tokens.iter().sum();
    let reader_prompt_tokens_per_query_mean = usage_mean(&prompt_tokens);
    let reader_completion_tokens_per_query_mean = usage_mean(&completion_tokens);
    prompt_tokens.sort_unstable();
    completion_tokens.sort_unstable();

    let (
        official_reader_accuracy,
        judged_queries,
        abstention_queries,
        abstention_correct,
        category_accuracy,
        judge_prompt_tokens_total,
        judge_completion_tokens_total,
    ) = match judged {
        Some(judged) if !judged.is_empty() => {
            let correct = judged.iter().filter(|entry| entry.correct).count();
            let abstention: Vec<&JudgedAnswer> =
                judged.iter().filter(|entry| entry.abstention).collect();
            let mut per_category: std::collections::BTreeMap<String, (usize, usize)> =
                std::collections::BTreeMap::new();
            for entry in judged {
                let slot = per_category.entry(entry.category.clone()).or_default();
                slot.1 += 1;
                if entry.correct {
                    slot.0 += 1;
                }
            }
            (
                Some(correct as f64 / judged.len() as f64),
                judged.len(),
                abstention.len(),
                abstention.iter().filter(|entry| entry.correct).count(),
                per_category
                    .into_iter()
                    .map(|(name, (correct, total))| ReaderCategoryAccuracy {
                        name,
                        accuracy: correct as f64 / total as f64,
                        total,
                    })
                    .collect(),
                judged.iter().map(|entry| entry.prompt_tokens).sum(),
                judged.iter().map(|entry| entry.completion_tokens).sum(),
            )
        }
        _ => (None, 0, 0, 0, Vec::new(), 0, 0),
    };

    ReaderJudgeReport {
        reader_model: reader_model.to_string(),
        judge_model: judge_model.map(str::to_string),
        judge_protocol: protocol.map(|protocol| protocol.label().to_string()),
        reader_temperature,
        official_reader_accuracy,
        answered_queries: answers.len(),
        judged_queries,
        abstention_queries,
        abstention_correct,
        category_accuracy,
        reader_prompt_tokens_total,
        reader_completion_tokens_total,
        reader_prompt_tokens_per_query_mean,
        reader_prompt_tokens_per_query_p50: usage_percentile(&prompt_tokens, 50),
        reader_prompt_tokens_per_query_p95: usage_percentile(&prompt_tokens, 95),
        reader_completion_tokens_per_query_mean,
        reader_completion_tokens_per_query_p50: usage_percentile(&completion_tokens, 50),
        reader_completion_tokens_per_query_p95: usage_percentile(&completion_tokens, 95),
        judge_prompt_tokens_total,
        judge_completion_tokens_total,
    }
}

/// Resolve the chat endpoint base URL: CLI value, else `OPENAI_BASE_URL`,
/// else the public OpenAI endpoint.
pub fn resolve_base_url(cli_value: Option<&str>) -> String {
    if let Some(value) = cli_value.map(str::trim).filter(|value| !value.is_empty()) {
        return value.to_string();
    }
    if let Ok(value) = std::env::var("OPENAI_BASE_URL") {
        let value = value.trim();
        if !value.is_empty() {
            return value.to_string();
        }
    }
    DEFAULT_OPENAI_BASE_URL.to_string()
}

#[cfg(test)]
mod tests {
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::thread;

    use super::*;

    fn chat_response_body(content: &str, prompt_tokens: usize, completion_tokens: usize) -> String {
        serde_json::json!({
            "choices": [{"message": {"role": "assistant", "content": content}}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        })
        .to_string()
    }

    /// Serve `responses` (status, body) pairs, one per connection.
    fn mock_chat_server(responses: Vec<(u16, String)>) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();

        thread::spawn(move || {
            for (status, body) in responses {
                let (mut stream, _) = listener.accept().unwrap();
                let mut buffer = vec![0u8; 65536];
                let _ = stream.read(&mut buffer).unwrap();
                let status_text = match status {
                    200 => "OK",
                    429 => "Too Many Requests",
                    500 => "Internal Server Error",
                    _ => "Error",
                };
                let response = format!(
                    "HTTP/1.1 {status} {status_text}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body,
                );
                stream.write_all(response.as_bytes()).unwrap();
                stream.flush().unwrap();
            }
        });

        format!("http://{addr}")
    }

    fn test_config(base_url: String) -> ChatClientConfig {
        ChatClientConfig {
            base_url,
            api_key: "test-key".to_string(),
            model: "gpt-4o".to_string(),
            temperature: 0.0,
            max_retries: 2,
            initial_backoff: Duration::from_millis(1),
        }
    }

    #[test]
    fn parse_chat_completion_extracts_content_and_exact_usage() {
        let response: serde_json::Value =
            serde_json::from_str(&chat_response_body("Lisbon", 321, 7)).unwrap();
        let completion = parse_chat_completion(&response).unwrap();
        assert_eq!(completion.content, "Lisbon");
        assert_eq!(completion.usage.prompt_tokens, 321);
        assert_eq!(completion.usage.completion_tokens, 7);
    }

    #[test]
    fn parse_chat_completion_requires_usage_field() {
        let response = serde_json::json!({
            "choices": [{"message": {"content": "Lisbon"}}],
        });
        let error = parse_chat_completion(&response).unwrap_err();
        assert!(error.contains("usage"), "unexpected error: {error}");
    }

    #[test]
    fn parse_chat_completion_surfaces_api_errors() {
        let response = serde_json::json!({"error": {"message": "invalid model"}});
        let error = parse_chat_completion(&response).unwrap_err();
        assert!(error.contains("invalid model"), "unexpected error: {error}");
    }

    #[test]
    fn reader_prompt_contains_context_question_and_abstention_instruction() {
        let prompt = build_reader_prompt("Where does Alice live?", "[s1] Alice: I moved to Lisbon");
        assert!(prompt.contains("Where does Alice live?"));
        assert!(prompt.contains("[s1] Alice: I moved to Lisbon"));
        assert!(prompt.contains("I don't know"));
    }

    #[test]
    fn longmemeval_judge_prompt_uses_official_default_template() {
        let prompt = longmemeval_judge_prompt("multi-session", "Q?", "gold", "resp", false);
        assert!(prompt.contains("Please answer yes if the response contains the correct answer"));
        assert!(prompt.contains("Question: Q?"));
        assert!(prompt.contains("Correct Answer: gold"));
        assert!(prompt.contains("Model Response: resp"));
        assert!(prompt.ends_with("Answer yes or no only."));
        assert!(!prompt.contains("off-by-one"));
    }

    #[test]
    fn longmemeval_judge_prompt_temporal_allows_off_by_one() {
        let prompt = longmemeval_judge_prompt("temporal-reasoning", "Q?", "18", "19", false);
        assert!(prompt.contains("do not penalize off-by-one errors"));
    }

    #[test]
    fn longmemeval_judge_prompt_knowledge_update_accepts_updated_answer() {
        let prompt = longmemeval_judge_prompt("knowledge-update", "Q?", "gold", "resp", false);
        assert!(prompt.contains("updated answer"));
    }

    #[test]
    fn longmemeval_judge_prompt_preference_uses_rubric() {
        let prompt =
            longmemeval_judge_prompt("single-session-preference", "Q?", "gold", "resp", false);
        assert!(prompt.contains("Rubric: gold"));
    }

    #[test]
    fn longmemeval_judge_prompt_abstention_variant() {
        let prompt = longmemeval_judge_prompt("multi-session", "Q?", "gold", "resp", true);
        assert!(prompt.contains("unanswerable"));
        assert!(prompt.contains("Incorrect Answer: gold"));
    }

    #[test]
    fn longmemeval_abstention_detected_from_abs_question_id() {
        assert!(is_longmemeval_abstention("lme-gpt4_2655b836_abs-12", false));
        assert!(is_longmemeval_abstention("lme-case-3", true));
        assert!(!is_longmemeval_abstention("lme-case-3", false));
    }

    #[test]
    fn beam_judge_prompt_cites_gold_answer() {
        let prompt = beam_judge_prompt("Q?", "gold answer", "model answer", false);
        assert!(prompt.contains("Gold Answer: gold answer"));
        assert!(prompt.contains("Model Answer: model answer"));
        assert!(prompt.ends_with("Answer yes or no only."));
    }

    #[test]
    fn beam_judge_prompt_abstention_variant() {
        let prompt = beam_judge_prompt("Q?", "gold", "resp", true);
        assert!(prompt.contains("correct behavior is to abstain"));
    }

    #[test]
    fn judge_verdict_parses_yes_no_and_rejects_garbage() {
        assert_eq!(parse_judge_verdict("Yes"), Some(true));
        assert_eq!(parse_judge_verdict(" yes."), Some(true));
        assert_eq!(parse_judge_verdict("No, the answer is wrong"), Some(false));
        assert_eq!(parse_judge_verdict("NO"), Some(false));
        assert_eq!(parse_judge_verdict("maybe"), None);
        assert_eq!(parse_judge_verdict(""), None);
    }

    #[test]
    fn judge_protocol_maps_format_names() {
        assert_eq!(
            JudgeProtocol::for_format_name("longmemeval"),
            JudgeProtocol::LongMemEvalOfficial
        );
        assert_eq!(
            JudgeProtocol::for_format_name("lme"),
            JudgeProtocol::LongMemEvalOfficial
        );
        assert_eq!(
            JudgeProtocol::for_format_name("beam"),
            JudgeProtocol::BeamReaderJudged
        );
        assert_eq!(
            JudgeProtocol::for_format_name("locomo"),
            JudgeProtocol::GenericReaderJudged
        );
    }

    #[test]
    fn chat_completion_sends_bearer_token_and_parses_usage() {
        let base_url = mock_chat_server(vec![(200, chat_response_body("Lisbon", 100, 5))]);
        let config = test_config(base_url);
        let completion = chat_completion(&config, "system", "user").unwrap();
        assert_eq!(completion.content, "Lisbon");
        assert_eq!(completion.usage.prompt_tokens, 100);
        assert_eq!(completion.usage.completion_tokens, 5);
    }

    #[test]
    fn chat_completion_retries_on_server_errors_with_backoff() {
        let base_url = mock_chat_server(vec![
            (500, "{}".to_string()),
            (429, "{}".to_string()),
            (200, chat_response_body("ok", 10, 2)),
        ]);
        let config = test_config(base_url);
        let completion = chat_completion(&config, "system", "user").unwrap();
        assert_eq!(completion.content, "ok");
    }

    #[test]
    fn chat_completion_fails_after_exhausting_retries() {
        let base_url = mock_chat_server(vec![
            (500, "{}".to_string()),
            (500, "{}".to_string()),
            (500, "{}".to_string()),
        ]);
        let config = test_config(base_url);
        let error = chat_completion(&config, "system", "user").unwrap_err();
        assert!(error.contains("after 3 attempt(s)"), "unexpected: {error}");
    }

    #[test]
    fn run_concurrently_preserves_input_order() {
        let items: Vec<usize> = (0..37).collect();
        let doubled = run_concurrently(&items, 5, |value| Ok(value * 2)).unwrap();
        assert_eq!(doubled, items.iter().map(|v| v * 2).collect::<Vec<_>>());
    }

    #[test]
    fn run_concurrently_propagates_errors() {
        let items: Vec<usize> = (0..10).collect();
        let error = run_concurrently(&items, 3, |value| {
            if *value == 4 {
                Err("boom".to_string())
            } else {
                Ok(*value)
            }
        })
        .unwrap_err();
        assert_eq!(error, "boom");
    }

    fn answer(id: &str, prompt: usize, completion: usize) -> ReaderAnswer {
        ReaderAnswer {
            query_id: id.to_string(),
            answer: "answer".to_string(),
            prompt_tokens: prompt,
            completion_tokens: completion,
        }
    }

    fn judged(id: &str, category: &str, abstention: bool, correct: bool) -> JudgedAnswer {
        JudgedAnswer {
            query_id: id.to_string(),
            category: category.to_string(),
            abstention,
            correct,
            prompt_tokens: 50,
            completion_tokens: 1,
        }
    }

    #[test]
    fn summarize_aggregates_exact_reader_usage() {
        let answers = vec![
            answer("q1", 100, 10),
            answer("q2", 300, 30),
            answer("q3", 200, 20),
        ];
        let report = summarize("gpt-4o", None, None, 0.0, &answers, None);

        assert_eq!(report.answered_queries, 3);
        assert_eq!(report.reader_prompt_tokens_total, 600);
        assert_eq!(report.reader_completion_tokens_total, 60);
        assert!((report.reader_prompt_tokens_per_query_mean - 200.0).abs() < 1e-9);
        assert!((report.reader_completion_tokens_per_query_mean - 20.0).abs() < 1e-9);
        assert_eq!(report.reader_prompt_tokens_per_query_p50, 200);
        assert_eq!(report.reader_prompt_tokens_per_query_p95, 300);
        assert_eq!(report.official_reader_accuracy, None);
        assert_eq!(report.judged_queries, 0);
    }

    #[test]
    fn summarize_computes_judged_accuracy_with_abstention_breakdown() {
        let answers = vec![
            answer("q1", 10, 1),
            answer("q2", 10, 1),
            answer("q3", 10, 1),
            answer("q4", 10, 1),
        ];
        let verdicts = vec![
            judged("q1", "multi-session", false, true),
            judged("q2", "multi-session", false, false),
            judged("q3", "abstention", true, true),
            judged("q4", "abstention", true, false),
        ];
        let report = summarize(
            "gpt-4o",
            Some("gpt-4o"),
            Some(JudgeProtocol::LongMemEvalOfficial),
            0.0,
            &answers,
            Some(&verdicts),
        );

        assert_eq!(report.official_reader_accuracy, Some(0.5));
        assert_eq!(report.judged_queries, 4);
        assert_eq!(report.abstention_queries, 2);
        assert_eq!(report.abstention_correct, 1);
        assert_eq!(report.judge_prompt_tokens_total, 200);
        assert_eq!(report.judge_completion_tokens_total, 4);
        assert_eq!(
            report.judge_protocol.as_deref(),
            Some("longmemeval-official")
        );
        let multi = report
            .category_accuracy
            .iter()
            .find(|category| category.name == "multi-session")
            .unwrap();
        assert!((multi.accuracy - 0.5).abs() < 1e-9);
        assert_eq!(multi.total, 2);
    }

    #[test]
    fn resolve_base_url_prefers_cli_value() {
        assert_eq!(
            resolve_base_url(Some("https://example.test/v1")),
            "https://example.test/v1"
        );
    }
}
