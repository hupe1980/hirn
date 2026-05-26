pub(crate) const RERANKER_INVALID_SCORES_TOTAL: &str = "hirn_reranker_invalid_scores_total";

pub(crate) fn record_invalid_reranker_score(provider: &str, reason: &str) {
    metrics::counter!(
        RERANKER_INVALID_SCORES_TOTAL,
        "provider" => provider.to_owned(),
        "reason" => reason.to_owned()
    )
    .increment(1);
}
