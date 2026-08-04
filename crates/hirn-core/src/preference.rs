//! Typed preference evidence inferred from first-person episodic statements.
//!
//! Preference evidence is stored in the existing durable metadata envelope so it
//! round-trips through every storage backend without a parallel preference table.
//! The public type keeps owner, polarity, target, qualifiers, and observed time
//! explicit; `functional_role = Preference` makes composition authority aware of
//! the evidence.

use std::collections::BTreeMap;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::HirnResult;
use crate::episodic::EpisodicRecord;
use crate::metadata::MetadataValue;
use crate::nlu::{DecisionSource, NluBudget};
use crate::timestamp::Timestamp;
use crate::types::{AgentId, MemoryType};

/// Metadata key containing the versioned typed preference envelope.
pub const PREFERENCE_EVIDENCE_METADATA_KEY: &str = "hirn.preference.v1";

/// Whether the owner favors or rejects the target.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PreferencePolarity {
    Positive,
    Negative,
}

impl PreferencePolarity {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Positive => "positive",
            Self::Negative => "negative",
        }
    }

    fn parse(value: &str) -> Option<Self> {
        match value {
            "positive" => Some(Self::Positive),
            "negative" => Some(Self::Negative),
            _ => None,
        }
    }
}

/// Durable, owner-scoped preference evidence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PreferenceEvidence {
    pub owner: AgentId,
    pub polarity: PreferencePolarity,
    pub target: String,
    pub qualifiers: Vec<String>,
    pub observed_at: Timestamp,
}

/// Collapse a preference target to a comparison key.
///
/// "Dark Mode", "dark-mode", and "the  dark_mode" name one preference, but
/// stored verbatim they are three, so a later contradiction never lines up with
/// what it contradicts and both survive into the answer.
///
/// Deliberately conservative: case, separators, surrounding punctuation, and a
/// leading article only. No stemming — "glasses" must not become "glasse", and
/// no synonym table, because collapsing two genuinely different targets is a
/// silent correctness failure while leaving them apart is merely a miss.
#[must_use]
pub fn normalize_preference_target(target: &str) -> String {
    let lowered = target.to_lowercase();
    let mut normalized = String::with_capacity(lowered.len());
    let mut pending_space = false;
    for character in lowered.chars() {
        if character.is_alphanumeric() {
            if pending_space && !normalized.is_empty() {
                normalized.push(' ');
            }
            pending_space = false;
            normalized.push(character);
        } else {
            pending_space = true;
        }
    }

    for article in ["the ", "a ", "an ", "my "] {
        if let Some(rest) = normalized.strip_prefix(article) {
            return rest.to_string();
        }
    }
    normalized
}

impl PreferenceEvidence {
    /// Comparison key for "is this the same preference?".
    #[must_use]
    pub fn normalized_target(&self) -> String {
        normalize_preference_target(&self.target)
    }

    /// Whether `self` and `other` describe the same owner's same target.
    #[must_use]
    pub fn concerns_same_preference_as(&self, other: &Self) -> bool {
        self.owner == other.owner && self.normalized_target() == other.normalized_target()
    }

    #[must_use]
    pub fn to_metadata_value(&self) -> MetadataValue {
        let mut map = BTreeMap::new();
        map.insert(
            "owner".into(),
            MetadataValue::String(self.owner.to_string()),
        );
        map.insert(
            "polarity".into(),
            MetadataValue::String(self.polarity.as_str().into()),
        );
        map.insert("target".into(), MetadataValue::String(self.target.clone()));
        map.insert(
            "qualifiers".into(),
            MetadataValue::List(
                self.qualifiers
                    .iter()
                    .cloned()
                    .map(MetadataValue::String)
                    .collect(),
            ),
        );
        map.insert(
            "observed_at_ms".into(),
            MetadataValue::Int(self.observed_at.timestamp_ms()),
        );
        MetadataValue::Map(map)
    }

    #[must_use]
    pub fn from_metadata_value(value: &MetadataValue) -> Option<Self> {
        let MetadataValue::Map(map) = value else {
            return None;
        };
        let MetadataValue::String(owner) = map.get("owner")? else {
            return None;
        };
        let MetadataValue::String(polarity) = map.get("polarity")? else {
            return None;
        };
        let MetadataValue::String(target) = map.get("target")? else {
            return None;
        };
        let MetadataValue::Int(observed_at_ms) = map.get("observed_at_ms")? else {
            return None;
        };
        let qualifiers = match map.get("qualifiers") {
            Some(MetadataValue::List(values)) => values
                .iter()
                .filter_map(|value| match value {
                    MetadataValue::String(value) => Some(value.clone()),
                    _ => None,
                })
                .collect(),
            None => Vec::new(),
            Some(_) => return None,
        };
        Some(Self {
            owner: AgentId::new(owner).ok()?,
            polarity: PreferencePolarity::parse(polarity)?,
            target: target.clone(),
            qualifiers,
            observed_at: Timestamp::from_millis((*observed_at_ms).max(0) as u64),
        })
    }
}

/// A preference a model read out of one message, before it is bound to a
/// record's owner and timestamp.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExtractedPreference {
    pub polarity: PreferencePolarity,
    /// What the preference is about, as a short noun phrase.
    pub target: String,
    /// Conditions the preference is scoped to ("when working at night").
    pub qualifiers: Vec<String>,
    /// Extraction confidence in `[0, 1]`.
    pub confidence: f32,
    pub source: DecisionSource,
}

/// A backend that reads a first-person preference out of a message.
///
/// The cue matcher in [`infer_preference_evidence`] is the deterministic floor
/// beneath this: it only fires on a fixed list of English first-person verb
/// phrases, so it misses "dark mode is the only way I can work at night",
/// "ich mag lieber Dunkelmodus", and every indirect or reported phrasing. A
/// model reads the same typed envelope out of any of them.
#[async_trait]
pub trait PreferenceExtractor: Send + Sync {
    /// Extract at most one preference from `text`.
    ///
    /// Returns `Ok(None)` when the text states no preference, or when the
    /// backend cannot produce one within budget — the caller falls back.
    async fn extract_preference(
        &self,
        text: &str,
        budget: &NluBudget,
    ) -> HirnResult<Option<ExtractedPreference>>;

    /// Stable model identifier.
    fn model_id(&self) -> &str;
}

/// Bind an extracted preference to a record's owner and observation time.
#[must_use]
pub fn bind_extracted_preference(
    record: &EpisodicRecord,
    extracted: ExtractedPreference,
) -> Option<PreferenceEvidence> {
    let target = extracted.target.trim();
    if target.len() < 2 {
        return None;
    }
    Some(PreferenceEvidence {
        owner: record.provenance.created_by,
        polarity: extracted.polarity,
        target: target.to_string(),
        qualifiers: extracted
            .qualifiers
            .into_iter()
            .map(|qualifier| qualifier.trim().to_string())
            .filter(|qualifier| !qualifier.is_empty())
            .collect(),
        observed_at: record.timestamp,
    })
}

/// Annotate a record using a model-backed extractor, falling back to the cue
/// matcher when it abstains or fails.
///
/// A caller-supplied typed envelope always wins over both.
pub async fn annotate_preference_with(
    record: &mut EpisodicRecord,
    extractor: Option<&dyn PreferenceExtractor>,
    budget: &NluBudget,
) {
    if record
        .metadata
        .contains_key(PREFERENCE_EVIDENCE_METADATA_KEY)
    {
        return;
    }

    if let Some(extractor) = extractor
        && let Some(text) = user_content(&record.content)
        && !text.trim().is_empty()
    {
        let text = text.to_string();
        match extractor.extract_preference(&text, budget).await {
            Ok(Some(extracted)) if extracted.confidence >= budget.min_confidence => {
                if let Some(evidence) = bind_extracted_preference(record, extracted) {
                    record.functional_role = MemoryType::Preference;
                    record.metadata.insert(
                        PREFERENCE_EVIDENCE_METADATA_KEY.into(),
                        evidence.to_metadata_value(),
                    );
                    return;
                }
            }
            // A confident "no preference here" is an answer: do not let the
            // cue matcher overrule it with a phrase match.
            Ok(None) => return,
            // Below the confidence gate, or the backend failed: fall through
            // to the cue matcher. `hirn-core` carries no logging dependency;
            // the provider records the abstention in
            // `hirn_nlu_abstentions_total`.
            Ok(Some(_)) | Err(_) => {}
        }
    }

    annotate_inferred_preference(record);
}

/// Infer one high-precision first-person preference from an episode.
///
/// **Deterministic floor.** Only explicit first-person preference verbs from a
/// fixed English list are accepted, which keeps false profile writes rare but
/// makes it blind to indirect phrasing, reported preference, and every other
/// language. [`annotate_preference_with`] prefers a model and falls back here.
#[must_use]
pub fn infer_preference_evidence(record: &EpisodicRecord) -> Option<PreferenceEvidence> {
    let content = user_content(&record.content)?;
    let lower = content.to_ascii_lowercase();
    const CUES: &[(&str, PreferencePolarity)] = &[
        ("i do not like ", PreferencePolarity::Negative),
        ("i don't like ", PreferencePolarity::Negative),
        ("i dislike ", PreferencePolarity::Negative),
        ("i hate ", PreferencePolarity::Negative),
        ("i prefer ", PreferencePolarity::Positive),
        ("i would prefer ", PreferencePolarity::Positive),
        ("i love ", PreferencePolarity::Positive),
        ("i enjoy ", PreferencePolarity::Positive),
        ("my favorite ", PreferencePolarity::Positive),
        ("my favourite ", PreferencePolarity::Positive),
        ("i am interested in ", PreferencePolarity::Positive),
        ("i'm interested in ", PreferencePolarity::Positive),
    ];

    let (start, polarity) = CUES
        .iter()
        .filter_map(|(cue, polarity)| {
            lower
                .find(cue)
                .filter(|position| {
                    !content[..*position]
                        .trim_end()
                        .ends_with(['"', '\'', '‘', '“'])
                })
                .map(|position| (position + cue.len(), *polarity))
        })
        .min_by_key(|(position, _)| *position)?;
    let raw = content.get(start..)?.trim();
    let end = raw.find(['.', '!', '?', '\n']).unwrap_or(raw.len());
    let phrase = raw[..end]
        .trim()
        .trim_matches(|character| matches!(character, '"' | '\'' | ',' | ';' | ':'));
    if phrase.len() < 2 {
        return None;
    }

    let (target, qualifiers) = split_qualifier(phrase);
    if target.len() < 2 {
        return None;
    }
    Some(PreferenceEvidence {
        owner: record.provenance.created_by,
        polarity,
        target: target.to_string(),
        qualifiers: qualifiers.into_iter().map(str::to_string).collect(),
        observed_at: record.timestamp,
    })
}

fn user_content(content: &str) -> Option<&str> {
    let trimmed = content.trim();
    let Some(close) = trimmed.find("] ") else {
        return Some(trimmed);
    };
    let message = &trimmed[close + 2..];
    let Some((speaker, content)) = message.split_once(':') else {
        return Some(trimmed);
    };
    speaker
        .trim()
        .eq_ignore_ascii_case("user")
        .then_some(content.trim())
}

/// One preference's current state, plus the observations it replaced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CurrentPreference {
    /// The most recent observation for this owner and normalized target.
    pub current: PreferenceEvidence,
    /// Earlier observations of the same preference, newest first.
    ///
    /// Kept rather than discarded: "I used to prefer X" is a real question, and
    /// a superseded observation is history, not noise. Only the *current* entry
    /// may be presented as what the user prefers now.
    pub superseded: Vec<PreferenceEvidence>,
}

impl CurrentPreference {
    /// Whether the owner reversed their position on this target.
    ///
    /// A flip is worth surfacing: answering "do you like X?" from a stale
    /// positive when the user has since said the opposite is a user-visible
    /// correctness failure, not a ranking miss.
    #[must_use]
    pub fn polarity_reversed(&self) -> bool {
        self.superseded
            .iter()
            .any(|earlier| earlier.polarity != self.current.polarity)
    }
}

/// Collapse observations into one current state per owner and target.
///
/// Without this, a user who says "I prefer aisle seats" and later "I prefer
/// window seats now" has both surfaced with no defined winner, and the answer
/// depends on retrieval order.
///
/// **Recency decides.** Later observation wins; ties keep the first seen so the
/// result is deterministic regardless of input order. Recency is the only
/// signal available here — confidence and specificity are not comparable across
/// extractions — and a stated change of mind is exactly what should win.
///
/// Output is ordered by normalized target so callers render deterministically.
#[must_use]
pub fn current_preferences(evidence: &[PreferenceEvidence]) -> Vec<CurrentPreference> {
    let mut grouped: BTreeMap<(String, String), Vec<PreferenceEvidence>> = BTreeMap::new();
    for entry in evidence {
        grouped
            .entry((entry.owner.to_string(), entry.normalized_target()))
            .or_default()
            .push(entry.clone());
    }

    grouped
        .into_values()
        .map(|mut observations| {
            // Newest first; `sort_by_key` is stable, so equal timestamps keep
            // input order and the winner is reproducible.
            observations.sort_by_key(|entry| std::cmp::Reverse(entry.observed_at));
            let current = observations.remove(0);
            CurrentPreference {
                current,
                superseded: observations,
            }
        })
        .collect()
}

/// Annotate a record unless the caller already supplied typed preference evidence.
pub fn annotate_inferred_preference(record: &mut EpisodicRecord) {
    if record
        .metadata
        .contains_key(PREFERENCE_EVIDENCE_METADATA_KEY)
    {
        return;
    }
    if let Some(evidence) = infer_preference_evidence(record) {
        record.functional_role = MemoryType::Preference;
        record.metadata.insert(
            PREFERENCE_EVIDENCE_METADATA_KEY.into(),
            evidence.to_metadata_value(),
        );
    }
}

fn split_qualifier(phrase: &str) -> (&str, Option<&str>) {
    const QUALIFIERS: &[&str] = &[" because ", " when ", " during ", " unless "];
    QUALIFIERS
        .iter()
        .filter_map(|separator| phrase.find(separator).map(|position| (position, separator)))
        .min_by_key(|(position, _)| *position)
        .map_or((phrase.trim(), None), |(position, separator)| {
            (
                phrase[..position].trim(),
                Some(phrase[position + separator.len()..].trim()),
            )
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::episodic::EpisodicRecord;
    use crate::types::EventType;

    fn owner(name: &str) -> AgentId {
        AgentId::new(name).unwrap()
    }

    fn evidence_at(
        who: &str,
        target: &str,
        polarity: PreferencePolarity,
        at_ms: u64,
    ) -> PreferenceEvidence {
        PreferenceEvidence {
            owner: owner(who),
            polarity,
            target: target.to_string(),
            qualifiers: Vec::new(),
            observed_at: Timestamp::from_millis(at_ms),
        }
    }

    #[test]
    fn normalization_collapses_separators_case_and_articles() {
        for (raw, expected) in [
            ("Dark Mode", "dark mode"),
            ("dark-mode", "dark mode"),
            ("  the   dark_mode!", "dark mode"),
            ("my aisle seats", "aisle seats"),
            ("An Espresso", "espresso"),
        ] {
            assert_eq!(normalize_preference_target(raw), expected, "raw: {raw}");
        }
    }

    /// Stemming and synonyms are deliberately out of scope: merging two
    /// genuinely different targets silently corrupts an answer, while leaving
    /// them apart is only a miss.
    #[test]
    fn normalization_does_not_stem_or_merge_distinct_targets() {
        assert_ne!(
            normalize_preference_target("glasses"),
            normalize_preference_target("glass")
        );
        assert_ne!(
            normalize_preference_target("window seats"),
            normalize_preference_target("aisle seats")
        );
    }

    #[test]
    fn a_later_statement_supersedes_an_earlier_one() {
        let observations = vec![
            evidence_at("u", "aisle seats", PreferencePolarity::Positive, 1_000),
            evidence_at("u", "Aisle-Seats", PreferencePolarity::Negative, 5_000),
        ];
        let current = current_preferences(&observations);

        assert_eq!(current.len(), 1, "different spellings are one preference");
        assert_eq!(current[0].current.polarity, PreferencePolarity::Negative);
        assert_eq!(current[0].superseded.len(), 1);
        assert!(
            current[0].polarity_reversed(),
            "a reversal must be detectable"
        );
    }

    #[test]
    fn distinct_targets_and_owners_stay_separate() {
        let observations = vec![
            evidence_at("u", "window seats", PreferencePolarity::Positive, 1_000),
            evidence_at("u", "aisle seats", PreferencePolarity::Negative, 2_000),
            evidence_at("other", "window seats", PreferencePolarity::Negative, 3_000),
        ];
        let current = current_preferences(&observations);
        assert_eq!(
            current.len(),
            3,
            "no cross-owner or cross-target collapsing"
        );
        assert!(current.iter().all(|entry| entry.superseded.is_empty()));
    }

    /// Order of retrieval must not decide the answer.
    #[test]
    fn supersession_is_independent_of_input_order() {
        let newer = evidence_at("u", "tea", PreferencePolarity::Negative, 9_000);
        let older = evidence_at("u", "tea", PreferencePolarity::Positive, 1_000);

        let forward = current_preferences(&[older.clone(), newer.clone()]);
        let reverse = current_preferences(&[newer, older]);
        assert_eq!(forward, reverse);
        assert_eq!(forward[0].current.polarity, PreferencePolarity::Negative);
    }

    #[test]
    fn equal_timestamps_resolve_deterministically() {
        let first = evidence_at("u", "coffee", PreferencePolarity::Positive, 7_000);
        let second = evidence_at("u", "coffee", PreferencePolarity::Negative, 7_000);
        let resolved = current_preferences(&[first.clone(), second]);
        assert_eq!(
            resolved[0].current, first,
            "a tie keeps the first observation seen, not an arbitrary one"
        );
    }

    fn record(content: &str) -> EpisodicRecord {
        EpisodicRecord::builder()
            .event_type(EventType::Observation)
            .content(content)
            .summary(content)
            .agent_id(AgentId::new("preference-owner").unwrap())
            .build()
            .unwrap()
    }

    #[test]
    fn explicit_preference_round_trips_through_metadata() {
        let mut record = record("[session] user: I prefer dark mode when working at night.");
        annotate_inferred_preference(&mut record);

        assert_eq!(record.functional_role, MemoryType::Preference);
        let evidence = PreferenceEvidence::from_metadata_value(
            &record.metadata[PREFERENCE_EVIDENCE_METADATA_KEY],
        )
        .unwrap();
        assert_eq!(evidence.polarity, PreferencePolarity::Positive);
        assert_eq!(evidence.target, "dark mode");
        assert_eq!(evidence.qualifiers, vec!["working at night"]);
        assert_eq!(evidence.owner, AgentId::new("preference-owner").unwrap());
    }

    #[test]
    fn rejects_generic_or_other_person_sentiment() {
        assert!(infer_preference_evidence(&record("The assistant likes dark mode.")).is_none());
        assert!(infer_preference_evidence(&record("Dark mode is available.")).is_none());
        assert!(
            infer_preference_evidence(&record(
                "[session] assistant: The user said, ‘I prefer dark mode.’"
            ))
            .is_none()
        );
        assert!(
            infer_preference_evidence(&record("[session] user: Alice said, ‘I prefer dark mode.’"))
                .is_none()
        );
        assert!(infer_preference_evidence(&record("I don't dislike dark mode.")).is_none());
    }

    #[test]
    fn unicode_prefix_preserves_extraction_offsets() {
        let evidence = infer_preference_evidence(&record(
            "[session] user: Café notes: I prefer dark mode when traveling.",
        ))
        .unwrap();

        assert_eq!(evidence.target, "dark mode");
        assert_eq!(evidence.qualifiers, vec!["traveling"]);
    }

    #[test]
    fn caller_supplied_evidence_is_not_overwritten() {
        let mut record = record("I prefer dark mode.");
        let supplied = PreferenceEvidence {
            owner: AgentId::new("preference-owner").unwrap(),
            polarity: PreferencePolarity::Negative,
            target: "screen glare".into(),
            qualifiers: vec!["at night".into()],
            observed_at: Timestamp::from_millis(42),
        };
        record.metadata.insert(
            PREFERENCE_EVIDENCE_METADATA_KEY.into(),
            supplied.to_metadata_value(),
        );

        annotate_inferred_preference(&mut record);

        let actual = PreferenceEvidence::from_metadata_value(
            &record.metadata[PREFERENCE_EVIDENCE_METADATA_KEY],
        )
        .unwrap();
        assert_eq!(actual, supplied);
    }

    // ── Model-backed extraction ──────────────────────────────────────────

    struct StubExtractor {
        result: Result<Option<ExtractedPreference>, ()>,
    }

    #[async_trait]
    impl PreferenceExtractor for StubExtractor {
        async fn extract_preference(
            &self,
            _text: &str,
            _budget: &NluBudget,
        ) -> HirnResult<Option<ExtractedPreference>> {
            match &self.result {
                Ok(extracted) => Ok(extracted.clone()),
                Err(()) => Err(crate::HirnError::provider("stub failure")),
            }
        }

        fn model_id(&self) -> &str {
            "stub-preference"
        }
    }

    fn extracted(target: &str, confidence: f32) -> ExtractedPreference {
        ExtractedPreference {
            polarity: PreferencePolarity::Positive,
            target: target.to_string(),
            qualifiers: vec!["working at night".to_string()],
            confidence,
            source: DecisionSource::Model,
        }
    }

    #[tokio::test]
    async fn model_reads_indirectly_phrased_preference() {
        // No first-person preference verb: the cue matcher finds nothing.
        let content = "[session] user: dark mode is the only way I can work at night.";
        let mut cue_only = record(content);
        annotate_inferred_preference(&mut cue_only);
        assert_ne!(cue_only.functional_role, MemoryType::Preference);

        let mut record = record(content);
        let extractor = StubExtractor {
            result: Ok(Some(extracted("dark mode", 0.9))),
        };
        annotate_preference_with(&mut record, Some(&extractor), &NluBudget::default()).await;

        assert_eq!(record.functional_role, MemoryType::Preference);
        let evidence = PreferenceEvidence::from_metadata_value(
            &record.metadata[PREFERENCE_EVIDENCE_METADATA_KEY],
        )
        .unwrap();
        assert_eq!(evidence.target, "dark mode");
        assert_eq!(evidence.qualifiers, vec!["working at night"]);
        assert_eq!(evidence.owner, AgentId::new("preference-owner").unwrap());
        // The envelope stores whole milliseconds, so compare at that precision
        // rather than against the record's sub-millisecond timestamp.
        assert_eq!(
            evidence.observed_at.timestamp_ms(),
            record.timestamp.timestamp_ms()
        );
    }

    #[tokio::test]
    async fn confident_no_preference_is_not_overruled_by_a_cue_match() {
        // The cue matcher would fire on "i prefer"; the model read the whole
        // sentence and said the speaker is describing someone else's taste.
        let mut record = record("[session] user: my manager said they prefer dark mode.");
        let extractor = StubExtractor { result: Ok(None) };
        annotate_preference_with(&mut record, Some(&extractor), &NluBudget::default()).await;
        assert_ne!(record.functional_role, MemoryType::Preference);
        assert!(
            !record
                .metadata
                .contains_key(PREFERENCE_EVIDENCE_METADATA_KEY)
        );
    }

    #[tokio::test]
    async fn low_confidence_extraction_falls_back_to_the_cue_matcher() {
        let mut record = record("[session] user: I prefer dark mode when working at night.");
        let extractor = StubExtractor {
            result: Ok(Some(extracted("something else entirely", 0.1))),
        };
        annotate_preference_with(&mut record, Some(&extractor), &NluBudget::default()).await;

        assert_eq!(record.functional_role, MemoryType::Preference);
        let evidence = PreferenceEvidence::from_metadata_value(
            &record.metadata[PREFERENCE_EVIDENCE_METADATA_KEY],
        )
        .unwrap();
        assert_eq!(evidence.target, "dark mode", "cue fallback decided");
    }

    #[tokio::test]
    async fn extractor_failure_falls_back_to_the_cue_matcher() {
        let mut record = record("[session] user: I prefer dark mode.");
        let extractor = StubExtractor { result: Err(()) };
        annotate_preference_with(&mut record, Some(&extractor), &NluBudget::default()).await;
        assert_eq!(record.functional_role, MemoryType::Preference);
    }

    #[tokio::test]
    async fn no_extractor_uses_the_cue_matcher() {
        let mut record = record("[session] user: I prefer dark mode.");
        annotate_preference_with(&mut record, None, &NluBudget::default()).await;
        assert_eq!(record.functional_role, MemoryType::Preference);
    }

    #[tokio::test]
    async fn caller_supplied_evidence_wins_over_both_paths() {
        let mut record = record("[session] user: I prefer dark mode.");
        let supplied = PreferenceEvidence {
            owner: AgentId::new("preference-owner").unwrap(),
            polarity: PreferencePolarity::Negative,
            target: "caller supplied".to_string(),
            qualifiers: vec![],
            observed_at: record.timestamp,
        };
        record.metadata.insert(
            PREFERENCE_EVIDENCE_METADATA_KEY.into(),
            supplied.to_metadata_value(),
        );

        let extractor = StubExtractor {
            result: Ok(Some(extracted("model said this", 0.99))),
        };
        annotate_preference_with(&mut record, Some(&extractor), &NluBudget::default()).await;

        let evidence = PreferenceEvidence::from_metadata_value(
            &record.metadata[PREFERENCE_EVIDENCE_METADATA_KEY],
        )
        .unwrap();
        assert_eq!(evidence.target, "caller supplied");
    }

    #[test]
    fn binding_rejects_a_degenerate_target() {
        let record = record("[session] user: whatever.");
        assert!(bind_extracted_preference(&record, extracted("x", 0.9)).is_none());
        // Blank qualifiers are dropped rather than stored as empty strings.
        let bound = bind_extracted_preference(
            &record,
            ExtractedPreference {
                qualifiers: vec!["  ".to_string(), "at night".to_string()],
                ..extracted("dark mode", 0.9)
            },
        )
        .unwrap();
        assert_eq!(bound.qualifiers, vec!["at night"]);
    }
}
