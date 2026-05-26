use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

use serde::{Deserialize, Serialize};

use hirn_core::error::HirnResult;
use hirn_core::id::MemoryId;
use hirn_core::types::AgentId;
use hirn_core::{DerivedArtifactKind, HydrationMode, ModalityProfile, ResourceId};

use crate::db::HirnDB;
use crate::ql::results::ScoredMemory;
use crate::recall::{RecallResult, ResourceEvidenceSummary};

const PREFERRED_PREVIEW_ARTIFACTS: [DerivedArtifactKind; 7] = [
    DerivedArtifactKind::Preview,
    DerivedArtifactKind::Transcript,
    DerivedArtifactKind::Caption,
    DerivedArtifactKind::OcrText,
    DerivedArtifactKind::SyntaxSummary,
    DerivedArtifactKind::SchemaSummary,
    DerivedArtifactKind::Thumbnail,
];

const RESOURCE_PREVIEW_RERANK_WEIGHT: f32 = 0.08;
const MAX_ATTRIBUTION_TERMS: usize = 6;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ResourcePreviewPackage {
    pub(crate) resource_id: ResourceId,
    pub(crate) role: hirn_core::EvidenceRole,
    pub(crate) display_name: Option<String>,
    pub(crate) modality: Option<ModalityProfile>,
    pub(crate) artifact_kind: DerivedArtifactKind,
    pub(crate) artifact_modality: ModalityProfile,
    pub(crate) text_content: String,
    pub(crate) truncated: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceScoreAttribution {
    pub resource_id: ResourceId,
    pub role: hirn_core::EvidenceRole,
    pub display_name: Option<String>,
    pub modality: Option<ModalityProfile>,
    pub artifact_kind: DerivedArtifactKind,
    pub artifact_modality: ModalityProfile,
    pub matched_terms: Vec<String>,
    pub match_score: f32,
    pub score_boost: f32,
}

#[derive(Debug, Clone)]
struct CachedResourcePreview {
    artifact_kind: DerivedArtifactKind,
    artifact_modality: ModalityProfile,
    text_content: String,
    truncated: bool,
}

#[derive(Debug, Default)]
pub(crate) struct PreviewPackageCache {
    cached_previews: BTreeMap<ResourceId, Option<CachedResourcePreview>>,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum PreviewPackageSurface {
    Recall,
    Think,
}

impl PreviewPackageSurface {
    const fn as_label(self) -> &'static str {
        match self {
            Self::Recall => "recall",
            Self::Think => "think",
        }
    }
}

pub(crate) async fn hydrate_resource_preview_packages_for_scored_records(
    db: &HirnDB,
    actor_id: &AgentId,
    scored: &[ScoredMemory],
    max_resource_previews_per_record: usize,
    max_resource_preview_chars: usize,
) -> HirnResult<BTreeMap<MemoryId, Vec<ResourcePreviewPackage>>> {
    if max_resource_previews_per_record == 0 || max_resource_preview_chars == 0 {
        return Ok(BTreeMap::new());
    }

    let mut preview_cache = PreviewPackageCache::default();
    let mut packaged = BTreeMap::new();

    for scored_record in scored {
        let preview_packages = package_resource_preview_packages_for_evidence(
            db,
            actor_id,
            &scored_record.resource_evidence,
            &scored_record.resource_preview_packages,
            max_resource_previews_per_record,
            max_resource_preview_chars,
            &mut preview_cache,
            PreviewPackageSurface::Recall,
        )
        .await;
        if !preview_packages.is_empty() {
            packaged.insert(scored_record.record.id(), preview_packages);
        }
    }

    Ok(packaged)
}

pub(crate) async fn apply_resource_preview_rerank(
    db: &HirnDB,
    actor_id: &AgentId,
    query_text: &str,
    results: &mut [RecallResult],
    max_resource_previews_per_result: usize,
    max_resource_preview_chars: usize,
) -> HirnResult<()> {
    if results.is_empty()
        || max_resource_previews_per_result == 0
        || max_resource_preview_chars == 0
    {
        return Ok(());
    }

    let query_terms = normalized_terms(query_text);
    if query_terms.is_empty() {
        return Ok(());
    }

    let mut preview_cache = PreviewPackageCache::default();
    for result in results.iter_mut() {
        let preview_packages = package_resource_preview_packages(
            db,
            actor_id,
            &result.resource_evidence,
            max_resource_previews_per_result,
            max_resource_preview_chars,
            &mut preview_cache,
        )
        .await;

        result
            .resource_preview_packages
            .clone_from(&preview_packages);

        if let Some(attribution) = best_resource_score_attribution(&query_terms, &preview_packages)
        {
            result.composite_score =
                (result.composite_score + attribution.score_boost).clamp(0.0, 1.0);
            result.resource_score_attribution = vec![attribution];
        } else {
            result.resource_score_attribution.clear();
        }
    }

    results.sort_by(|left, right| {
        right
            .composite_score
            .total_cmp(&left.composite_score)
            .then_with(|| right.similarity.total_cmp(&left.similarity))
    });

    Ok(())
}

pub(crate) async fn apply_resource_preview_rerank_to_scored_records(
    db: &HirnDB,
    actor_id: &AgentId,
    query_text: &str,
    results: &mut [ScoredMemory],
    max_resource_previews_per_result: usize,
    max_resource_preview_chars: usize,
) -> HirnResult<()> {
    if results.is_empty()
        || max_resource_previews_per_result == 0
        || max_resource_preview_chars == 0
    {
        return Ok(());
    }

    let query_terms = normalized_terms(query_text);
    if query_terms.is_empty() {
        return Ok(());
    }

    let mut preview_cache = PreviewPackageCache::default();
    for result in results.iter_mut() {
        let preview_packages = package_resource_preview_packages_for_evidence(
            db,
            actor_id,
            &result.resource_evidence,
            &result.resource_preview_packages,
            max_resource_previews_per_result,
            max_resource_preview_chars,
            &mut preview_cache,
            PreviewPackageSurface::Recall,
        )
        .await;

        result
            .resource_preview_packages
            .clone_from(&preview_packages);

        if let Some(attribution) = best_resource_score_attribution(&query_terms, &preview_packages)
        {
            result.score = (result.score + attribution.score_boost).clamp(0.0, 1.0);
            result.resource_score_attribution = vec![attribution];
        } else {
            result.resource_score_attribution.clear();
        }
    }

    results.sort_by(|left, right| {
        right.score.total_cmp(&left.score).then_with(|| {
            right
                .score_breakdown
                .similarity
                .total_cmp(&left.score_breakdown.similarity)
        })
    });

    Ok(())
}

pub(crate) async fn package_resource_preview_packages_for_evidence(
    db: &HirnDB,
    actor_id: &AgentId,
    resource_evidence: &[ResourceEvidenceSummary],
    seeded_packages: &[ResourcePreviewPackage],
    max_resource_previews: usize,
    max_resource_preview_chars: usize,
    preview_cache: &mut PreviewPackageCache,
    surface: PreviewPackageSurface,
) -> Vec<ResourcePreviewPackage> {
    let started = Instant::now();

    if max_resource_previews == 0 || max_resource_preview_chars == 0 {
        return Vec::new();
    }

    if let Some(reused) = reuse_seeded_preview_packages(
        seeded_packages,
        max_resource_previews,
        max_resource_preview_chars,
    ) {
        if !reused.is_empty() {
            record_preview_package_resolution(surface, "seeded_reuse", started.elapsed());
        }
        return reused;
    }

    let has_previewable_evidence = resource_evidence
        .iter()
        .any(|summary| summary.has_preview && summary.can_hydrate_preview);

    let packaged = package_resource_preview_packages(
        db,
        actor_id,
        resource_evidence,
        max_resource_previews,
        max_resource_preview_chars,
        preview_cache,
    )
    .await;

    if has_previewable_evidence {
        record_preview_package_resolution(surface, "hydrated_refetch", started.elapsed());
    }

    packaged
}

pub(crate) fn resource_preview_packages_to_json(
    packages: &[ResourcePreviewPackage],
) -> serde_json::Value {
    serde_json::Value::Array(
        packages
            .iter()
            .map(|package| {
                serde_json::json!({
                    "resource_id": package.resource_id.to_string(),
                    "role": package.role.as_str(),
                    "display_name": package.display_name,
                    "modality": package.modality.map(|modality| modality.as_str()),
                    "artifact_kind": package.artifact_kind.as_str(),
                    "artifact_modality": package.artifact_modality.as_str(),
                    "text_content": package.text_content,
                    "truncated": package.truncated,
                })
            })
            .collect(),
    )
}

fn record_preview_package_resolution(
    surface: PreviewPackageSurface,
    path: &'static str,
    elapsed: std::time::Duration,
) {
    metrics::counter!(
        crate::metrics::PREVIEW_PACKAGE_PATH_TOTAL,
        "surface" => surface.as_label(),
        "path" => path
    )
    .increment(1);
    metrics::histogram!(
        crate::metrics::PREVIEW_PACKAGE_RESOLUTION_SECONDS,
        "surface" => surface.as_label(),
        "path" => path
    )
    .record(elapsed.as_secs_f64());
}

pub(crate) fn resource_score_attribution_to_json(
    attributions: &[ResourceScoreAttribution],
) -> serde_json::Value {
    serde_json::Value::Array(
        attributions
            .iter()
            .map(|attribution| {
                serde_json::json!({
                    "resource_id": attribution.resource_id.to_string(),
                    "role": attribution.role.as_str(),
                    "display_name": attribution.display_name,
                    "modality": attribution.modality.map(|modality| modality.as_str()),
                    "artifact_kind": attribution.artifact_kind.as_str(),
                    "artifact_modality": attribution.artifact_modality.as_str(),
                    "matched_terms": attribution.matched_terms,
                    "match_score": attribution.match_score,
                    "score_boost": attribution.score_boost,
                })
            })
            .collect(),
    )
}

async fn package_resource_preview_packages(
    db: &HirnDB,
    actor_id: &AgentId,
    resource_evidence: &[ResourceEvidenceSummary],
    max_resource_previews: usize,
    max_resource_preview_chars: usize,
    preview_cache: &mut PreviewPackageCache,
) -> Vec<ResourcePreviewPackage> {
    let mut packaged = Vec::new();
    for summary in resource_evidence
        .iter()
        .filter(|summary| summary.has_preview && summary.can_hydrate_preview)
        .take(max_resource_previews)
    {
        let cached = if let Some(cached) = preview_cache.cached_previews.get(&summary.resource_id) {
            cached.clone()
        } else {
            let preview = match db
                .fetch_resource(actor_id, summary.resource_id, HydrationMode::Preview)
                .await
            {
                Ok(Some(resource)) => {
                    select_cached_resource_preview(&resource, max_resource_preview_chars)
                }
                Ok(None) | Err(_) => None,
            };
            preview_cache
                .cached_previews
                .insert(summary.resource_id, preview.clone());
            preview
        };

        if let Some(cached) = cached {
            packaged.push(ResourcePreviewPackage {
                resource_id: summary.resource_id,
                role: summary.role,
                display_name: summary.display_name.clone(),
                modality: summary.modality,
                artifact_kind: cached.artifact_kind,
                artifact_modality: cached.artifact_modality,
                text_content: cached.text_content,
                truncated: cached.truncated,
            });
        }
    }

    packaged
}

fn select_cached_resource_preview(
    hydrated: &hirn_storage::HydratedResource,
    max_chars: usize,
) -> Option<CachedResourcePreview> {
    let artifact = PREFERRED_PREVIEW_ARTIFACTS
        .iter()
        .find_map(|kind| {
            hydrated.artifacts.iter().find(|artifact| {
                artifact.kind == *kind
                    && artifact
                        .text_content
                        .as_deref()
                        .is_some_and(|text| !text.trim().is_empty())
            })
        })
        .or_else(|| {
            hydrated.artifacts.iter().find(|artifact| {
                artifact.kind.is_previewable()
                    && artifact
                        .text_content
                        .as_deref()
                        .is_some_and(|text| !text.trim().is_empty())
            })
        })?;

    let text = artifact.text_content.as_deref()?.trim();
    if text.is_empty() {
        return None;
    }

    let truncated = text.chars().count() > max_chars;
    let text_content = if truncated {
        hirn_core::text_util::truncate_at_word_boundary(text, max_chars)
    } else {
        text.to_string()
    };

    Some(CachedResourcePreview {
        artifact_kind: artifact.kind,
        artifact_modality: artifact.modality,
        text_content,
        truncated,
    })
}

fn best_resource_score_attribution(
    query_terms: &BTreeSet<String>,
    preview_packages: &[ResourcePreviewPackage],
) -> Option<ResourceScoreAttribution> {
    preview_packages
        .iter()
        .filter_map(|package| build_resource_score_attribution(query_terms, package))
        .max_by(|left, right| {
            left.match_score
                .total_cmp(&right.match_score)
                .then_with(|| left.score_boost.total_cmp(&right.score_boost))
        })
}

fn build_resource_score_attribution(
    query_terms: &BTreeSet<String>,
    package: &ResourcePreviewPackage,
) -> Option<ResourceScoreAttribution> {
    let preview_terms = normalized_terms(&package.text_content);
    if preview_terms.is_empty() {
        return None;
    }

    let matched_terms: Vec<String> = query_terms
        .intersection(&preview_terms)
        .take(MAX_ATTRIBUTION_TERMS)
        .cloned()
        .collect();
    if matched_terms.is_empty() {
        return None;
    }

    let coverage = matched_terms.len() as f32 / query_terms.len() as f32;
    let density = matched_terms.len() as f32 / preview_terms.len() as f32;
    let match_score = (coverage * 0.75 + density * 0.25).clamp(0.0, 1.0);
    let score_boost = (match_score * RESOURCE_PREVIEW_RERANK_WEIGHT).clamp(0.0, 1.0);

    Some(ResourceScoreAttribution {
        resource_id: package.resource_id,
        role: package.role,
        display_name: package.display_name.clone(),
        modality: package.modality,
        artifact_kind: package.artifact_kind,
        artifact_modality: package.artifact_modality,
        matched_terms,
        match_score,
        score_boost,
    })
}

pub(crate) fn reuse_seeded_preview_packages(
    packages: &[ResourcePreviewPackage],
    max_resource_previews: usize,
    max_resource_preview_chars: usize,
) -> Option<Vec<ResourcePreviewPackage>> {
    if max_resource_previews == 0 || max_resource_preview_chars == 0 {
        return Some(Vec::new());
    }
    if packages.len() < max_resource_previews {
        return None;
    }

    let mut reused = Vec::with_capacity(max_resource_previews);
    for package in packages.iter().take(max_resource_previews) {
        let visible_text = if package.truncated {
            package
                .text_content
                .strip_suffix("...")
                .unwrap_or(&package.text_content)
        } else {
            &package.text_content
        };
        let visible_chars = visible_text.chars().count();

        if package.truncated && max_resource_preview_chars > visible_chars {
            return None;
        }

        let mut reused_package = package.clone();
        if visible_chars > max_resource_preview_chars {
            reused_package.text_content = hirn_core::text_util::truncate_at_word_boundary(
                visible_text,
                max_resource_preview_chars,
            );
            reused_package.truncated = true;
        } else if package.truncated {
            reused_package
                .text_content
                .clone_from(&package.text_content);
            reused_package.truncated = true;
        } else {
            reused_package.text_content = visible_text.to_string();
            reused_package.truncated = false;
        }
        reused.push(reused_package);
    }

    Some(reused)
}

fn normalized_terms(text: &str) -> BTreeSet<String> {
    let mut terms = BTreeSet::new();
    let mut current = String::new();

    for ch in text.chars() {
        if ch.is_alphanumeric() {
            current.extend(ch.to_lowercase());
        } else {
            push_normalized_term(&mut terms, &mut current);
        }
    }
    push_normalized_term(&mut terms, &mut current);

    terms
}

fn push_normalized_term(terms: &mut BTreeSet<String>, current: &mut String) {
    if current.chars().count() >= 3 {
        terms.insert(std::mem::take(current));
    } else {
        current.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hirn_core::types::Namespace;

    #[test]
    fn resource_score_attribution_uses_query_overlap() {
        let package = ResourcePreviewPackage {
            resource_id: ResourceId::new(),
            role: hirn_core::EvidenceRole::Source,
            display_name: Some("incident.png".to_string()),
            modality: Some(ModalityProfile::Image),
            artifact_kind: DerivedArtifactKind::Preview,
            artifact_modality: ModalityProfile::Text,
            text_content: "waf latency spike timeline with annotated edge nodes".to_string(),
            truncated: false,
        };

        let attribution = build_resource_score_attribution(
            &normalized_terms("investigate waf latency spike"),
            &package,
        )
        .expect("preview text should overlap the query");

        assert!(attribution.match_score > 0.0);
        assert!(attribution.score_boost > 0.0);
        assert!(
            attribution
                .matched_terms
                .iter()
                .any(|term| term == "latency")
        );
    }

    #[test]
    fn seeded_preview_packages_can_be_reused_for_smaller_budget() {
        let packages = vec![ResourcePreviewPackage {
            resource_id: ResourceId::new(),
            role: hirn_core::EvidenceRole::Attachment,
            display_name: Some("preview".to_string()),
            modality: Some(ModalityProfile::Image),
            artifact_kind: DerivedArtifactKind::Preview,
            artifact_modality: ModalityProfile::Text,
            text_content: "alpha beta gamma delta epsilon zeta...".to_string(),
            truncated: true,
        }];

        let reused = reuse_seeded_preview_packages(&packages, 1, 15).unwrap();

        assert_eq!(reused.len(), 1);
        assert!(reused[0].truncated);
        assert!(reused[0].text_content.ends_with("..."));
        assert!(reused[0].text_content.contains("alpha beta"));
    }

    #[test]
    fn seeded_preview_packages_refetch_when_budget_needs_more_text() {
        let packages = vec![ResourcePreviewPackage {
            resource_id: ResourceId::new(),
            role: hirn_core::EvidenceRole::Attachment,
            display_name: Some("preview".to_string()),
            modality: Some(ModalityProfile::Image),
            artifact_kind: DerivedArtifactKind::Preview,
            artifact_modality: ModalityProfile::Text,
            text_content: "alpha beta gamma delta epsilon zeta...".to_string(),
            truncated: true,
        }];

        assert!(reuse_seeded_preview_packages(&packages, 1, 128).is_none());
    }

    #[test]
    fn cached_preview_prefers_caption_over_ocr_text_when_both_exist() {
        let resource = hirn_core::ResourceObject::builder()
            .modality(ModalityProfile::Image)
            .location(hirn_core::ResourceLocation::Blob { blob_index: 0 })
            .build()
            .unwrap();
        let hydrated = hirn_storage::HydratedResource {
            resource,
            artifacts: vec![
                hirn_core::DerivedArtifact::builder()
                    .resource_id(ResourceId::new())
                    .kind(DerivedArtifactKind::OcrText)
                    .modality(ModalityProfile::Text)
                    .text_content("fallback ocr text")
                    .namespace(Namespace::default())
                    .build()
                    .unwrap(),
                hirn_core::DerivedArtifact::builder()
                    .resource_id(ResourceId::new())
                    .kind(DerivedArtifactKind::Caption)
                    .modality(ModalityProfile::Text)
                    .text_content("caption text")
                    .namespace(Namespace::default())
                    .build()
                    .unwrap(),
            ],
            blob: None,
        };

        let preview = select_cached_resource_preview(&hydrated, 128).unwrap();
        assert_eq!(preview.artifact_kind, DerivedArtifactKind::Caption);
        assert_eq!(preview.text_content, "caption text");
    }
}
