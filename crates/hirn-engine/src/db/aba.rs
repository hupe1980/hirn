//! ABA conflict resolution with AGM belief revision.
//!
//! Resolves contradictions using Assumption-Based Argumentation (ABA): each
//! memory is an argument with assumptions (source, recency, evidence count),
//! and the grounded extension determines the "winner". The pure decision
//! functions here (`resolve_aba`, `resolve_aba_multi`) compute *who wins* and
//! the loser's AGM-contracted confidence; the mutation is applied by
//! [`CausalView::apply_aba_resolution`](crate::CausalView::apply_aba_resolution),
//! which reduces the loser's importance, appends a successor revision, and
//! records the reconsolidation metadata.
//!
//! This decision logic previously lived inside a DataFusion `ExecutionPlan`
//! operator that was never emitted into any compiled plan (R-20b). The operator
//! was retired; the algorithm lives here in the engine next to its apply path.

/// ABA resolution result for a contradiction pair.
#[derive(Debug, Clone)]
pub struct AbaResolution {
    /// The winning memory ID.
    pub winner_id: String,
    /// The losing memory ID.
    pub loser_id: String,
    /// Reason for the resolution.
    pub reason: String,
    /// Revised confidence for the loser (reduced, not zero).
    pub loser_revised_confidence: f32,
}

/// Resolve a contradiction between two arguments using ABA grounded semantics.
///
/// The argument with higher composite support wins. The loser's confidence
/// is reduced but not zeroed (AGM contraction: minimal change principle).
pub fn resolve_aba(id_a: &str, score_a: f32, id_b: &str, score_b: f32) -> AbaResolution {
    // Composite support score: higher score = stronger argument.
    // In a full implementation, this would consider:
    // - Evidence count (from provenance)
    // - Source reliability (from origin type)
    // - Recency (from timestamp)
    // - Supporting evidence chain length
    // For now, we use the retrieval score as a proxy for argument strength.

    if score_a >= score_b {
        AbaResolution {
            winner_id: id_a.to_string(),
            loser_id: id_b.to_string(),
            reason: format!(
                "argument {} (score={:.3}) defeats {} (score={:.3}) by grounded extension",
                id_a, score_a, id_b, score_b
            ),
            // AGM contraction: reduce loser confidence by 30-50% depending on margin.
            loser_revised_confidence: score_b * agm_contraction_factor(score_a, score_b),
        }
    } else {
        AbaResolution {
            winner_id: id_b.to_string(),
            loser_id: id_a.to_string(),
            reason: format!(
                "argument {} (score={:.3}) defeats {} (score={:.3}) by grounded extension",
                id_b, score_b, id_a, score_a
            ),
            loser_revised_confidence: score_a * agm_contraction_factor(score_b, score_a),
        }
    }
}

/// AGM contraction factor: how much to reduce the loser's confidence.
///
/// Large margins → more contraction (0.3–0.5 retention).
/// Small margins → less contraction (0.6–0.8 retention).
fn agm_contraction_factor(winner_score: f32, loser_score: f32) -> f32 {
    let margin = (winner_score - loser_score).abs();
    // Scale: margin 0 → retain 0.8, margin 1 → retain 0.3
    (0.8 - margin * 0.5).clamp(0.3, 0.8)
}

/// Resolve a multi-argument cycle using ABA grounded extension.
///
/// Given N mutually contradicting arguments (each identified by an ID and
/// a composite score), computes the grounded extension: the unique minimal
/// complete set of arguments that survives all attacks.
///
/// Algorithm:
/// 1. Fixed-point iteration: start with all arguments acceptable.
/// 2. Each round, an argument is "defeated" if any undefeated argument
///    with a strictly higher score attacks it.
/// 3. Iterate until stable (no changes).
/// 4. Remaining arguments form the grounded extension (winners).
/// 5. Losers get AGM contraction relative to the best winner.
///
/// Returns (winners, losers_with_revised_confidence).
pub fn resolve_aba_multi(args: &[(&str, f32)]) -> (Vec<String>, Vec<AbaResolution>) {
    if args.is_empty() {
        return (Vec::new(), Vec::new());
    }
    if args.len() == 1 {
        return (vec![args[0].0.to_string()], Vec::new());
    }
    if args.len() == 2 {
        let res = resolve_aba(args[0].0, args[0].1, args[1].0, args[1].1);
        let winner = res.winner_id.clone();
        return (vec![winner], vec![res]);
    }

    // Fixed-point iteration for grounded extension.
    let mut alive: Vec<bool> = vec![true; args.len()];
    let mut changed = true;

    while changed {
        changed = false;
        for i in 0..args.len() {
            if !alive[i] {
                continue;
            }
            // Check if any alive argument strictly defeats this one.
            for j in 0..args.len() {
                if i == j || !alive[j] {
                    continue;
                }
                if args[j].1 > args[i].1 {
                    alive[i] = false;
                    changed = true;
                    break;
                }
            }
        }
    }

    // Collect winners (survived arguments).
    let winners: Vec<String> = args
        .iter()
        .enumerate()
        .filter(|(i, _)| alive[*i])
        .map(|(_, (id, _))| (*id).to_string())
        .collect();

    // Best winner score for AGM contraction.
    let best_winner_score = args
        .iter()
        .enumerate()
        .filter(|(i, _)| alive[*i])
        .map(|(_, (_, s))| *s)
        .max_by(|a, b| a.total_cmp(b))
        .unwrap_or(0.0);

    // Losers: defeated arguments with revised confidence.
    let losers: Vec<AbaResolution> = args
        .iter()
        .enumerate()
        .filter(|(i, _)| !alive[*i])
        .map(|(_, (id, score))| {
            let factor = agm_contraction_factor(best_winner_score, *score);
            AbaResolution {
                winner_id: winners.first().cloned().unwrap_or_default(),
                loser_id: (*id).to_string(),
                reason: format!(
                    "grounded extension: {} defeated by winner(s) {:?} (score={:.3} vs best={:.3})",
                    id, winners, score, best_winner_score
                ),
                loser_revised_confidence: score * factor,
            }
        })
        .collect();

    (winners, losers)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn newer_evidence_wins() {
        let result = resolve_aba("mem_new", 0.9, "mem_old", 0.4);
        assert_eq!(result.winner_id, "mem_new");
        assert_eq!(result.loser_id, "mem_old");
        assert!(result.loser_revised_confidence < 0.4);
        assert!(!result.reason.is_empty());
    }

    #[test]
    fn loser_confidence_reduced_not_zeroed() {
        let result = resolve_aba("a", 0.8, "b", 0.6);
        assert!(result.loser_revised_confidence > 0.0);
        assert!(result.loser_revised_confidence < 0.6);
    }

    #[test]
    fn grounded_extension_clear_hierarchy() {
        let args = vec![("A", 0.9_f32), ("B", 0.6), ("C", 0.3)];
        let (winners, losers) = resolve_aba_multi(&args);
        assert_eq!(winners, vec!["A".to_string()]);
        assert_eq!(losers.len(), 2);
        for loser in &losers {
            assert!(loser.loser_revised_confidence > 0.0);
        }
    }

    #[test]
    fn grounded_extension_tie_both_survive() {
        let args = vec![("A", 0.7_f32), ("B", 0.7), ("C", 0.3)];
        let (winners, losers) = resolve_aba_multi(&args);
        assert_eq!(winners.len(), 2);
        assert!(winners.contains(&"A".to_string()));
        assert!(winners.contains(&"B".to_string()));
        assert_eq!(losers.len(), 1);
        assert_eq!(losers[0].loser_id, "C");
    }
}
