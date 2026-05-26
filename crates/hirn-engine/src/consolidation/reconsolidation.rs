use super::*;

// ═══════════════════════════════════════════════════════════════════════════
// Memory Reconsolidation
// ═══════════════════════════════════════════════════════════════════════════

/// Updates that can be applied during a reconsolidation window.
#[derive(Debug, Clone, Default)]
pub struct ReconsolidationUpdate {
    /// New importance value (if any).
    pub importance: Option<f32>,
    /// New surprise value (if any).
    pub surprise: Option<f32>,
    /// New summary (if any).
    pub summary: Option<String>,
    /// Additional entity references to add.
    pub new_entities: Vec<hirn_core::episodic::EntityRef>,
    /// Additional graph links to add.
    pub new_links: Vec<(MemoryId, EdgeRelation)>,
    /// Updated embedding (if any).
    pub embedding: Option<Vec<f32>>,
    /// Reason for the reconsolidation.
    pub reason: String,
}

/// Tracks labile (reconsolidation) windows for retrieved memories.
///
/// F-36: Uses `Timestamp` instead of `Instant` so window state is recoverable
/// across restarts (timestamps are absolute wall-clock millis).
pub struct ReconsolidationTracker {
    /// Maps memory ID → (opened_at timestamp, window duration in seconds).
    windows: parking_lot::RwLock<HashMap<MemoryId, (hirn_core::Timestamp, u64)>>,
}

impl ReconsolidationTracker {
    pub fn new() -> Self {
        Self {
            windows: parking_lot::RwLock::new(HashMap::new()),
        }
    }

    /// Open a labile window for a retrieved memory.
    pub fn open_window(&self, id: MemoryId, duration_secs: u64) {
        let mut windows = self.windows.write();
        windows.insert(id, (hirn_core::Timestamp::now(), duration_secs));
    }

    /// Check if a memory is currently in a labile state.
    pub fn is_labile(&self, id: MemoryId) -> bool {
        let windows = self.windows.read();
        if let Some((opened_at, duration)) = windows.get(&id) {
            let now = hirn_core::Timestamp::now();
            let elapsed_ms = now.millis().saturating_sub(opened_at.millis());
            elapsed_ms < duration.saturating_mul(1000)
        } else {
            false
        }
    }

    /// Close an expired window. Returns true if the window was open.
    pub fn close_window(&self, id: MemoryId) -> bool {
        let mut windows = self.windows.write();
        windows.remove(&id).is_some()
    }

    /// Garbage-collect expired windows.
    pub fn gc(&self) {
        let now = hirn_core::Timestamp::now();
        let mut windows = self.windows.write();
        windows.retain(|_, (opened_at, duration)| {
            let elapsed_ms = now.millis().saturating_sub(opened_at.millis());
            elapsed_ms < duration.saturating_mul(1000)
        });
    }

    /// Snapshot all open windows for persistence (F-36).
    pub fn snapshot(&self) -> Vec<(MemoryId, u64, u64)> {
        let windows = self.windows.read();
        windows
            .iter()
            .map(|(&id, &(ref ts, dur))| (id, ts.millis(), dur))
            .collect()
    }

    /// Restore windows from a persisted snapshot (F-36).
    pub fn restore(&self, entries: &[(MemoryId, u64, u64)]) {
        let now = hirn_core::Timestamp::now();
        let mut windows = self.windows.write();
        for &(id, opened_ms, duration_secs) in entries {
            let elapsed_ms = now.millis().saturating_sub(opened_ms);
            // Only restore windows that haven't expired yet.
            if elapsed_ms < duration_secs.saturating_mul(1000) {
                windows.insert(
                    id,
                    (hirn_core::Timestamp::from_millis(opened_ms), duration_secs),
                );
            }
        }
    }
}

impl Default for ReconsolidationTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Apply a reconsolidation update to a memory during its labile window.
pub async fn reconsolidate(
    db: &HirnDB,
    tracker: &ReconsolidationTracker,
    id: MemoryId,
    update: &ReconsolidationUpdate,
) -> HirnResult<()> {
    if !tracker.is_labile(id) {
        return Err(HirnError::InvalidInput(
            "memory is not in labile reconsolidation state".to_string(),
        ));
    }

    // Apply the reconsolidation to the episodic record.
    let updated = apply_episodic_reconsolidation(db, id, update).await?;

    // Create new graph edges if any.
    for (target_id, relation) in &update.new_links {
        db.connect_with(updated.id, *target_id, *relation, 0.5, Metadata::default())
            .await?;
    }

    Ok(())
}

/// Apply reconsolidation updates to an episodic record.
async fn apply_episodic_reconsolidation(
    db: &HirnDB,
    id: MemoryId,
    update: &ReconsolidationUpdate,
) -> HirnResult<hirn_core::episodic::EpisodicRecord> {
    db.update_episode_returning_head(id, |rec| {
        let now = Timestamp::now();

        if let Some(new_importance) = update.importance {
            let mutation = Mutation {
                timestamp: now,
                trigger: MutationTrigger::Reconsolidation,
                field: "importance".to_string(),
                old_value: rec.importance.to_string(),
                new_value: new_importance.to_string(),
                reason: update.reason.clone(),
            };
            rec.provenance.record_mutation(mutation);
            rec.importance = new_importance.clamp(0.0, 1.0);
        }

        if let Some(new_surprise) = update.surprise {
            let mutation = Mutation {
                timestamp: now,
                trigger: MutationTrigger::Reconsolidation,
                field: "surprise".to_string(),
                old_value: rec.surprise.to_string(),
                new_value: new_surprise.to_string(),
                reason: update.reason.clone(),
            };
            rec.provenance.record_mutation(mutation);
            rec.surprise = new_surprise.clamp(0.0, 1.0);
        }

        if let Some(ref new_summary) = update.summary {
            let mutation = Mutation {
                timestamp: now,
                trigger: MutationTrigger::Reconsolidation,
                field: "summary".to_string(),
                old_value: rec.summary.clone(),
                new_value: new_summary.clone(),
                reason: update.reason.clone(),
            };
            rec.provenance.record_mutation(mutation);
            rec.summary.clone_from(new_summary);
        }

        // Append new entities (dedup by name).
        if !update.new_entities.is_empty() {
            let existing_names: std::collections::HashSet<String> =
                rec.entities.iter().map(|e| e.name.clone()).collect();
            for entity in &update.new_entities {
                if !existing_names.contains(&entity.name) {
                    rec.entities.push(entity.clone());
                }
            }
        }

        // Update embedding if provided.
        if let Some(ref emb) = update.embedding {
            rec.embedding = Some(emb.clone());
        }
    })
    .await
}
