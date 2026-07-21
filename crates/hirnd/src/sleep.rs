//! Sleep-time consolidation: opportunistic cognitive maintenance while idle.
//!
//! Inspired by "sleep-time compute" (Letta, arXiv:2504.13171): memory
//! reorganization runs while the daemon is quiet instead of on the request
//! hot path. Every authenticated request on the HTTP, gRPC, and MCP surfaces
//! touches a process-wide [`ActivityTracker`]; a background scheduler wakes
//! periodically and — once the daemon has been idle long enough and the
//! previous pass is old enough — runs one budgeted "sleep pass" per open
//! realm:
//!
//! 1. the consolidation pipeline (segmentation → patterns → communities →
//!    RAPTOR → forgetting) via `db.admin().consolidate()`, and
//! 2. a bounded set of offline cognition jobs (one dream + one reconcile)
//!    when the engine's offline scheduler is enabled.
//!
//! The pass aborts between phases as soon as foreground activity resumes, so
//! interactive traffic never waits behind maintenance work.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use hirn_core::{CognitiveJob, CognitiveJobKind, OfflineJobTarget};
use tracing::{debug, info, warn};

use crate::config::SleepConfig;
use crate::realm::RealmManager;

/// Agent identity used for daemon-initiated maintenance (Cedar policy checks
/// and offline-job attribution). Matches the identity the MCP surface falls
/// back to in insecure dev mode.
const SLEEP_AGENT_ID: &str = "system";

/// Milliseconds since the unix epoch.
pub fn now_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Cheap, lock-free record of foreground request activity.
///
/// `touch()` is one atomic `fetch_max` plus one `fetch_add`, so it is safe to
/// call on every request without measurable overhead.
pub struct ActivityTracker {
    /// Unix-ms timestamp of the most recent foreground request (0 = never).
    last_activity_ms: AtomicU64,
    /// Total number of foreground requests observed.
    requests_total: AtomicU64,
}

/// Single process-wide tracker shared by the HTTP, gRPC, and MCP surfaces.
/// A static (rather than an `Arc` threaded through every constructor) keeps
/// the per-surface touch-points to a single line each.
static GLOBAL_TRACKER: ActivityTracker = ActivityTracker::new();

/// Unix-ms timestamp when the last sleep pass finished (0 = never).
static LAST_PASS_FINISHED_MS: AtomicU64 = AtomicU64::new(0);

impl ActivityTracker {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            last_activity_ms: AtomicU64::new(0),
            requests_total: AtomicU64::new(0),
        }
    }

    /// The process-wide tracker used by the request surfaces.
    #[must_use]
    pub fn global() -> &'static Self {
        &GLOBAL_TRACKER
    }

    /// Record a foreground request at the current wall-clock time.
    pub fn touch(&self) {
        self.mark_active_at(now_unix_ms());
        self.requests_total.fetch_add(1, Ordering::Relaxed);
    }

    /// Advance the last-activity timestamp without counting a request
    /// (used for explicit timestamps in tests and for startup).
    pub fn mark_active_at(&self, now_ms: u64) {
        // fetch_max keeps the timestamp monotonic even if clocks step back
        // or two threads race.
        self.last_activity_ms.fetch_max(now_ms, Ordering::Relaxed);
    }

    #[must_use]
    pub fn last_activity_ms(&self) -> u64 {
        self.last_activity_ms.load(Ordering::Relaxed)
    }

    #[must_use]
    pub fn requests_total(&self) -> u64 {
        self.requests_total.load(Ordering::Relaxed)
    }
}

/// Unix-ms timestamp of the last completed sleep pass, if any.
/// Surfaced in `/debug/brain-stats`.
#[must_use]
pub fn last_pass_unix_ms() -> Option<u64> {
    match LAST_PASS_FINISHED_MS.load(Ordering::Relaxed) {
        0 => None,
        ms => Some(ms),
    }
}

/// Axum middleware that records request activity. Layered inside the auth
/// middleware so only authenticated API traffic resets the idle clock —
/// health probes and unauthenticated scans must not keep the daemon "awake".
pub async fn track_http_activity(
    request: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    ActivityTracker::global().touch();
    next.run(request).await
}

/// Pure gating decision for a sleep pass: the daemon must have been idle for
/// longer than `idle_after_secs`, and the previous pass (if any) must have
/// finished at least `min_pass_interval_secs` ago.
///
/// `last_activity_ms == 0` means "no request observed yet"; callers seed the
/// tracker at startup so a freshly booted daemon still waits one full idle
/// window before its first pass.
#[must_use]
pub fn should_run_pass(
    now_ms: u64,
    last_activity_ms: u64,
    last_pass_finished_ms: Option<u64>,
    cfg: &SleepConfig,
) -> bool {
    if !cfg.enabled {
        return false;
    }

    let idle_ms = cfg.idle_after_secs.saturating_mul(1000);
    if now_ms.saturating_sub(last_activity_ms) <= idle_ms {
        return false;
    }

    match last_pass_finished_ms {
        None => true,
        Some(finished_ms) => {
            let min_gap_ms = cfg.min_pass_interval_secs.saturating_mul(1000);
            now_ms.saturating_sub(finished_ms) >= min_gap_ms
        }
    }
}

/// What a single sleep pass accomplished (or skipped).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct SleepPassOutcome {
    /// Realms that completed at least the consolidation phase.
    pub realms_processed: usize,
    /// Consolidation runs that finished successfully.
    pub consolidations_run: usize,
    /// Offline cognition jobs enqueued (dream/reconcile).
    pub jobs_enqueued: usize,
    /// True when foreground activity resumed and the pass stopped early.
    pub aborted: bool,
}

/// Run one budgeted sleep pass across every open realm.
///
/// Between phases the tracker is re-checked: any foreground request that
/// arrives after the pass started aborts the remaining work so maintenance
/// never competes with live traffic.
pub async fn run_sleep_pass(realms: &RealmManager, tracker: &ActivityTracker) -> SleepPassOutcome {
    use tracing::Instrument;
    run_sleep_pass_inner(realms, tracker)
        .instrument(tracing::info_span!("sleep_pass"))
        .await
}

async fn run_sleep_pass_inner(
    realms: &RealmManager,
    tracker: &ActivityTracker,
) -> SleepPassOutcome {
    let pass_start_ms = now_unix_ms();
    let pass_start = Instant::now();
    let mut outcome = SleepPassOutcome::default();

    let activity_resumed = |outcome: &SleepPassOutcome| {
        let resumed = tracker.last_activity_ms() > pass_start_ms;
        if resumed {
            info!(
                realms_processed = outcome.realms_processed,
                "sleep pass aborted: foreground activity resumed"
            );
        }
        resumed
    };

    info!("sleep pass started");
    for realm in realms.realms().await {
        if activity_resumed(&outcome) {
            outcome.aborted = true;
            break;
        }

        let db = match realms.get(&realm).await {
            Ok(db) => db,
            Err(error) => {
                warn!(realm, %error, "sleep pass: failed to open realm — skipping");
                continue;
            }
        };

        // Phase 1: consolidation pipeline (segmentation → patterns →
        // communities → RAPTOR → forgetting). Engine defaults keep the run
        // budgeted; the daemon does not override thresholds.
        let phase_start = Instant::now();
        match db
            .admin()
            .consolidate()
            .agent_id(SLEEP_AGENT_ID)
            .execute()
            .await
        {
            Ok(result) => {
                outcome.consolidations_run += 1;
                info!(
                    realm,
                    phase = "consolidate",
                    duration_ms = phase_start.elapsed().as_millis() as u64,
                    records_processed = result.records_processed,
                    episodes_archived = result.episodes_archived,
                    "sleep pass: consolidation finished"
                );
            }
            Err(error) => {
                warn!(realm, phase = "consolidate", %error, "sleep pass: consolidation failed");
            }
        }

        if activity_resumed(&outcome) {
            outcome.aborted = true;
            info!(realm, "sleep pass: offline job phase skipped");
            break;
        }

        // Phase 2: bounded offline cognition (one dream + one reconcile),
        // only when the engine's offline scheduler is enabled. Budgets come
        // from the scheduler's configured default budget.
        if db.config().offline_scheduler.enabled {
            let phase_start = Instant::now();
            for kind in [CognitiveJobKind::Dream, CognitiveJobKind::Reconcile] {
                let mut job = CognitiveJob::new(kind, OfflineJobTarget::realm(&realm));
                job.scheduled_by = hirn_core::types::AgentId::new(SLEEP_AGENT_ID).ok();
                job.rationale = Some("idle-time sleep pass".to_string());
                match db.admin().schedule_offline_job(job).await {
                    Ok(job_id) => {
                        outcome.jobs_enqueued += 1;
                        info!(realm, phase = "offline_jobs", ?kind, %job_id, "sleep pass: job enqueued");
                    }
                    Err(error) => {
                        warn!(realm, phase = "offline_jobs", ?kind, %error, "sleep pass: enqueue failed");
                    }
                }
            }
            info!(
                realm,
                phase = "offline_jobs",
                duration_ms = phase_start.elapsed().as_millis() as u64,
                "sleep pass: offline job phase finished"
            );
        } else {
            debug!(
                realm,
                "sleep pass: offline scheduler disabled — dream/reconcile skipped"
            );
        }

        outcome.realms_processed += 1;
    }

    info!(
        duration_ms = pass_start.elapsed().as_millis() as u64,
        realms_processed = outcome.realms_processed,
        consolidations_run = outcome.consolidations_run,
        jobs_enqueued = outcome.jobs_enqueued,
        aborted = outcome.aborted,
        "sleep pass finished"
    );
    outcome
}

/// Background task that triggers sleep passes while the daemon is idle.
///
/// Spawned from `main.rs`; exits when the shutdown watch channel fires.
pub struct SleepScheduler {
    cfg: SleepConfig,
    realms: Arc<RealmManager>,
    last_pass_finished_ms: Option<u64>,
}

impl SleepScheduler {
    #[must_use]
    pub fn new(cfg: SleepConfig, realms: Arc<RealmManager>) -> Self {
        Self {
            cfg,
            realms,
            last_pass_finished_ms: None,
        }
    }

    /// Loop until shutdown: wake every `check_interval_secs`, run one pass
    /// when the idle and pass-spacing gates both open.
    pub async fn run(mut self, mut shutdown_rx: tokio::sync::watch::Receiver<()>) {
        // Treat startup as activity so the daemon must be quiet for a full
        // idle window before the first pass.
        ActivityTracker::global().mark_active_at(now_unix_ms());

        let interval = Duration::from_secs(self.cfg.check_interval_secs.max(1));
        info!(
            idle_after_secs = self.cfg.idle_after_secs,
            check_interval_secs = self.cfg.check_interval_secs,
            min_pass_interval_secs = self.cfg.min_pass_interval_secs,
            "sleep scheduler started"
        );

        loop {
            tokio::select! {
                _ = shutdown_rx.changed() => {
                    info!("sleep scheduler stopped (shutdown)");
                    return;
                }
                () = tokio::time::sleep(interval) => {}
            }

            let now = now_unix_ms();
            let last_activity = ActivityTracker::global().last_activity_ms();
            if should_run_pass(now, last_activity, self.last_pass_finished_ms, &self.cfg) {
                self.run_pass_once().await;
            }
        }
    }

    /// Run exactly one sleep pass and record its completion timestamp.
    /// Public so operators/tests can trigger a pass without the timer loop.
    pub async fn run_pass_once(&mut self) -> SleepPassOutcome {
        let outcome = run_sleep_pass(&self.realms, ActivityTracker::global()).await;

        let finished_ms = now_unix_ms();
        self.last_pass_finished_ms = Some(finished_ms);
        LAST_PASS_FINISHED_MS.fetch_max(finished_ms, Ordering::Relaxed);

        let result = if outcome.aborted {
            "aborted"
        } else {
            "completed"
        };
        metrics::counter!("hirnd_sleep_passes_total", "result" => result).increment(1);
        metrics::gauge!("hirnd_sleep_last_pass_timestamp_seconds").set((finished_ms / 1000) as f64);

        outcome
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> SleepConfig {
        SleepConfig {
            enabled: true,
            idle_after_secs: 300,
            check_interval_secs: 60,
            min_pass_interval_secs: 3600,
        }
    }

    const MIN: u64 = 60_000;

    #[test]
    fn pass_denied_while_active() {
        // Last request 1 minute ago, idle window is 5 minutes.
        assert!(!should_run_pass(10 * MIN, 9 * MIN, None, &cfg()));
    }

    #[test]
    fn pass_denied_at_exact_idle_boundary() {
        // now - last_activity == idle_after: not yet strictly idle.
        assert!(!should_run_pass(10 * MIN, 5 * MIN, None, &cfg()));
    }

    #[test]
    fn pass_allowed_when_idle_and_never_passed() {
        assert!(should_run_pass(10 * MIN, 4 * MIN, None, &cfg()));
    }

    #[test]
    fn pass_denied_when_previous_pass_too_recent() {
        // Idle, but the last pass finished 30 minutes ago (< 1 hour spacing).
        let now = 120 * MIN;
        assert!(!should_run_pass(now, 10 * MIN, Some(90 * MIN), &cfg()));
    }

    #[test]
    fn pass_allowed_at_exact_min_pass_interval() {
        // Spacing gate is inclusive: exactly min_pass_interval ago is enough.
        let now = 120 * MIN;
        assert!(should_run_pass(now, 10 * MIN, Some(60 * MIN), &cfg()));
    }

    #[test]
    fn pass_denied_when_disabled() {
        let mut disabled = cfg();
        disabled.enabled = false;
        assert!(!should_run_pass(10 * MIN, 0, None, &disabled));
    }

    #[test]
    fn pass_allowed_with_no_activity_ever_after_idle_window() {
        // last_activity 0 = never; well past the idle window.
        assert!(should_run_pass(10 * MIN, 0, None, &cfg()));
    }

    #[test]
    fn pass_denied_just_after_boot_when_tracker_seeded() {
        // The scheduler seeds the tracker at startup, so `now == last_activity`.
        assert!(!should_run_pass(7 * MIN, 7 * MIN, None, &cfg()));
    }

    #[test]
    fn clock_regression_does_not_panic_or_allow() {
        // now < last_activity (clock stepped back): saturating math treats
        // the daemon as active.
        assert!(!should_run_pass(MIN, 10 * MIN, None, &cfg()));
    }

    #[test]
    fn tracker_touch_is_monotonic() {
        let tracker = ActivityTracker::new();
        tracker.mark_active_at(1_000);
        tracker.mark_active_at(500); // stale timestamp must not regress
        assert_eq!(tracker.last_activity_ms(), 1_000);
        assert_eq!(tracker.requests_total(), 0);

        tracker.touch();
        assert!(tracker.last_activity_ms() >= 1_000);
        assert_eq!(tracker.requests_total(), 1);
    }

    async fn memory_realm() -> Arc<RealmManager> {
        let tmp = tempfile::TempDir::new().unwrap();
        let storage: Arc<dyn hirn_storage::PhysicalStore> =
            Arc::new(hirn_storage::memory_store::MemoryStore::new());
        let db = hirn_engine::HirnDB::open(&tmp.path().join("brain"), storage)
            .await
            .unwrap();
        Arc::new(RealmManager::from_db(Arc::new(db)))
    }

    /// Smoke test: a pass over a MemoryStore-backed default realm runs the
    /// consolidation phase and completes without abort. Offline jobs are
    /// skipped because the engine's offline scheduler defaults to disabled.
    #[tokio::test(flavor = "multi_thread")]
    async fn sleep_pass_runs_against_memory_realm() {
        let realms = memory_realm().await;
        let tracker = ActivityTracker::new();

        let outcome = run_sleep_pass(&realms, &tracker).await;

        assert_eq!(outcome.realms_processed, 1);
        assert_eq!(outcome.consolidations_run, 1);
        assert_eq!(outcome.jobs_enqueued, 0);
        assert!(!outcome.aborted);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn sleep_pass_aborts_when_activity_resumes() {
        let realms = memory_realm().await;
        let tracker = ActivityTracker::new();
        // Activity timestamped after the pass starts (future ms) forces the
        // between-phase check to see resumed traffic immediately.
        tracker.mark_active_at(now_unix_ms() + 60_000);

        let outcome = run_sleep_pass(&realms, &tracker).await;

        assert!(outcome.aborted);
        assert_eq!(outcome.realms_processed, 0);
        assert_eq!(outcome.consolidations_run, 0);
    }
}
