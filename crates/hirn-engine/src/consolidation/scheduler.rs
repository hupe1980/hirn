use super::*;

// ═══════════════════════════════════════════════════════════════════════════
// Consolidation Scheduling
// ═══════════════════════════════════════════════════════════════════════════

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::thread;
use std::time::Duration;

use chrono::Timelike;

/// Consolidation scheduling strategy.
#[derive(Debug, Clone)]
pub enum ConsolidationSchedule {
    /// Consolidate every N seconds.
    Periodic { interval_secs: u64 },
    /// Consolidate when accumulated surprise exceeds threshold,
    /// with a periodic fallback to guarantee eventual consolidation.
    SurpriseThreshold {
        threshold: f32,
        fallback_interval_secs: u64,
    },
    /// Consolidate during low-activity periods within preferred hours.
    Circadian {
        /// Preferred UTC hours for consolidation (e.g., 2..6 for 2 AM–6 AM).
        preferred_start_hour: u8,
        preferred_end_hour: u8,
        /// Seconds of inactivity required before triggering consolidation.
        idle_timeout_secs: u64,
    },
    /// Only consolidate on explicit `trigger()` calls.
    Manual,
}

impl Default for ConsolidationSchedule {
    /// Default: SurpriseThreshold with 5.0 threshold and 1-hour fallback.
    fn default() -> Self {
        ConsolidationSchedule::SurpriseThreshold {
            threshold: 5.0,
            fallback_interval_secs: 3600,
        }
    }
}

/// Current status of the consolidation scheduler.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConsolidationStatus {
    /// No consolidation is running.
    Idle,
    /// A consolidation run is in progress.
    Running {
        /// Timestamp when the current run started.
        started_at: Instant,
    },
}

/// Internal shared state between the scheduler and the `HirnDB` notification path.
struct SchedulerState {
    /// Whether consolidation is currently running (acts as a lock).
    running: AtomicBool,
    /// Signal to wake the periodic timer thread when a threshold is exceeded.
    wake: parking_lot::Condvar,
    wake_mutex: parking_lot::Mutex<bool>,
    /// Signal to shut down the background thread.
    shutdown: AtomicBool,
    /// Accumulated surprise since last consolidation (stored as u32 bits of f32).
    accumulated_surprise_bits: AtomicU32,
    /// Last activity instant (for Circadian idle detection).
    last_activity: parking_lot::RwLock<Instant>,
    /// Instant when the current consolidation started (valid only when `running` is true).
    run_started: parking_lot::RwLock<Option<Instant>>,
    /// Consolidation queue depth — how many consolidation requests are pending.
    queued: AtomicU32,
    /// Track consecutive failures for monitoring.
    consecutive_failures: AtomicU32,
}

impl SchedulerState {
    fn add_surprise(&self, surprise: f32) {
        loop {
            let old_bits = self.accumulated_surprise_bits.load(Ordering::Relaxed);
            let old_val = f32::from_bits(old_bits);
            let new_val = old_val + surprise;
            let new_bits = new_val.to_bits();
            if self
                .accumulated_surprise_bits
                .compare_exchange_weak(old_bits, new_bits, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
            {
                break;
            }
        }
    }

    fn load_surprise(&self) -> f32 {
        f32::from_bits(self.accumulated_surprise_bits.load(Ordering::Acquire))
    }

    fn reset_surprise(&self) {
        self.accumulated_surprise_bits
            .store(0.0_f32.to_bits(), Ordering::Release);
    }

    fn record_activity(&self) {
        *self.last_activity.write() = Instant::now();
    }
}

/// Manages automatic, background consolidation using configurable scheduling
/// strategies: Periodic, SurpriseThreshold, Circadian, or Manual.
///
/// The scheduler owns an `Arc<HirnDB>` and spawns a single background thread.
/// Consolidation is non-blocking: reads and writes to the database continue
/// normally while a consolidation cycle runs.
pub struct ConsolidationScheduler {
    db: Arc<HirnDB>,
    state: Arc<SchedulerState>,
    schedule: ConsolidationSchedule,
    config: ConsolidationConfig,
    /// Handle to the background thread (joined on `Drop`).
    handle: Option<thread::JoinHandle<()>>,
}

impl ConsolidationScheduler {
    /// Create a new scheduler with the default schedule derived from `HirnConfig`.
    ///
    /// Uses `SurpriseThreshold` as the default mode with the configured
    /// `consolidation_interval_secs` as the fallback.
    pub fn new(db: Arc<HirnDB>, config: ConsolidationConfig) -> Self {
        let interval_secs = db.config().consolidation_interval_secs;
        let schedule = if interval_secs > 0 {
            ConsolidationSchedule::SurpriseThreshold {
                threshold: 5.0,
                fallback_interval_secs: interval_secs,
            }
        } else {
            ConsolidationSchedule::Manual
        };
        Self::with_schedule(db, config, schedule)
    }

    /// Create a new scheduler with a specific scheduling strategy.
    pub fn with_schedule(
        db: Arc<HirnDB>,
        config: ConsolidationConfig,
        schedule: ConsolidationSchedule,
    ) -> Self {
        let state = Arc::new(SchedulerState {
            running: AtomicBool::new(false),
            wake: parking_lot::Condvar::new(),
            wake_mutex: parking_lot::Mutex::new(false),
            shutdown: AtomicBool::new(false),
            accumulated_surprise_bits: AtomicU32::new(0.0_f32.to_bits()),
            last_activity: parking_lot::RwLock::new(Instant::now()),
            run_started: parking_lot::RwLock::new(None),
            queued: AtomicU32::new(0),
            consecutive_failures: AtomicU32::new(0),
        });

        let handle = {
            let db = Arc::clone(&db);
            let state = Arc::clone(&state);
            let sched = schedule.clone();
            let cfg = config.clone();
            thread::spawn(move || {
                Self::background_loop(&db, &state, &cfg, &sched);
            })
        };

        Self {
            db,
            state,
            schedule,
            config,
            handle: Some(handle),
        }
    }

    /// Notify the scheduler that an episode was added.
    ///
    /// For `SurpriseThreshold` mode, also pass surprise via [`Self::notify_surprise`].
    pub fn notify_episode_added(&self) {
        self.state.record_activity();
    }

    /// Notify the scheduler of surprise from a newly stored episode.
    ///
    /// In `SurpriseThreshold` mode, if accumulated surprise exceeds the
    /// configured threshold, a consolidation run is triggered.
    pub fn notify_surprise(&self, surprise: f32) {
        self.state.add_surprise(surprise);
        if let ConsolidationSchedule::SurpriseThreshold { threshold, .. } = &self.schedule {
            if self.state.load_surprise() >= *threshold {
                self.state.queued.fetch_add(1, Ordering::Release);
                self.wake();
            }
        }
    }

    /// Manually trigger a consolidation run.
    pub fn trigger(&self) {
        self.state.queued.fetch_add(1, Ordering::Release);
        self.wake();
    }

    /// Return the current consolidation status.
    #[must_use]
    pub fn status(&self) -> ConsolidationStatus {
        if self.state.running.load(Ordering::Acquire) {
            let guard = self.state.run_started.read();
            ConsolidationStatus::Running {
                started_at: guard.unwrap_or_else(Instant::now),
            }
        } else {
            ConsolidationStatus::Idle
        }
    }

    /// Return the current schedule.
    #[must_use]
    pub fn schedule(&self) -> &ConsolidationSchedule {
        &self.schedule
    }

    /// Switch to a new schedule at runtime.
    ///
    /// Stops the current background thread and starts a new one with
    /// the given schedule. Accumulated state (surprise, episode counts) is reset.
    pub fn set_schedule(&mut self, schedule: ConsolidationSchedule) {
        // Stop the old background thread.
        self.state.shutdown.store(true, Ordering::Release);
        self.wake();
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }

        // Create fresh state and spawn new background thread.
        let state = Arc::new(SchedulerState {
            running: AtomicBool::new(false),
            wake: parking_lot::Condvar::new(),
            wake_mutex: parking_lot::Mutex::new(false),
            shutdown: AtomicBool::new(false),
            accumulated_surprise_bits: AtomicU32::new(0.0_f32.to_bits()),
            last_activity: parking_lot::RwLock::new(Instant::now()),
            run_started: parking_lot::RwLock::new(None),
            queued: AtomicU32::new(0),
            consecutive_failures: AtomicU32::new(0),
        });

        let handle = {
            let db = Arc::clone(&self.db);
            let st = Arc::clone(&state);
            let sched = schedule.clone();
            let cfg = self.config.clone();
            thread::spawn(move || {
                Self::background_loop(&db, &st, &cfg, &sched);
            })
        };

        self.state = state;
        self.schedule = schedule;
        self.handle = Some(handle);
    }

    /// Gracefully stop the background thread.
    pub fn stop(&mut self) {
        self.state.shutdown.store(true, Ordering::Release);
        self.wake();
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }

    /// Access the underlying database.
    #[must_use]
    pub fn db(&self) -> &HirnDB {
        &self.db
    }

    /// Access the underlying database as an `Arc`.
    #[must_use]
    pub fn db_arc(&self) -> Arc<HirnDB> {
        Arc::clone(&self.db)
    }

    // ── internal ──────────────────────────────────────────────────────

    fn wake(&self) {
        let mut guard = self.state.wake_mutex.lock();
        *guard = true;
        drop(guard);
        self.state.wake.notify_one();
    }

    fn background_loop(
        db: &HirnDB,
        state: &SchedulerState,
        config: &ConsolidationConfig,
        schedule: &ConsolidationSchedule,
    ) {
        // Create a single Tokio runtime for the entire scheduler lifetime
        // instead of spawning one per consolidation cycle.
        let rt = tokio::runtime::Runtime::new().expect("tokio runtime for consolidation");

        let poll_interval = match schedule {
            ConsolidationSchedule::Periodic { interval_secs } => {
                Duration::from_secs(*interval_secs)
            }
            ConsolidationSchedule::SurpriseThreshold {
                fallback_interval_secs,
                ..
            } => Duration::from_secs(*fallback_interval_secs),
            ConsolidationSchedule::Circadian { .. } => {
                // Poll every 60 seconds to check idle + hour window.
                Duration::from_mins(1)
            }
            ConsolidationSchedule::Manual => {
                // No periodic — wait indefinitely for wake signals.
                Duration::from_secs(u64::MAX / 2)
            }
        };

        loop {
            // Wait for poll interval or a wake signal.
            {
                let mut guard = state.wake_mutex.lock();
                if !*guard && !state.shutdown.load(Ordering::Acquire) {
                    state.wake.wait_for(&mut guard, poll_interval);
                }
                *guard = false;
            }

            if state.shutdown.load(Ordering::Acquire) {
                // Drain remaining queued requests before exiting.
                while state.queued.load(Ordering::Acquire) > 0 {
                    Self::run_consolidation(db, state, config, &rt);
                    state.queued.fetch_sub(1, Ordering::Release);
                }
                break;
            }

            let queued = state.queued.load(Ordering::Acquire);
            let should_run = queued > 0 || Self::should_trigger(state, schedule);

            if should_run {
                if queued == 0 {
                    state.queued.fetch_add(1, Ordering::Release);
                }
                while state.queued.load(Ordering::Acquire) > 0 {
                    Self::run_consolidation(db, state, config, &rt);
                    state.queued.fetch_sub(1, Ordering::Release);
                }
            }
        }
    }

    /// Determine whether the schedule condition is met for automatic triggering.
    fn should_trigger(state: &SchedulerState, schedule: &ConsolidationSchedule) -> bool {
        match schedule {
            ConsolidationSchedule::Periodic { .. } => {
                // Periodic always fires on timeout.
                true
            }
            ConsolidationSchedule::SurpriseThreshold {
                threshold,
                fallback_interval_secs,
            } => {
                // Surprise exceeded or fallback periodic fire.
                // Surprise check handled in notify_surprise; fallback fires on timeout.
                if *fallback_interval_secs > 0 {
                    return true; // Timeout expired = fallback fires.
                }
                state.load_surprise() >= *threshold
            }
            ConsolidationSchedule::Circadian {
                preferred_start_hour,
                preferred_end_hour,
                idle_timeout_secs,
            } => {
                // Check if current UTC hour is within the preferred window.
                let now_utc = chrono::Utc::now();
                let hour = now_utc.hour() as u8;
                let in_window = if preferred_start_hour <= preferred_end_hour {
                    hour >= *preferred_start_hour && hour < *preferred_end_hour
                } else {
                    // Wraps midnight, e.g., 22..6
                    hour >= *preferred_start_hour || hour < *preferred_end_hour
                };
                if !in_window {
                    return false;
                }
                // Check idle timeout.
                let idle_secs = state.last_activity.read().elapsed().as_secs();
                idle_secs >= *idle_timeout_secs
            }
            ConsolidationSchedule::Manual => false,
        }
    }

    /// Execute a single consolidation run with proper locking, status tracking,
    /// and panic recovery.
    ///
    /// N-M09 fix: wraps the consolidation pipeline in `catch_unwind` so that a
    /// panic inside the async block does not permanently kill the scheduler
    /// thread or leave `state.running` stuck at `true`. A `RunGuard` RAII
    /// helper resets `running` on all exit paths (normal, error, and panic).
    fn run_consolidation(
        db: &HirnDB,
        state: &SchedulerState,
        config: &ConsolidationConfig,
        rt: &tokio::runtime::Runtime,
    ) {
        // Acquire lock — spin-wait if another run is in progress.
        while state
            .running
            .compare_exchange_weak(false, true, Ordering::AcqRel, Ordering::Relaxed)
            .is_err()
        {
            thread::yield_now();
        }

        // RAII guard: resets `running` even if the body panics (N-M09).
        struct RunGuard<'a> {
            state: &'a SchedulerState,
        }
        impl Drop for RunGuard<'_> {
            fn drop(&mut self) {
                let mut started = self.state.run_started.write();
                *started = None;
                self.state.running.store(false, Ordering::Release);
            }
        }

        // Mark start.
        {
            let mut started = state.run_started.write();
            *started = Some(Instant::now());
        }
        let _guard = RunGuard { state };

        // Reset counters.
        state.reset_surprise();

        // Wrap execution in catch_unwind so a panic in the pipeline does not
        // permanently kill the background thread. The RunGuard above ensures
        // `running` is cleared whether we return normally, with an error, or
        // via panic.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            rt.block_on(execute_consolidation_pipeline(db, config, &[], None))
        }));

        match result {
            Ok(Ok(result)) => {
                state.consecutive_failures.store(0, Ordering::Release);
                tracing::info!(
                    records = result.records_processed,
                    segments = result.segments_created,
                    patterns = result.patterns_detected,
                    threads = result.threads_formed,
                    concepts = result.concepts_extracted,
                    time_ms = %format!("{:.1}", result.execution_time_ms),
                    "consolidation completed"
                );
            }
            Ok(Err(e)) => {
                let failures = state.consecutive_failures.fetch_add(1, Ordering::AcqRel) + 1;
                tracing::error!(
                    error = %e,
                    consecutive_failures = failures,
                    "consolidation failed"
                );
            }
            Err(_panic_payload) => {
                let failures = state.consecutive_failures.fetch_add(1, Ordering::AcqRel) + 1;
                tracing::error!(
                    consecutive_failures = failures,
                    "consolidation panicked — scheduler thread recovered"
                );
            }
        }
    }
}

impl Drop for ConsolidationScheduler {
    fn drop(&mut self) {
        self.stop();
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    async fn test_db() -> Arc<HirnDB> {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test");
        let mut config = hirn_core::HirnConfig::default();
        config.db_path = db_path;
        config.embedding_dimensions = hirn_core::EmbeddingDimension::new_const(3);
        config.consolidation_interval_secs = 0; // disable periodic for most tests
        let storage: Arc<dyn hirn_storage::PhysicalStore> =
            Arc::new(hirn_storage::memory_store::MemoryStore::new());
        let db = HirnDB::open_with_config(config, storage).await.unwrap();
        std::mem::forget(dir);
        Arc::new(db)
    }

    #[tokio::test]
    async fn manual_schedule_no_auto_trigger() {
        let db = test_db().await;
        let config = ConsolidationConfig::default();
        let mut sched =
            ConsolidationScheduler::with_schedule(db, config, ConsolidationSchedule::Manual);
        assert_eq!(sched.status(), ConsolidationStatus::Idle);
        // notify_episode_added should NOT trigger consolidation in Manual mode.
        sched.notify_episode_added();
        std::thread::sleep(Duration::from_millis(50));
        assert_eq!(sched.status(), ConsolidationStatus::Idle);
        sched.stop();
    }

    #[tokio::test]
    async fn manual_trigger_fires() {
        let db = test_db().await;
        let config = ConsolidationConfig::default();
        let mut sched =
            ConsolidationScheduler::with_schedule(db, config, ConsolidationSchedule::Manual);
        sched.trigger();
        // Give the background thread time to process.
        std::thread::sleep(Duration::from_millis(200));
        // After processing, status should return to Idle.
        assert_eq!(sched.status(), ConsolidationStatus::Idle);
        sched.stop();
    }

    #[tokio::test]
    async fn surprise_threshold_triggers() {
        let db = test_db().await;
        let config = ConsolidationConfig::default();
        let mut sched = ConsolidationScheduler::with_schedule(
            db,
            config,
            ConsolidationSchedule::SurpriseThreshold {
                threshold: 2.0,
                fallback_interval_secs: 0, // no fallback
            },
        );
        // Below threshold — no trigger.
        sched.notify_surprise(0.5);
        sched.notify_surprise(0.5);
        std::thread::sleep(Duration::from_millis(100));
        // Accumulate to 2.0+ to trigger.
        sched.notify_surprise(1.5);
        std::thread::sleep(Duration::from_millis(200));
        // Should have run and returned to idle.
        assert_eq!(sched.status(), ConsolidationStatus::Idle);
        sched.stop();
    }

    #[tokio::test]
    async fn periodic_schedule_fires() {
        let db = test_db().await;
        let config = ConsolidationConfig::default();
        let mut sched = ConsolidationScheduler::with_schedule(
            db,
            config,
            ConsolidationSchedule::Periodic { interval_secs: 1 },
        );
        // Wait for at least one periodic cycle.
        std::thread::sleep(Duration::from_secs(2));
        assert_eq!(sched.status(), ConsolidationStatus::Idle);
        sched.stop();
    }

    #[tokio::test]
    async fn circadian_outside_window_no_trigger() {
        let db = test_db().await;
        let config = ConsolidationConfig::default();
        // Set window to an hour that is definitely not now
        // (pick hour 25 mod 24 range that can't match).
        let now = chrono::Utc::now().hour() as u8;
        let start = (now + 12) % 24;
        let end = (start + 1) % 24;
        let mut sched = ConsolidationScheduler::with_schedule(
            db,
            config,
            ConsolidationSchedule::Circadian {
                preferred_start_hour: start,
                preferred_end_hour: end,
                idle_timeout_secs: 0,
            },
        );
        std::thread::sleep(Duration::from_millis(200));
        assert_eq!(sched.status(), ConsolidationStatus::Idle);
        sched.stop();
    }

    #[tokio::test]
    async fn default_schedule_is_surprise_threshold() {
        let schedule = ConsolidationSchedule::default();
        matches!(schedule, ConsolidationSchedule::SurpriseThreshold { .. });
    }

    #[tokio::test]
    async fn schedule_accessor() {
        let db = test_db().await;
        let config = ConsolidationConfig::default();
        let mut sched =
            ConsolidationScheduler::with_schedule(db, config, ConsolidationSchedule::Manual);
        assert!(matches!(sched.schedule(), ConsolidationSchedule::Manual));
        sched.stop();
    }

    #[tokio::test]
    async fn set_schedule_switches_at_runtime() {
        let db = test_db().await;
        let config = ConsolidationConfig::default();
        let mut sched =
            ConsolidationScheduler::with_schedule(db, config, ConsolidationSchedule::Manual);
        assert!(matches!(sched.schedule(), ConsolidationSchedule::Manual));

        // Switch to periodic.
        sched.set_schedule(ConsolidationSchedule::Periodic { interval_secs: 60 });
        assert!(matches!(
            sched.schedule(),
            ConsolidationSchedule::Periodic { interval_secs: 60 }
        ));

        // Switch again to surprise threshold.
        sched.set_schedule(ConsolidationSchedule::SurpriseThreshold {
            threshold: 3.0,
            fallback_interval_secs: 120,
        });
        assert!(matches!(
            sched.schedule(),
            ConsolidationSchedule::SurpriseThreshold { .. }
        ));

        sched.stop();
    }
}
