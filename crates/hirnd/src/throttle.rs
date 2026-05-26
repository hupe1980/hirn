use std::collections::HashMap;
use std::net::IpAddr;
use std::time::Instant;

use metrics::counter;
use parking_lot::Mutex;

use crate::config::{RateLimitBudget, ThrottleConfig};

const DEFAULT_MAX_RATE_LIMIT_ENTRIES: usize = 10_000;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum RateLimitClass {
    Auth,
    Read,
    Write,
    Admin,
}

impl RateLimitClass {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Auth => "auth",
            Self::Read => "read",
            Self::Write => "write",
            Self::Admin => "admin",
        }
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
enum RateLimitActor {
    Agent { realm: String, agent_id: String },
    PeerIp(IpAddr),
}

impl RateLimitActor {
    const fn kind(&self) -> &'static str {
        match self {
            Self::Agent { .. } => "agent",
            Self::PeerIp(_) => "peer_ip",
        }
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct RateLimitKey {
    class: RateLimitClass,
    actor: RateLimitActor,
}

struct RateLimitEntry {
    hits: Vec<Instant>,
}

pub struct RateLimiter {
    entries: Mutex<HashMap<RateLimitKey, RateLimitEntry>>,
    enabled: bool,
    max_entries: usize,
    auth_budget: RateLimitBudget,
    read_budget: RateLimitBudget,
    write_budget: RateLimitBudget,
    admin_budget: RateLimitBudget,
}

impl RateLimiter {
    pub fn new(max_requests: usize, window_secs: u64) -> Self {
        let budget = RateLimitBudget {
            max_requests,
            window_secs,
        };

        Self {
            entries: Mutex::new(HashMap::new()),
            enabled: true,
            max_entries: DEFAULT_MAX_RATE_LIMIT_ENTRIES,
            auth_budget: budget,
            read_budget: budget,
            write_budget: budget,
            admin_budget: budget,
        }
    }

    pub fn from_config(config: &ThrottleConfig) -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
            enabled: config.enabled,
            max_entries: config.max_entries,
            auth_budget: config.auth,
            read_budget: config.read,
            write_budget: config.write,
            admin_budget: config.admin,
        }
    }

    pub fn check_agent(&self, class: RateLimitClass, realm: &str, agent_id: &str) -> bool {
        self.check(
            class,
            RateLimitActor::Agent {
                realm: realm.to_owned(),
                agent_id: agent_id.to_owned(),
            },
        )
    }

    pub fn check_ip(&self, class: RateLimitClass, ip: IpAddr) -> bool {
        self.check(class, RateLimitActor::PeerIp(ip))
    }

    fn check(&self, class: RateLimitClass, actor: RateLimitActor) -> bool {
        if !self.enabled {
            return true;
        }

        let budget = self.budget(class);
        let now = Instant::now();
        let mut entries = self.entries.lock();

        if entries.len() > self.max_entries {
            entries.retain(|key, entry| {
                let entry_budget = self.budget(key.class);
                entry.hits.retain(|timestamp| {
                    now.duration_since(*timestamp).as_secs() < entry_budget.window_secs
                });
                !entry.hits.is_empty()
            });
        }

        let key = RateLimitKey { class, actor };
        let entry = entries
            .entry(key.clone())
            .or_insert_with(|| RateLimitEntry { hits: Vec::new() });
        entry
            .hits
            .retain(|timestamp| now.duration_since(*timestamp).as_secs() < budget.window_secs);
        if entry.hits.len() >= budget.max_requests {
            counter!(
                "hirnd_rate_limit_rejections_total",
                "class" => class.as_str(),
                "actor_kind" => key.actor.kind(),
            )
            .increment(1);
            return false;
        }

        entry.hits.push(now);
        true
    }

    fn budget(&self, class: RateLimitClass) -> RateLimitBudget {
        match class {
            RateLimitClass::Auth => self.auth_budget,
            RateLimitClass::Read => self.read_budget,
            RateLimitClass::Write => self.write_budget,
            RateLimitClass::Admin => self.admin_budget,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{RateLimitClass, RateLimiter};

    #[test]
    fn route_classes_are_independent() {
        let limiter = RateLimiter::new(1, 60);

        assert!(limiter.check_agent(RateLimitClass::Read, "default", "agent-a"));
        assert!(!limiter.check_agent(RateLimitClass::Read, "default", "agent-a"));
        assert!(limiter.check_agent(RateLimitClass::Write, "default", "agent-a"));
    }

    #[test]
    fn actors_are_independent() {
        let limiter = RateLimiter::new(1, 60);

        assert!(limiter.check_agent(RateLimitClass::Write, "default", "agent-a"));
        assert!(limiter.check_agent(RateLimitClass::Write, "default", "agent-b"));
    }
}
