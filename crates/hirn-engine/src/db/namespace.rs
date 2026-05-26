use arrow_array::Array;

use super::*;

pub(super) const NAMESPACE_DELETE_MUTATION_KIND: &str = "namespace_delete";
pub(super) const AGENT_REGISTER_MUTATION_KIND: &str = "agent_register";
pub(super) const AGENT_DEREGISTER_MUTATION_KIND: &str = "agent_deregister";

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct NamespaceDeleteEnvelope {
    namespace: Namespace,
    episodic_ids: Vec<MemoryId>,
    semantic_ids: Vec<MemoryId>,
    procedural_ids: Vec<MemoryId>,
    audit_entry_id: MemoryId,
    audit_timestamp: Timestamp,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct AgentRegisterEnvelope {
    agent: hirn_core::agent::AgentRecord,
    private_namespace: hirn_core::namespace::NamespaceRecord,
    audit_entry_id: MemoryId,
    audit_timestamp: Timestamp,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct AgentDeregisterEnvelope {
    agent_id: hirn_core::types::AgentId,
    private_namespace: Namespace,
    audit_entry_id: MemoryId,
    audit_timestamp: Timestamp,
}

fn namespace_sql_value(namespace: &Namespace) -> String {
    namespace.as_str().replace('\'', "''")
}

fn namespace_id_filter(namespace: &Namespace) -> String {
    format!("id = '{}'", namespace_sql_value(namespace))
}

fn namespace_column_filter(namespace: &Namespace) -> String {
    format!("namespace = '{}'", namespace_sql_value(namespace))
}

fn escaped_id_filter(id: &str) -> String {
    format!("id = '{}'", id.replace('\'', "''"))
}

fn encode_namespace_delete_envelope(payload: &NamespaceDeleteEnvelope) -> HirnResult<Vec<u8>> {
    serde_json::to_vec(payload).map_err(|error| {
        HirnError::storage(format!("namespace delete envelope serialize: {error}"))
    })
}

fn encode_agent_register_envelope(payload: &AgentRegisterEnvelope) -> HirnResult<Vec<u8>> {
    serde_json::to_vec(payload)
        .map_err(|error| HirnError::storage(format!("agent register envelope serialize: {error}")))
}

fn encode_agent_deregister_envelope(payload: &AgentDeregisterEnvelope) -> HirnResult<Vec<u8>> {
    serde_json::to_vec(payload).map_err(|error| {
        HirnError::storage(format!("agent deregister envelope serialize: {error}"))
    })
}

fn decode_namespace_delete_envelope(
    envelope: &hirn_storage::MutationEnvelopeRecord,
) -> HirnResult<NamespaceDeleteEnvelope> {
    serde_json::from_slice(&envelope.payload).map_err(|error| {
        HirnError::storage(format!("namespace delete envelope deserialize: {error}"))
    })
}

fn decode_agent_register_envelope(
    envelope: &hirn_storage::MutationEnvelopeRecord,
) -> HirnResult<AgentRegisterEnvelope> {
    serde_json::from_slice(&envelope.payload).map_err(|error| {
        HirnError::storage(format!("agent register envelope deserialize: {error}"))
    })
}

fn decode_agent_deregister_envelope(
    envelope: &hirn_storage::MutationEnvelopeRecord,
) -> HirnResult<AgentDeregisterEnvelope> {
    serde_json::from_slice(&envelope.payload).map_err(|error| {
        HirnError::storage(format!("agent deregister envelope deserialize: {error}"))
    })
}

fn build_namespace_delete_envelope(
    namespace: Namespace,
    mut episodic_ids: Vec<MemoryId>,
    mut semantic_ids: Vec<MemoryId>,
    mut procedural_ids: Vec<MemoryId>,
) -> HirnResult<hirn_storage::MutationEnvelopeRecord> {
    episodic_ids.sort_unstable();
    episodic_ids.dedup();
    semantic_ids.sort_unstable();
    semantic_ids.dedup();
    procedural_ids.sort_unstable();
    procedural_ids.dedup();

    let audit_entry_id = MemoryId::new();
    let payload = NamespaceDeleteEnvelope {
        namespace,
        episodic_ids,
        semantic_ids,
        procedural_ids,
        audit_entry_id,
        audit_timestamp: Timestamp::now(),
    };
    let payload = encode_namespace_delete_envelope(&payload)?;

    Ok(hirn_storage::MutationEnvelopeRecord::pending(
        format!("namespace-delete:{namespace}:{audit_entry_id}"),
        NAMESPACE_DELETE_MUTATION_KIND,
        payload,
    ))
}

fn build_agent_register_envelope(
    agent: hirn_core::agent::AgentRecord,
    private_namespace: hirn_core::namespace::NamespaceRecord,
) -> HirnResult<hirn_storage::MutationEnvelopeRecord> {
    let agent_id = agent.id;
    let audit_entry_id = MemoryId::new();
    let payload = AgentRegisterEnvelope {
        agent,
        private_namespace,
        audit_entry_id,
        audit_timestamp: Timestamp::now(),
    };
    let payload = encode_agent_register_envelope(&payload)?;

    Ok(hirn_storage::MutationEnvelopeRecord::pending(
        format!("agent-register:{agent_id}:{audit_entry_id}"),
        AGENT_REGISTER_MUTATION_KIND,
        payload,
    ))
}

fn build_agent_deregister_envelope(
    agent_id: hirn_core::types::AgentId,
) -> HirnResult<hirn_storage::MutationEnvelopeRecord> {
    let audit_entry_id = MemoryId::new();
    let payload = AgentDeregisterEnvelope {
        private_namespace: Namespace::private_for(&agent_id),
        agent_id,
        audit_entry_id,
        audit_timestamp: Timestamp::now(),
    };
    let payload = encode_agent_deregister_envelope(&payload)?;

    Ok(hirn_storage::MutationEnvelopeRecord::pending(
        format!("agent-deregister:{agent_id}:{audit_entry_id}"),
        AGENT_DEREGISTER_MUTATION_KIND,
        payload,
    ))
}

impl HirnDB {
    // ── Event Subscription ──────────────────────────────────────────────

    /// Subscribe to real-time memory events.
    ///
    /// Returns a [`tokio::sync::broadcast::Receiver<MemoryEvent>`] that yields
    /// events whenever the database state changes (create, archive,
    /// consolidate, etc.).  The broadcast ring buffer is lock-free; lagging
    /// subscribers receive a
    /// [`tokio::sync::broadcast::error::RecvError::Lagged`] error and skip
    /// missed events rather than blocking the write path.
    pub fn subscribe(&self) -> tokio::sync::broadcast::Receiver<crate::event::MemoryEvent> {
        self.event_runtime().subscribe()
    }

    /// Broadcast an event with explicit realm/namespace/agent context.
    pub(crate) async fn emit_in_realm(
        &self,
        realm: &str,
        namespace: &str,
        agent_id: &str,
        event: MemoryEvent,
    ) {
        self.event_runtime()
            .emit(realm, namespace, agent_id, event)
            .await;
    }

    pub(crate) async fn emit_in_realm_checked(
        &self,
        realm: &str,
        namespace: &str,
        agent_id: &str,
        event: MemoryEvent,
    ) -> HirnResult<()> {
        self.event_runtime()
            .emit_checked(realm, namespace, agent_id, event)
            .await
    }

    /// Broadcast an event using the default realm with explicit namespace/agent context.
    pub(crate) async fn emit_scoped(&self, namespace: &str, agent_id: &str, event: MemoryEvent) {
        self.emit_in_realm(&self.config.default_realm, namespace, agent_id, event)
            .await;
    }

    pub(crate) async fn emit_scoped_checked(
        &self,
        namespace: &str,
        agent_id: &str,
        event: MemoryEvent,
    ) -> HirnResult<()> {
        self.emit_in_realm_checked(&self.config.default_realm, namespace, agent_id, event)
            .await
    }

    /// Broadcast an event to all live subscribers and, when event sourcing
    /// is enabled, append it to the durable event log.
    pub(crate) async fn emit(&self, event: MemoryEvent) {
        self.emit_in_realm(&self.config.default_realm, "shared", "", event)
            .await;
    }

    // ── Namespace Management ────────────────────────────────────────────

    /// Create a new namespace.
    pub(crate) async fn create_namespace(
        &self,
        name: &str,
        kind: hirn_core::types::NamespaceKind,
        members: Vec<hirn_core::types::AgentId>,
    ) -> HirnResult<()> {
        let ns = Namespace::new(name)?;

        // Check for existing namespace.
        let filter = namespace_id_filter(&ns);
        let count = self
            .storage_runtime
            .count(
                hirn_storage::datasets::namespace::DATASET_NAME,
                Some(&filter),
            )
            .await
            .map_err(|e| HirnError::storage(e))?;
        if count > 0 {
            return Err(HirnError::AlreadyExists(format!(
                "namespace '{name}' already exists"
            )));
        }

        let rec = hirn_core::namespace::NamespaceRecord {
            namespace: ns,
            kind,
            created_at: Timestamp::now(),
            member_agents: members,
        };

        let batch = hirn_storage::datasets::namespace::to_batch(std::slice::from_ref(&rec))
            .map_err(|e| HirnError::storage(e))?;
        self.storage_runtime
            .append(hirn_storage::datasets::namespace::DATASET_NAME, batch)
            .await
            .map_err(|e| HirnError::storage(e))?;

        self.namespace_runtime.invalidate_namespaces();

        self.append_audit(
            None,
            hirn_core::audit::AuditAction::NamespaceCreated {
                namespace: name.to_string(),
            },
        )
        .await?;

        Ok(())
    }

    /// List all namespaces.
    pub(crate) async fn list_namespaces(
        &self,
    ) -> HirnResult<Vec<hirn_core::namespace::NamespaceRecord>> {
        if let Some(cached) = self.namespace_runtime.cached_namespaces() {
            return Ok(cached.as_ref().clone());
        }

        let opts = hirn_storage::store::ScanOptions::default();
        let batches = self
            .storage_runtime
            .scan(hirn_storage::datasets::namespace::DATASET_NAME, opts)
            .await
            .map_err(|e| HirnError::storage(e))?;

        let mut result = Vec::new();
        for batch in &batches {
            let recs = hirn_storage::datasets::namespace::from_batch(batch)
                .map_err(|e| HirnError::storage(e))?;
            result.extend(recs);
        }

        self.namespace_runtime.cache_namespaces(result.clone());

        Ok(result)
    }

    /// Get a namespace record by name.
    pub(crate) async fn get_namespace(
        &self,
        name: &str,
    ) -> HirnResult<hirn_core::namespace::NamespaceRecord> {
        if let Some(cached) = self.namespace_runtime.cached_namespaces() {
            if let Some(rec) = cached.iter().find(|rec| rec.namespace.as_str() == name) {
                return Ok(rec.clone());
            }
        }

        let ns = Namespace::new(name)?;
        let filter = namespace_id_filter(&ns);
        let opts = hirn_storage::store::ScanOptions {
            filter: Some(filter),
            ..Default::default()
        };
        let batches = self
            .storage_runtime
            .scan(hirn_storage::datasets::namespace::DATASET_NAME, opts)
            .await
            .map_err(|e| HirnError::storage(e))?;

        for batch in &batches {
            let recs = hirn_storage::datasets::namespace::from_batch(batch)
                .map_err(|e| HirnError::storage(e))?;
            if let Some(rec) = recs.into_iter().next() {
                return Ok(rec);
            }
        }
        Err(HirnError::NotFound(format!("namespace '{name}'")))
    }

    /// Delete a namespace and all its memories.
    pub(crate) async fn delete_namespace(&self, name: &str) -> HirnResult<()> {
        let ns = Namespace::new(name)?;
        if ns == Namespace::shared() {
            return Err(HirnError::InvalidInput(
                "cannot delete the shared namespace".into(),
            ));
        }

        // Verify namespace exists.
        self.get_namespace(name).await?;

        let ep_ids = self.list_episodic_ids_in_namespace(&ns).await?;
        let sem_ids = self.list_semantic_ids_in_namespace(&ns).await?;
        let proc_ids = self.list_procedural_ids_in_namespace(&ns).await?;
        let envelope = build_namespace_delete_envelope(ns, ep_ids, sem_ids, proc_ids)?;
        hirn_storage::append_mutation_envelope(self.storage_backend(), &envelope)
            .await
            .map_err(HirnError::storage)?;

        let payload = decode_namespace_delete_envelope(&envelope)?;
        self.apply_namespace_delete_plan(&payload).await?;

        if let Err(error) = hirn_storage::update_mutation_envelope_state(
            self.storage_backend(),
            &envelope.id,
            hirn_storage::MutationEnvelopeState::Applied,
            None,
        )
        .await
        {
            tracing::warn!(
                namespace = %ns,
                envelope_id = %envelope.id,
                error = %error,
                "namespace delete mutation envelope finalize failed; recovery will retry"
            );
        }

        Ok(())
    }

    pub(crate) async fn reconcile_pending_namespace_delete_mutations(&self) -> HirnResult<usize> {
        let envelopes = hirn_storage::list_pending_mutation_envelopes(
            self.storage_backend(),
            Some(NAMESPACE_DELETE_MUTATION_KIND),
        )
        .await
        .map_err(HirnError::storage)?;
        let mut reconciled = 0usize;

        for envelope in envelopes {
            match self
                .reconcile_single_pending_namespace_delete_mutation(&envelope)
                .await
            {
                Ok(true) => reconciled += 1,
                Ok(false) => {}
                Err(error) => {
                    hirn_storage::update_mutation_envelope_state(
                        self.storage_backend(),
                        &envelope.id,
                        hirn_storage::MutationEnvelopeState::Failed,
                        Some(error.to_string()),
                    )
                    .await
                    .map_err(HirnError::storage)?;
                }
            }
        }

        Ok(reconciled)
    }

    pub(crate) async fn reconcile_pending_agent_register_mutations(&self) -> HirnResult<usize> {
        let envelopes = hirn_storage::list_pending_mutation_envelopes(
            self.storage_backend(),
            Some(AGENT_REGISTER_MUTATION_KIND),
        )
        .await
        .map_err(HirnError::storage)?;
        let mut reconciled = 0usize;

        for envelope in envelopes {
            match self
                .reconcile_single_pending_agent_register_mutation(&envelope)
                .await
            {
                Ok(true) => reconciled += 1,
                Ok(false) => {}
                Err(error) => {
                    hirn_storage::update_mutation_envelope_state(
                        self.storage_backend(),
                        &envelope.id,
                        hirn_storage::MutationEnvelopeState::Failed,
                        Some(error.to_string()),
                    )
                    .await
                    .map_err(HirnError::storage)?;
                }
            }
        }

        Ok(reconciled)
    }

    pub(crate) async fn reconcile_pending_agent_deregister_mutations(&self) -> HirnResult<usize> {
        let envelopes = hirn_storage::list_pending_mutation_envelopes(
            self.storage_backend(),
            Some(AGENT_DEREGISTER_MUTATION_KIND),
        )
        .await
        .map_err(HirnError::storage)?;
        let mut reconciled = 0usize;

        for envelope in envelopes {
            match self
                .reconcile_single_pending_agent_deregister_mutation(&envelope)
                .await
            {
                Ok(true) => reconciled += 1,
                Ok(false) => {}
                Err(error) => {
                    hirn_storage::update_mutation_envelope_state(
                        self.storage_backend(),
                        &envelope.id,
                        hirn_storage::MutationEnvelopeState::Failed,
                        Some(error.to_string()),
                    )
                    .await
                    .map_err(HirnError::storage)?;
                }
            }
        }

        Ok(reconciled)
    }

    async fn reconcile_single_pending_agent_register_mutation(
        &self,
        envelope: &hirn_storage::MutationEnvelopeRecord,
    ) -> HirnResult<bool> {
        let payload = decode_agent_register_envelope(envelope)?;
        self.apply_agent_register_plan(&payload).await?;
        hirn_storage::update_mutation_envelope_state(
            self.storage_backend(),
            &envelope.id,
            hirn_storage::MutationEnvelopeState::Applied,
            None,
        )
        .await
        .map_err(HirnError::storage)?;
        Ok(true)
    }

    async fn reconcile_single_pending_agent_deregister_mutation(
        &self,
        envelope: &hirn_storage::MutationEnvelopeRecord,
    ) -> HirnResult<bool> {
        let payload = decode_agent_deregister_envelope(envelope)?;
        self.apply_agent_deregister_plan(&payload).await?;
        hirn_storage::update_mutation_envelope_state(
            self.storage_backend(),
            &envelope.id,
            hirn_storage::MutationEnvelopeState::Applied,
            None,
        )
        .await
        .map_err(HirnError::storage)?;
        Ok(true)
    }

    async fn reconcile_single_pending_namespace_delete_mutation(
        &self,
        envelope: &hirn_storage::MutationEnvelopeRecord,
    ) -> HirnResult<bool> {
        let payload = decode_namespace_delete_envelope(envelope)?;
        self.apply_namespace_delete_plan(&payload).await?;
        hirn_storage::update_mutation_envelope_state(
            self.storage_backend(),
            &envelope.id,
            hirn_storage::MutationEnvelopeState::Applied,
            None,
        )
        .await
        .map_err(HirnError::storage)?;
        Ok(true)
    }

    async fn apply_namespace_delete_plan(
        &self,
        payload: &NamespaceDeleteEnvelope,
    ) -> HirnResult<()> {
        for id in &payload.episodic_ids {
            self.delete_episode_if_present(*id).await?;
        }
        for id in &payload.semantic_ids {
            self.purge_semantic_if_present(*id).await?;
        }
        for id in &payload.procedural_ids {
            self.delete_procedural_if_present(*id).await?;
        }

        // Remove the namespace record.
        let predicate = namespace_id_filter(&payload.namespace);
        self.storage_runtime
            .delete(hirn_storage::datasets::namespace::DATASET_NAME, &predicate)
            .await
            .map_err(|e| HirnError::storage(e))?;

        self.namespace_runtime.invalidate_namespaces();

        self.append_namespace_delete_audit_once(payload).await?;

        Ok(())
    }

    async fn append_namespace_delete_audit_once(
        &self,
        payload: &NamespaceDeleteEnvelope,
    ) -> HirnResult<()> {
        let audit_filter = escaped_id_filter(&payload.audit_entry_id.to_string());
        let existing = self
            .storage_runtime
            .count(
                hirn_storage::datasets::audit::DATASET_NAME,
                Some(&audit_filter),
            )
            .await
            .map_err(HirnError::storage)?;
        if existing > 0 {
            return Ok(());
        }

        let entry = hirn_core::audit::AuditEntry {
            id: payload.audit_entry_id,
            timestamp: payload.audit_timestamp,
            actor: None,
            action: hirn_core::audit::AuditAction::NamespaceDeleted {
                namespace: payload.namespace.to_string(),
            },
        };
        let batch = hirn_storage::datasets::audit::to_batch(std::slice::from_ref(&entry))
            .map_err(HirnError::storage)?;
        self.storage_runtime
            .append(hirn_storage::datasets::audit::DATASET_NAME, batch)
            .await
            .map_err(HirnError::storage)?;
        Ok(())
    }

    async fn apply_agent_register_plan(&self, payload: &AgentRegisterEnvelope) -> HirnResult<()> {
        let agent_batch =
            hirn_storage::datasets::agent::to_batch(std::slice::from_ref(&payload.agent))
                .map_err(HirnError::storage)?;
        self.storage_backend()
            .merge_insert(
                hirn_storage::datasets::agent::DATASET_NAME,
                &["id"],
                agent_batch,
            )
            .await
            .map_err(HirnError::storage)?;

        let namespace_batch = hirn_storage::datasets::namespace::to_batch(std::slice::from_ref(
            &payload.private_namespace,
        ))
        .map_err(HirnError::storage)?;
        self.storage_backend()
            .merge_insert(
                hirn_storage::datasets::namespace::DATASET_NAME,
                &["id"],
                namespace_batch,
            )
            .await
            .map_err(HirnError::storage)?;

        self.namespace_runtime.cache_agent(payload.agent.clone());
        self.namespace_runtime.invalidate_namespaces();
        self.append_agent_register_audit_once(payload).await?;
        Ok(())
    }

    async fn apply_agent_deregister_plan(
        &self,
        payload: &AgentDeregisterEnvelope,
    ) -> HirnResult<()> {
        match self
            .delete_namespace(payload.private_namespace.as_str())
            .await
        {
            Ok(()) | Err(HirnError::NotFound(_)) => {}
            Err(error) => return Err(error),
        }

        let predicate = escaped_id_filter(payload.agent_id.as_str());
        self.storage_runtime
            .delete(hirn_storage::datasets::agent::DATASET_NAME, &predicate)
            .await
            .map_err(HirnError::storage)?;

        self.namespace_runtime.evict_agent(&payload.agent_id);
        self.namespace_runtime.invalidate_namespaces();
        self.append_agent_deregister_audit_once(payload).await?;
        Ok(())
    }

    async fn append_agent_register_audit_once(
        &self,
        payload: &AgentRegisterEnvelope,
    ) -> HirnResult<()> {
        let audit_filter = escaped_id_filter(&payload.audit_entry_id.to_string());
        let existing = self
            .storage_runtime
            .count(
                hirn_storage::datasets::audit::DATASET_NAME,
                Some(&audit_filter),
            )
            .await
            .map_err(HirnError::storage)?;
        if existing > 0 {
            return Ok(());
        }

        let entry = hirn_core::audit::AuditEntry {
            id: payload.audit_entry_id,
            timestamp: payload.audit_timestamp,
            actor: None,
            action: hirn_core::audit::AuditAction::AgentRegistered {
                agent_id: payload.agent.id,
            },
        };
        let batch = hirn_storage::datasets::audit::to_batch(std::slice::from_ref(&entry))
            .map_err(HirnError::storage)?;
        self.storage_runtime
            .append(hirn_storage::datasets::audit::DATASET_NAME, batch)
            .await
            .map_err(HirnError::storage)?;
        Ok(())
    }

    async fn append_agent_deregister_audit_once(
        &self,
        payload: &AgentDeregisterEnvelope,
    ) -> HirnResult<()> {
        let audit_filter = escaped_id_filter(&payload.audit_entry_id.to_string());
        let existing = self
            .storage_runtime
            .count(
                hirn_storage::datasets::audit::DATASET_NAME,
                Some(&audit_filter),
            )
            .await
            .map_err(HirnError::storage)?;
        if existing > 0 {
            return Ok(());
        }

        let entry = hirn_core::audit::AuditEntry {
            id: payload.audit_entry_id,
            timestamp: payload.audit_timestamp,
            actor: None,
            action: hirn_core::audit::AuditAction::AgentDeregistered {
                agent_id: payload.agent_id,
            },
        };
        let batch = hirn_storage::datasets::audit::to_batch(std::slice::from_ref(&entry))
            .map_err(HirnError::storage)?;
        self.storage_runtime
            .append(hirn_storage::datasets::audit::DATASET_NAME, batch)
            .await
            .map_err(HirnError::storage)?;
        Ok(())
    }

    async fn delete_episode_if_present(&self, id: MemoryId) -> HirnResult<()> {
        match self.read_episodic_record(id).await {
            Ok(_) => {}
            Err(HirnError::NotFound(_)) => return Ok(()),
            Err(error) => return Err(error),
        }
        match self.delete_episode(id).await {
            Ok(()) | Err(HirnError::NotFound(_)) => Ok(()),
            Err(error) => Err(error),
        }
    }

    async fn purge_semantic_if_present(&self, id: MemoryId) -> HirnResult<()> {
        match self.read_semantic_record(id).await {
            Ok(_) => {}
            Err(HirnError::NotFound(_)) => return Ok(()),
            Err(error) => return Err(error),
        }
        match self.purge_semantic(id).await {
            Ok(()) | Err(HirnError::NotFound(_)) => Ok(()),
            Err(error) => Err(error),
        }
    }

    async fn delete_procedural_if_present(&self, id: MemoryId) -> HirnResult<()> {
        match self.get_procedural(id).await {
            Ok(_) => {}
            Err(HirnError::NotFound(_)) => return Ok(()),
            Err(error) => return Err(error),
        }
        match self.delete_procedural(id).await {
            Ok(()) | Err(HirnError::NotFound(_)) => Ok(()),
            Err(error) => Err(error),
        }
    }

    /// List episodic record IDs in a namespace.
    pub(crate) async fn list_episodic_ids_in_namespace(
        &self,
        ns: &Namespace,
    ) -> HirnResult<Vec<MemoryId>> {
        let filter = namespace_column_filter(ns);
        let opts = hirn_storage::store::ScanOptions {
            filter: Some(filter),
            columns: Some(vec!["id".to_string()]),
            ..Default::default()
        };
        let batches = self
            .storage_runtime
            .scan(hirn_storage::datasets::episodic::DATASET_NAME, opts)
            .await
            .map_err(|e| HirnError::storage(e))?;

        let mut ids = Vec::new();
        for batch in &batches {
            let id_col = batch
                .column_by_name("id")
                .and_then(|c| c.as_any().downcast_ref::<arrow_array::StringArray>());
            if let Some(col) = id_col {
                for i in 0..col.len() {
                    let id = MemoryId::parse(col.value(i))
                        .map_err(|e| HirnError::InvalidInput(e.to_string()))?;
                    ids.push(id);
                }
            }
        }
        Ok(ids)
    }

    /// List semantic record IDs in a namespace.
    pub(crate) async fn list_semantic_ids_in_namespace(
        &self,
        ns: &Namespace,
    ) -> HirnResult<Vec<MemoryId>> {
        let filter = namespace_column_filter(ns);
        let opts = hirn_storage::store::ScanOptions {
            filter: Some(filter),
            columns: Some(vec!["id".to_string()]),
            ..Default::default()
        };
        let batches = self
            .storage_runtime
            .scan(hirn_storage::datasets::semantic::DATASET_NAME, opts)
            .await
            .map_err(|e| HirnError::storage(e))?;

        let mut ids = Vec::new();
        for batch in &batches {
            let id_col = batch
                .column_by_name("id")
                .and_then(|c| c.as_any().downcast_ref::<arrow_array::StringArray>());
            if let Some(col) = id_col {
                for i in 0..col.len() {
                    let id = MemoryId::parse(col.value(i))
                        .map_err(|e| HirnError::InvalidInput(e.to_string()))?;
                    ids.push(id);
                }
            }
        }
        Ok(ids)
    }

    /// List procedural record IDs in a namespace.
    pub(crate) async fn list_procedural_ids_in_namespace(
        &self,
        ns: &Namespace,
    ) -> HirnResult<Vec<MemoryId>> {
        let filter = namespace_column_filter(ns);
        let opts = hirn_storage::store::ScanOptions {
            filter: Some(filter),
            columns: Some(vec!["id".to_string()]),
            ..Default::default()
        };
        let batches = self
            .storage_runtime
            .scan(hirn_storage::datasets::procedural::DATASET_NAME, opts)
            .await
            .map_err(|e| HirnError::storage(e))?;

        let mut ids = Vec::new();
        for batch in &batches {
            let id_col = batch
                .column_by_name("id")
                .and_then(|c| c.as_any().downcast_ref::<arrow_array::StringArray>());
            if let Some(col) = id_col {
                for i in 0..col.len() {
                    let id = MemoryId::parse(col.value(i))
                        .map_err(|e| HirnError::InvalidInput(e.to_string()))?;
                    ids.push(id);
                }
            }
        }
        Ok(ids)
    }

    // ── Agent Registration ──────────────────────────────────────────────

    /// Register a new agent. Creates private namespace `private:{agent_id}`.
    pub async fn register_agent(
        &self,
        agent_id: &hirn_core::types::AgentId,
        display_name: impl Into<String>,
    ) -> HirnResult<()> {
        if self.namespace_runtime.cached_agent(agent_id).is_some() {
            return Err(HirnError::AlreadyExists(format!(
                "agent '{}' already registered",
                agent_id
            )));
        }

        // Check for existing agent.
        let filter = escaped_id_filter(agent_id.as_str());
        let count = self
            .storage_runtime
            .count(hirn_storage::datasets::agent::DATASET_NAME, Some(&filter))
            .await
            .map_err(|e| HirnError::storage(e))?;
        if count > 0 {
            return Err(HirnError::AlreadyExists(format!(
                "agent '{}' already registered",
                agent_id
            )));
        }

        let rec = hirn_core::agent::AgentRecord::new(agent_id.clone(), display_name);
        let ns_rec = hirn_core::namespace::NamespaceRecord::private_for(agent_id);
        let envelope = build_agent_register_envelope(rec, ns_rec)?;
        hirn_storage::append_mutation_envelope(self.storage_backend(), &envelope)
            .await
            .map_err(HirnError::storage)?;

        let payload = decode_agent_register_envelope(&envelope)?;
        self.apply_agent_register_plan(&payload).await?;

        if let Err(error) = hirn_storage::update_mutation_envelope_state(
            self.storage_backend(),
            &envelope.id,
            hirn_storage::MutationEnvelopeState::Applied,
            None,
        )
        .await
        {
            tracing::warn!(
                agent_id = %agent_id,
                envelope_id = %envelope.id,
                error = %error,
                "agent register mutation envelope finalize failed; recovery will retry"
            );
        }

        Ok(())
    }

    /// List all registered agents.
    pub async fn list_agents(&self) -> HirnResult<Vec<hirn_core::agent::AgentRecord>> {
        let opts = hirn_storage::store::ScanOptions::default();
        let batches = self
            .storage_runtime
            .scan(hirn_storage::datasets::agent::DATASET_NAME, opts)
            .await
            .map_err(|e| HirnError::storage(e))?;

        let mut result = Vec::new();
        for batch in &batches {
            let recs = hirn_storage::datasets::agent::from_batch(batch)
                .map_err(|e| HirnError::storage(e))?;
            result.extend(recs);
        }
        Ok(result)
    }

    /// Get a registered agent.
    pub async fn get_agent(
        &self,
        agent_id: &hirn_core::types::AgentId,
    ) -> HirnResult<hirn_core::agent::AgentRecord> {
        if let Some(agent) = self.namespace_runtime.cached_agent(agent_id) {
            return Ok(agent);
        }

        let filter = escaped_id_filter(agent_id.as_str());
        let opts = hirn_storage::store::ScanOptions {
            filter: Some(filter),
            ..Default::default()
        };
        let batches = self
            .storage_runtime
            .scan(hirn_storage::datasets::agent::DATASET_NAME, opts)
            .await
            .map_err(|e| HirnError::storage(e))?;

        for batch in &batches {
            let recs = hirn_storage::datasets::agent::from_batch(batch)
                .map_err(|e| HirnError::storage(e))?;
            if let Some(rec) = recs.into_iter().next() {
                self.namespace_runtime.cache_agent(rec.clone());
                return Ok(rec);
            }
        }
        Err(HirnError::NotFound(format!("agent '{agent_id}'")))
    }

    /// Update a registered agent record.
    pub async fn update_agent(&self, agent: &hirn_core::agent::AgentRecord) -> HirnResult<()> {
        let filter = escaped_id_filter(agent.id.as_str());
        let count = self
            .storage_runtime
            .count(hirn_storage::datasets::agent::DATASET_NAME, Some(&filter))
            .await
            .map_err(|e| HirnError::storage(e))?;
        if count == 0 {
            return Err(HirnError::NotFound(format!("agent '{}'", agent.id)));
        }
        let batch = hirn_storage::datasets::agent::to_batch(std::slice::from_ref(agent))
            .map_err(|e| HirnError::storage(e))?;
        self.storage_backend()
            .merge_insert(hirn_storage::datasets::agent::DATASET_NAME, &["id"], batch)
            .await
            .map_err(|e| HirnError::storage(e))?;

        self.namespace_runtime.cache_agent(agent.clone());
        Ok(())
    }

    /// Deregister an agent and delete its private namespace.
    pub async fn deregister_agent(&self, agent_id: &hirn_core::types::AgentId) -> HirnResult<()> {
        // Verify agent exists.
        self.get_agent(agent_id).await?;

        let envelope = build_agent_deregister_envelope(*agent_id)?;
        hirn_storage::append_mutation_envelope(self.storage_backend(), &envelope)
            .await
            .map_err(HirnError::storage)?;

        let payload = decode_agent_deregister_envelope(&envelope)?;
        self.apply_agent_deregister_plan(&payload).await?;

        if let Err(error) = hirn_storage::update_mutation_envelope_state(
            self.storage_backend(),
            &envelope.id,
            hirn_storage::MutationEnvelopeState::Applied,
            None,
        )
        .await
        {
            tracing::warn!(
                agent_id = %agent_id,
                envelope_id = %envelope.id,
                error = %error,
                "agent deregister mutation envelope finalize failed; recovery will retry"
            );
        }

        Ok(())
    }

    // ── Team Namespace Management ───────────────────────────────────────

    /// Create a team namespace with the given agent members.
    pub async fn create_team_namespace(
        &self,
        name: &str,
        agent_ids: Vec<hirn_core::types::AgentId>,
    ) -> HirnResult<()> {
        self.create_namespace(name, hirn_core::types::NamespaceKind::Team, agent_ids)
            .await
    }

    /// Add an agent to a team namespace.
    pub async fn add_agent_to_team(
        &self,
        agent_id: &hirn_core::types::AgentId,
        team_name: &str,
    ) -> HirnResult<()> {
        let mut ns_rec = self.get_namespace(team_name).await?;
        if ns_rec.kind != hirn_core::types::NamespaceKind::Team {
            return Err(HirnError::InvalidInput(format!(
                "'{team_name}' is not a team namespace"
            )));
        }
        if ns_rec.member_agents.contains(agent_id) {
            return Ok(()); // already a member
        }
        ns_rec.member_agents.push(agent_id.clone());
        self.update_namespace_record(&ns_rec).await?;

        self.append_audit(
            None,
            hirn_core::audit::AuditAction::AgentAddedToTeam {
                agent_id: agent_id.clone(),
                team: team_name.to_string(),
            },
        )
        .await?;
        Ok(())
    }

    /// Remove an agent from a team namespace.
    pub async fn remove_agent_from_team(
        &self,
        agent_id: &hirn_core::types::AgentId,
        team_name: &str,
    ) -> HirnResult<()> {
        let mut ns_rec = self.get_namespace(team_name).await?;
        if ns_rec.kind != hirn_core::types::NamespaceKind::Team {
            return Err(HirnError::InvalidInput(format!(
                "'{team_name}' is not a team namespace"
            )));
        }
        ns_rec.member_agents.retain(|a| a != agent_id);
        self.update_namespace_record(&ns_rec).await?;

        self.append_audit(
            None,
            hirn_core::audit::AuditAction::AgentRemovedFromTeam {
                agent_id: agent_id.clone(),
                team: team_name.to_string(),
            },
        )
        .await?;
        Ok(())
    }

    /// Update a namespace record in storage.
    async fn update_namespace_record(
        &self,
        rec: &hirn_core::namespace::NamespaceRecord,
    ) -> HirnResult<()> {
        let batch = hirn_storage::datasets::namespace::to_batch(std::slice::from_ref(rec))
            .map_err(|e| HirnError::storage(e))?;
        self.storage_backend()
            .merge_insert(
                hirn_storage::datasets::namespace::DATASET_NAME,
                &["id"],
                batch,
            )
            .await
            .map_err(|e| HirnError::storage(e))?;

        self.namespace_runtime.invalidate_namespaces();
        Ok(())
    }

    // ── Audit Trail ─────────────────────────────────────────────────────

    /// Append an entry to the audit log.
    pub(crate) async fn append_audit(
        &self,
        actor: Option<hirn_core::types::AgentId>,
        action: hirn_core::audit::AuditAction,
    ) -> HirnResult<()> {
        self.policy_runtime().append_audit(actor, action).await
    }

    /// Query the audit log, optionally filtering by time range.
    pub(crate) async fn audit_log(
        &self,
        after: Option<&Timestamp>,
        before: Option<&Timestamp>,
    ) -> HirnResult<Vec<hirn_core::audit::AuditEntry>> {
        self.policy_runtime().audit_log(after, before).await
    }

    // ── Agent Context ───────────────────────────────────────────────────

    /// Register an agent if not already registered. Returns `Ok(())` in either case.
    pub async fn ensure_agent(&self, agent_id: &hirn_core::types::AgentId) -> HirnResult<()> {
        if self.namespace_runtime.cached_agent(agent_id).is_some() {
            return Ok(());
        }

        if self.get_agent(agent_id).await.is_ok() {
            return Ok(());
        }

        match self.register_agent(agent_id, agent_id.as_str()).await {
            Ok(()) | Err(HirnError::AlreadyExists(_)) => Ok(()),
            Err(e) => Err(e),
        }
    }

    /// Create an agent-scoped context for namespace-isolated operations.
    pub async fn as_agent(
        &self,
        agent_id: &hirn_core::types::AgentId,
    ) -> HirnResult<crate::agent_context::AgentContext<'_>> {
        let private_namespace = Namespace::private_for(agent_id);

        if let Some(accessible) = self
            .namespace_runtime
            .cached_accessible_namespaces(agent_id)
        {
            let mut accessible = accessible;
            if !accessible.contains(&private_namespace) {
                accessible.push(private_namespace);
            }
            return Ok(crate::agent_context::AgentContext::new(
                self,
                agent_id.clone(),
                accessible,
            ));
        }

        // Verify the agent is registered.
        self.get_agent(agent_id).await?;

        // Collect namespaces accessible to this agent.
        let namespaces = self.list_namespaces().await?;
        let mut accessible: Vec<Namespace> = namespaces
            .iter()
            .filter(|ns| ns.agent_has_access(agent_id))
            .map(|ns| ns.namespace.clone())
            .collect();
        if !accessible.contains(&private_namespace) {
            accessible.push(private_namespace);
        }

        self.namespace_runtime
            .cache_accessible_namespaces(agent_id.clone(), accessible.clone());

        Ok(crate::agent_context::AgentContext::new(
            self,
            agent_id.clone(),
            accessible,
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};

    use arrow_array::RecordBatch;
    use async_trait::async_trait;
    use datafusion::catalog::TableProvider;
    use hirn_core::types::{AgentId, EventType, NamespaceKind};
    use hirn_storage::memory_store::MemoryStore;
    use hirn_storage::store::{
        ColumnTransform, CompactOptions, CompactResult, DatasetInfo, FtsSearchOptions,
        HybridSearchOptions, IndexConfig, MultivectorSearchOptions, RecordBatchStream, ScanOptions,
        VectorSearchOptions, VersionTag,
    };
    use hirn_storage::{HirnDb, HirnDbConfig, HirnDbError, PhysicalStore};

    use super::*;

    struct FaultInjectingNamespaceStore {
        inner: MemoryStore,
        fail_agent_writes: AtomicBool,
        fail_agent_deletes: AtomicBool,
        fail_namespace_writes: AtomicBool,
    }

    impl FaultInjectingNamespaceStore {
        fn new() -> Self {
            Self {
                inner: MemoryStore::new(),
                fail_agent_writes: AtomicBool::new(false),
                fail_agent_deletes: AtomicBool::new(false),
                fail_namespace_writes: AtomicBool::new(false),
            }
        }

        fn fail_agent_writes(&self) {
            self.fail_agent_writes.store(true, AtomicOrdering::Release);
        }

        fn allow_agent_writes(&self) {
            self.fail_agent_writes.store(false, AtomicOrdering::Release);
        }

        fn fail_agent_deletes(&self) {
            self.fail_agent_deletes.store(true, AtomicOrdering::Release);
        }

        fn allow_agent_deletes(&self) {
            self.fail_agent_deletes
                .store(false, AtomicOrdering::Release);
        }

        fn fail_namespace_writes(&self) {
            self.fail_namespace_writes
                .store(true, AtomicOrdering::Release);
        }

        fn allow_namespace_writes(&self) {
            self.fail_namespace_writes
                .store(false, AtomicOrdering::Release);
        }
    }

    #[async_trait]
    impl PhysicalStore for FaultInjectingNamespaceStore {
        async fn append(&self, dataset: &str, batch: RecordBatch) -> Result<(), HirnDbError> {
            if dataset == hirn_storage::datasets::agent::DATASET_NAME
                && self.fail_agent_writes.load(AtomicOrdering::Acquire)
            {
                return Err(HirnDbError::Unsupported(
                    "simulated agent append failure".to_string(),
                ));
            }
            if dataset == hirn_storage::datasets::namespace::DATASET_NAME
                && self.fail_namespace_writes.load(AtomicOrdering::Acquire)
            {
                return Err(HirnDbError::Unsupported(
                    "simulated namespace append failure".to_string(),
                ));
            }
            self.inner.append(dataset, batch).await
        }

        async fn append_batches(
            &self,
            dataset: &str,
            batches: Vec<RecordBatch>,
        ) -> Result<(), HirnDbError> {
            self.inner.append_batches(dataset, batches).await
        }

        async fn scan(
            &self,
            dataset: &str,
            opts: ScanOptions,
        ) -> Result<Vec<RecordBatch>, HirnDbError> {
            self.inner.scan(dataset, opts).await
        }

        async fn scan_stream(
            &self,
            dataset: &str,
            opts: ScanOptions,
        ) -> Result<RecordBatchStream, HirnDbError> {
            self.inner.scan_stream(dataset, opts).await
        }

        async fn delete(&self, dataset: &str, predicate: &str) -> Result<u64, HirnDbError> {
            if dataset == hirn_storage::datasets::agent::DATASET_NAME
                && self.fail_agent_deletes.load(AtomicOrdering::Acquire)
            {
                return Err(HirnDbError::Unsupported(
                    "simulated agent delete failure".to_string(),
                ));
            }
            self.inner.delete(dataset, predicate).await
        }

        async fn update_where(
            &self,
            dataset: &str,
            filter: &str,
            updates: &[(&str, &str)],
        ) -> Result<u64, HirnDbError> {
            self.inner.update_where(dataset, filter, updates).await
        }

        async fn merge_insert(
            &self,
            dataset: &str,
            on: &[&str],
            batch: RecordBatch,
        ) -> Result<(), HirnDbError> {
            if dataset == hirn_storage::datasets::agent::DATASET_NAME
                && self.fail_agent_writes.load(AtomicOrdering::Acquire)
            {
                return Err(HirnDbError::Unsupported(
                    "simulated agent merge_insert failure".to_string(),
                ));
            }
            if dataset == hirn_storage::datasets::namespace::DATASET_NAME
                && self.fail_namespace_writes.load(AtomicOrdering::Acquire)
            {
                return Err(HirnDbError::Unsupported(
                    "simulated namespace merge_insert failure".to_string(),
                ));
            }
            self.inner.merge_insert(dataset, on, batch).await
        }

        async fn count(&self, dataset: &str, filter: Option<&str>) -> Result<u64, HirnDbError> {
            self.inner.count(dataset, filter).await
        }

        async fn vector_search(
            &self,
            dataset: &str,
            opts: VectorSearchOptions,
        ) -> Result<Vec<RecordBatch>, HirnDbError> {
            self.inner.vector_search(dataset, opts).await
        }

        async fn vector_search_many(
            &self,
            dataset: &str,
            queries: Vec<VectorSearchOptions>,
        ) -> Result<Vec<Vec<RecordBatch>>, HirnDbError> {
            self.inner.vector_search_many(dataset, queries).await
        }

        async fn fts_search(
            &self,
            dataset: &str,
            opts: FtsSearchOptions,
        ) -> Result<Vec<RecordBatch>, HirnDbError> {
            self.inner.fts_search(dataset, opts).await
        }

        async fn hybrid_search(
            &self,
            dataset: &str,
            opts: HybridSearchOptions,
        ) -> Result<Vec<RecordBatch>, HirnDbError> {
            self.inner.hybrid_search(dataset, opts).await
        }

        async fn multivector_search(
            &self,
            dataset: &str,
            opts: MultivectorSearchOptions,
        ) -> Result<Vec<RecordBatch>, HirnDbError> {
            self.inner.multivector_search(dataset, opts).await
        }

        async fn create_index(
            &self,
            dataset: &str,
            config: IndexConfig,
        ) -> Result<(), HirnDbError> {
            self.inner.create_index(dataset, config).await
        }

        async fn optimize_indices(&self, dataset: &str) -> Result<(), HirnDbError> {
            self.inner.optimize_indices(dataset).await
        }

        async fn compact(
            &self,
            dataset: &str,
            opts: CompactOptions,
        ) -> Result<CompactResult, HirnDbError> {
            self.inner.compact(dataset, opts).await
        }

        async fn version(&self, dataset: &str) -> Result<u64, HirnDbError> {
            self.inner.version(dataset).await
        }

        async fn tag(&self, dataset: &str, tag: &str) -> Result<(), HirnDbError> {
            self.inner.tag(dataset, tag).await
        }

        async fn checkout(&self, dataset: &str, version: u64) -> Result<(), HirnDbError> {
            self.inner.checkout(dataset, version).await
        }

        async fn list_tags(&self, dataset: &str) -> Result<Vec<VersionTag>, HirnDbError> {
            self.inner.list_tags(dataset).await
        }

        async fn list_datasets(&self) -> Result<Vec<DatasetInfo>, HirnDbError> {
            self.inner.list_datasets().await
        }

        async fn exists(&self, dataset: &str) -> Result<bool, HirnDbError> {
            self.inner.exists(dataset).await
        }

        async fn list_namespaces(&self) -> Result<Vec<String>, HirnDbError> {
            self.inner.list_namespaces().await
        }

        async fn create_namespace(&self, name: &str) -> Result<(), HirnDbError> {
            self.inner.create_namespace(name).await
        }

        async fn drop_namespace(&self, name: &str) -> Result<(), HirnDbError> {
            self.inner.drop_namespace(name).await
        }

        async fn add_columns(
            &self,
            dataset: &str,
            transforms: Vec<ColumnTransform>,
        ) -> Result<(), HirnDbError> {
            self.inner.add_columns(dataset, transforms).await
        }

        async fn drop_columns(&self, dataset: &str, columns: &[&str]) -> Result<(), HirnDbError> {
            self.inner.drop_columns(dataset, columns).await
        }

        async fn table_provider(&self, dataset: &str) -> Option<Arc<dyn TableProvider>> {
            self.inner.table_provider(dataset).await
        }
    }

    fn test_agent() -> AgentId {
        AgentId::new("namespace_delete_agent").unwrap()
    }

    async fn temp_db() -> (
        HirnDB,
        HirnConfig,
        Arc<dyn PhysicalStore>,
        tempfile::TempDir,
    ) {
        let dir = tempfile::tempdir().unwrap();
        let lance_path = dir.path().join("lance");
        let storage: Arc<dyn PhysicalStore> = HirnDb::open(HirnDbConfig::local(
            lance_path.to_str().expect("temp path should be utf8"),
        ))
        .await
        .unwrap()
        .store_arc();
        let config = HirnConfig::builder()
            .db_path(dir.path().join("db"))
            .working_memory_token_limit(100_000)
            .build()
            .unwrap();
        let db = HirnDB::open_with_config(config.clone(), Arc::clone(&storage))
            .await
            .unwrap();
        (db, config, storage, dir)
    }

    async fn temp_db_with_storage(
        storage: Arc<dyn PhysicalStore>,
    ) -> (
        HirnDB,
        HirnConfig,
        Arc<dyn PhysicalStore>,
        tempfile::TempDir,
    ) {
        let dir = tempfile::tempdir().unwrap();
        let config = HirnConfig::builder()
            .db_path(dir.path().join("db"))
            .working_memory_token_limit(100_000)
            .build()
            .unwrap();
        let db = HirnDB::open_with_config(config.clone(), Arc::clone(&storage))
            .await
            .unwrap();
        (db, config, storage, dir)
    }

    fn episode(namespace: Namespace, content: &str) -> EpisodicRecord {
        EpisodicRecord::builder()
            .content(content)
            .summary(content)
            .event_type(EventType::Observation)
            .importance(0.5)
            .namespace(namespace)
            .agent_id(test_agent())
            .build()
            .unwrap()
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn delete_namespace_records_applied_mutation_envelope() {
        let (db, _config, _storage, _dir) = temp_db().await;
        let namespace = Namespace::new("recoverable_ns_delete").unwrap();
        db.create_namespace(namespace.as_str(), NamespaceKind::Team, vec![])
            .await
            .unwrap();
        let survivor_id = db
            .episodic()
            .remember(episode(Namespace::shared(), "survivor"))
            .await
            .unwrap();
        let episode_id = db
            .episodic()
            .remember(episode(namespace, "delete me"))
            .await
            .unwrap();

        db.delete_namespace(namespace.as_str()).await.unwrap();

        assert!(matches!(
            db.get_namespace(namespace.as_str()).await,
            Err(HirnError::NotFound(_))
        ));
        assert!(matches!(
            db.read_episodic_record(episode_id).await,
            Err(HirnError::NotFound(_))
        ));
        assert!(db.read_episodic_record(survivor_id).await.is_ok());

        let envelopes = hirn_storage::list_mutation_envelopes(
            db.storage_backend(),
            Some(NAMESPACE_DELETE_MUTATION_KIND),
            Some(hirn_storage::MutationEnvelopeState::Applied),
        )
        .await
        .unwrap();
        let envelope = envelopes
            .iter()
            .find(|envelope| {
                envelope
                    .id
                    .starts_with(&format!("namespace-delete:{namespace}:"))
            })
            .expect("namespace delete envelope should exist");
        assert_eq!(envelope.kind, NAMESPACE_DELETE_MUTATION_KIND);
        assert_eq!(envelope.state, hirn_storage::MutationEnvelopeState::Applied);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn namespace_name_can_be_deleted_recreated_and_deleted_again() {
        let (db, _config, _storage, _dir) = temp_db().await;
        let namespace = Namespace::new("reusable_delete_name").unwrap();

        db.create_namespace(namespace.as_str(), NamespaceKind::Team, vec![])
            .await
            .unwrap();
        db.delete_namespace(namespace.as_str()).await.unwrap();
        db.create_namespace(namespace.as_str(), NamespaceKind::Team, vec![])
            .await
            .unwrap();
        db.delete_namespace(namespace.as_str()).await.unwrap();

        let envelopes = hirn_storage::list_mutation_envelopes(
            db.storage_backend(),
            Some(NAMESPACE_DELETE_MUTATION_KIND),
            Some(hirn_storage::MutationEnvelopeState::Applied),
        )
        .await
        .unwrap();
        let namespace_envelopes = envelopes
            .iter()
            .filter(|envelope| {
                envelope
                    .id
                    .starts_with(&format!("namespace-delete:{namespace}:"))
            })
            .count();
        assert_eq!(namespace_envelopes, 2);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn open_reconciles_pending_namespace_delete_mutation_after_partial_cleanup() {
        let (db, config, storage, _dir) = temp_db().await;
        let namespace = Namespace::new("replay_ns_delete").unwrap();
        db.create_namespace(namespace.as_str(), NamespaceKind::Team, vec![])
            .await
            .unwrap();
        let survivor_id = db
            .episodic()
            .remember(episode(Namespace::shared(), "survivor before replay"))
            .await
            .unwrap();
        let episode_id = db
            .episodic()
            .remember(episode(namespace, "already gone before replay"))
            .await
            .unwrap();
        let envelope =
            build_namespace_delete_envelope(namespace, vec![episode_id], vec![], vec![]).unwrap();
        let payload = decode_namespace_delete_envelope(&envelope).unwrap();
        hirn_storage::append_mutation_envelope(db.storage_backend(), &envelope)
            .await
            .unwrap();
        db.delete_episode(episode_id).await.unwrap();
        db.append_namespace_delete_audit_once(&payload)
            .await
            .unwrap();
        drop(db);

        let reopened = HirnDB::open_with_config(config, Arc::clone(&storage))
            .await
            .unwrap();

        assert!(matches!(
            reopened.get_namespace(namespace.as_str()).await,
            Err(HirnError::NotFound(_))
        ));
        assert!(reopened.read_episodic_record(survivor_id).await.is_ok());
        let stored_envelope = hirn_storage::get_mutation_envelope(storage.as_ref(), &envelope.id)
            .await
            .unwrap()
            .expect("namespace delete envelope should remain queryable");
        assert_eq!(
            stored_envelope.state,
            hirn_storage::MutationEnvelopeState::Applied
        );
        let audit_log = reopened.audit_log(None, None).await.unwrap();
        let matching = audit_log
            .iter()
            .filter(|entry| {
                entry.id == payload.audit_entry_id
                    && matches!(
                        &entry.action,
                        hirn_core::audit::AuditAction::NamespaceDeleted { namespace: deleted }
                            if deleted == namespace.as_str()
                    )
            })
            .count();
        assert_eq!(
            matching, 1,
            "namespace delete audit replay must be idempotent"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn failed_agent_update_preserves_existing_row() {
        let fault_store = Arc::new(FaultInjectingNamespaceStore::new());
        let storage: Arc<dyn PhysicalStore> = fault_store.clone();
        let (db, _config, _storage, _dir) = temp_db_with_storage(storage).await;

        let agent_id = AgentId::new("upserted_agent").unwrap();
        db.register_agent(&agent_id, "Original Agent")
            .await
            .unwrap();

        let mut updated = db.get_agent(&agent_id).await.unwrap();
        updated.display_name = "Updated Agent".to_string();

        fault_store.fail_agent_writes();

        let error = db.update_agent(&updated).await.unwrap_err();
        assert!(
            error
                .to_string()
                .contains("simulated agent merge_insert failure"),
            "expected keyed-upsert failure, got: {error}"
        );

        let agents = db.list_agents().await.unwrap();
        let stored = agents
            .iter()
            .find(|agent| agent.id == agent_id)
            .expect("original agent row should still exist");
        assert_eq!(stored.display_name, "Original Agent");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn failed_team_membership_update_preserves_existing_namespace_row() {
        let fault_store = Arc::new(FaultInjectingNamespaceStore::new());
        let storage: Arc<dyn PhysicalStore> = fault_store.clone();
        let (db, _config, _storage, _dir) = temp_db_with_storage(storage).await;

        let team = "team_upsert_guard";
        let member = AgentId::new("team_member").unwrap();
        db.create_namespace(team, NamespaceKind::Team, vec![])
            .await
            .unwrap();

        fault_store.fail_namespace_writes();

        let error = db.add_agent_to_team(&member, team).await.unwrap_err();
        assert!(
            error
                .to_string()
                .contains("simulated namespace merge_insert failure"),
            "expected keyed-upsert failure, got: {error}"
        );

        let stored = db.get_namespace(team).await.unwrap();
        assert!(stored.member_agents.is_empty());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn open_reconciles_pending_agent_register_mutation_after_partial_namespace_write() {
        let fault_store = Arc::new(FaultInjectingNamespaceStore::new());
        let storage: Arc<dyn PhysicalStore> = fault_store.clone();
        let (db, config, _storage, _dir) = temp_db_with_storage(storage.clone()).await;

        let agent_id = AgentId::new("recoverable_agent_register").unwrap();
        let private_ns = Namespace::private_for(&agent_id);

        fault_store.fail_namespace_writes();
        let error = db
            .register_agent(&agent_id, "Recoverable Agent")
            .await
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("simulated namespace merge_insert failure"),
            "expected namespace upsert failure, got: {error}"
        );

        let agents = db.list_agents().await.unwrap();
        assert!(agents.iter().any(|agent| agent.id == agent_id));
        assert!(matches!(
            db.get_namespace(private_ns.as_str()).await,
            Err(HirnError::NotFound(_))
        ));
        drop(db);

        fault_store.allow_namespace_writes();
        let reopened = HirnDB::open_with_config(config, storage).await.unwrap();

        assert_eq!(
            reopened.get_agent(&agent_id).await.unwrap().display_name,
            "Recoverable Agent"
        );
        assert_eq!(
            reopened
                .get_namespace(private_ns.as_str())
                .await
                .unwrap()
                .namespace,
            private_ns
        );

        let envelopes = hirn_storage::list_mutation_envelopes(
            reopened.storage_backend(),
            Some(AGENT_REGISTER_MUTATION_KIND),
            Some(hirn_storage::MutationEnvelopeState::Applied),
        )
        .await
        .unwrap();
        assert!(envelopes.iter().any(|envelope| {
            envelope
                .id
                .starts_with(&format!("agent-register:{agent_id}:"))
        }));

        let audit_entries = reopened.audit_log(None, None).await.unwrap();
        let matching = audit_entries
            .iter()
            .filter(|entry| {
                matches!(
                    &entry.action,
                    hirn_core::audit::AuditAction::AgentRegistered { agent_id: recorded }
                        if recorded == &agent_id
                )
            })
            .count();
        assert_eq!(matching, 1);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn open_reconciles_pending_agent_deregister_mutation_after_partial_agent_delete() {
        let fault_store = Arc::new(FaultInjectingNamespaceStore::new());
        let storage: Arc<dyn PhysicalStore> = fault_store.clone();
        let (db, config, _storage, _dir) = temp_db_with_storage(storage.clone()).await;

        let agent_id = AgentId::new("recoverable_agent_deregister").unwrap();
        let private_ns = Namespace::private_for(&agent_id);
        db.register_agent(&agent_id, "Recoverable Delete")
            .await
            .unwrap();

        fault_store.fail_agent_deletes();
        let error = db.deregister_agent(&agent_id).await.unwrap_err();
        assert!(
            error.to_string().contains("simulated agent delete failure"),
            "expected agent delete failure, got: {error}"
        );

        assert!(matches!(
            db.get_namespace(private_ns.as_str()).await,
            Err(HirnError::NotFound(_))
        ));
        let agents = db.list_agents().await.unwrap();
        assert!(agents.iter().any(|agent| agent.id == agent_id));
        drop(db);

        fault_store.allow_agent_deletes();
        fault_store.allow_agent_writes();
        let reopened = HirnDB::open_with_config(config, storage).await.unwrap();

        assert!(matches!(
            reopened.get_agent(&agent_id).await,
            Err(HirnError::NotFound(_))
        ));
        assert!(matches!(
            reopened.get_namespace(private_ns.as_str()).await,
            Err(HirnError::NotFound(_))
        ));

        let envelopes = hirn_storage::list_mutation_envelopes(
            reopened.storage_backend(),
            Some(AGENT_DEREGISTER_MUTATION_KIND),
            Some(hirn_storage::MutationEnvelopeState::Applied),
        )
        .await
        .unwrap();
        assert!(envelopes.iter().any(|envelope| {
            envelope
                .id
                .starts_with(&format!("agent-deregister:{agent_id}:"))
        }));

        let audit_entries = reopened.audit_log(None, None).await.unwrap();
        let matching = audit_entries
            .iter()
            .filter(|entry| {
                matches!(
                    &entry.action,
                    hirn_core::audit::AuditAction::AgentDeregistered { agent_id: recorded }
                        if recorded == &agent_id
                )
            })
            .count();
        assert_eq!(matching, 1);
    }
}
