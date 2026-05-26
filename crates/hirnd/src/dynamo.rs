//! DynamoDB-backed metadata store for serverless deployments.
//!
//! Provides distributed coordination without Raft by using DynamoDB as a
//! metadata store with conditional writes for optimistic concurrency.
//!
//! Two tables:
//! - `hirn_metadata`: realm configuration, node registry, cluster state
//! - `hirn_locks`: compaction leases with TTL for automatic expiry
//!
//! This module is only compiled when the `serverless` feature is enabled.

#[cfg(feature = "serverless")]
pub mod store {
    use std::collections::HashMap;
    use std::time::{SystemTime, UNIX_EPOCH};

    use aws_sdk_dynamodb::Client;
    use aws_sdk_dynamodb::types::{AttributeValue, BillingMode};
    use tracing::{debug, info, warn};

    /// DynamoDB metadata store for serverless hirnd deployments.
    pub struct DynamoMetadataStore {
        client: Client,
        metadata_table: String,
        locks_table: String,
    }

    /// Configuration for the DynamoDB metadata store.
    #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
    pub struct DynamoConfig {
        /// DynamoDB table name for metadata (default: "hirn_metadata").
        #[serde(default = "default_metadata_table")]
        pub metadata_table: String,
        /// DynamoDB table name for locks (default: "hirn_locks").
        #[serde(default = "default_locks_table")]
        pub locks_table: String,
        /// AWS region override (uses SDK default chain if not set).
        pub region: Option<String>,
        /// Custom endpoint URL (for LocalStack / DynamoDB Local development).
        pub endpoint_url: Option<String>,
    }

    fn default_metadata_table() -> String {
        "hirn_metadata".to_string()
    }

    fn default_locks_table() -> String {
        "hirn_locks".to_string()
    }

    fn now_epoch_secs() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock is before UNIX epoch")
            .as_secs()
    }

    impl DynamoMetadataStore {
        /// Create a new DynamoDB metadata store.
        pub async fn new(config: &DynamoConfig) -> Self {
            let mut sdk_config = aws_config::defaults(aws_config::BehaviorVersion::latest());
            if let Some(ref region) = config.region {
                sdk_config = sdk_config.region(aws_config::Region::new(region.clone()));
            }
            if let Some(ref endpoint) = config.endpoint_url {
                sdk_config = sdk_config.endpoint_url(endpoint);
            }
            let sdk_config = sdk_config.load().await;
            let client = Client::new(&sdk_config);

            Self {
                client,
                metadata_table: config.metadata_table.clone(),
                locks_table: config.locks_table.clone(),
            }
        }

        /// Ensure required DynamoDB tables exist (creates them if missing).
        pub async fn ensure_tables(&self) -> Result<(), String> {
            // Check if metadata table exists, create if not.
            self.ensure_table(&self.metadata_table, "pk", "S", Some(("sk", "S")))
                .await?;

            // Locks table with TTL.
            self.ensure_table(&self.locks_table, "pk", "S", None)
                .await?;

            // Enable TTL on the locks table so DynamoDB auto-expires lease items.
            self.enable_ttl(&self.locks_table, "ttl").await?;

            info!(
                metadata_table = %self.metadata_table,
                locks_table = %self.locks_table,
                "DynamoDB tables verified"
            );
            Ok(())
        }

        async fn ensure_table(
            &self,
            table_name: &str,
            pk_name: &str,
            pk_type: &str,
            sk: Option<(&str, &str)>,
        ) -> Result<(), String> {
            use aws_sdk_dynamodb::types::{
                AttributeDefinition, KeySchemaElement, KeyType, ScalarAttributeType,
            };

            let pk_attr_type = match pk_type {
                "N" => ScalarAttributeType::N,
                _ => ScalarAttributeType::S,
            };

            let mut attrs = vec![
                AttributeDefinition::builder()
                    .attribute_name(pk_name)
                    .attribute_type(pk_attr_type.clone())
                    .build()
                    .map_err(|e| e.to_string())?,
            ];

            let mut keys = vec![
                KeySchemaElement::builder()
                    .attribute_name(pk_name)
                    .key_type(KeyType::Hash)
                    .build()
                    .map_err(|e| e.to_string())?,
            ];

            if let Some((sk_name, sk_type)) = sk {
                let sk_attr_type = match sk_type {
                    "N" => ScalarAttributeType::N,
                    _ => ScalarAttributeType::S,
                };
                attrs.push(
                    AttributeDefinition::builder()
                        .attribute_name(sk_name)
                        .attribute_type(sk_attr_type)
                        .build()
                        .map_err(|e| e.to_string())?,
                );
                keys.push(
                    KeySchemaElement::builder()
                        .attribute_name(sk_name)
                        .key_type(KeyType::Range)
                        .build()
                        .map_err(|e| e.to_string())?,
                );
            }

            match self
                .client
                .create_table()
                .table_name(table_name)
                .set_attribute_definitions(Some(attrs))
                .set_key_schema(Some(keys))
                .billing_mode(BillingMode::PayPerRequest)
                .send()
                .await
            {
                Ok(_) => {
                    info!(table = table_name, "DynamoDB table created");
                    Ok(())
                }
                Err(sdk_err) => {
                    if sdk_err
                        .as_service_error()
                        .is_some_and(|e| e.is_resource_in_use_exception())
                    {
                        debug!(table = table_name, "DynamoDB table already exists");
                        Ok(())
                    } else {
                        Err(format!(
                            "failed to create DynamoDB table '{table_name}': {sdk_err}"
                        ))
                    }
                }
            }
        }

        /// Enable TTL on a DynamoDB table. Idempotent — succeeds if already enabled.
        async fn enable_ttl(&self, table_name: &str, attribute_name: &str) -> Result<(), String> {
            use aws_sdk_dynamodb::types::TimeToLiveSpecification;

            let ttl_spec = TimeToLiveSpecification::builder()
                .enabled(true)
                .attribute_name(attribute_name)
                .build()
                .map_err(|e| e.to_string())?;

            match self
                .client
                .update_time_to_live()
                .table_name(table_name)
                .time_to_live_specification(ttl_spec)
                .send()
                .await
            {
                Ok(_) => {
                    debug!(
                        table = table_name,
                        attribute = attribute_name,
                        "TTL enabled"
                    );
                    Ok(())
                }
                Err(sdk_err) => {
                    // ValidationException occurs when TTL is already enabled — that's fine.
                    let err_str = sdk_err.to_string();
                    if err_str.contains("already enabled")
                        || err_str.contains("TimeToLive is already")
                    {
                        debug!(table = table_name, "TTL already enabled");
                        Ok(())
                    } else {
                        warn!(table = table_name, error = %sdk_err, "failed to enable TTL — leases may require manual cleanup");
                        Ok(()) // Non-fatal: TTL is optimization, not correctness requirement.
                    }
                }
            }
        }

        /// Acquire a compaction lease for a realm using conditional writes.
        /// Returns `Ok(true)` if acquired, `Ok(false)` if already held.
        pub async fn acquire_lease(
            &self,
            realm: &str,
            holder: &str,
            duration_secs: u64,
        ) -> Result<bool, String> {
            let now = now_epoch_secs();
            let expires = now + duration_secs;

            let result = self
                .client
                .put_item()
                .table_name(&self.locks_table)
                .item("pk", AttributeValue::S(format!("lease#{realm}")))
                .item("holder", AttributeValue::S(holder.to_string()))
                .item("acquired_at", AttributeValue::N(now.to_string()))
                .item("expires_at", AttributeValue::N(expires.to_string()))
                .item("ttl", AttributeValue::N((expires + 60).to_string()))
                .condition_expression(
                    "attribute_not_exists(pk) OR expires_at < :now OR holder = :holder",
                )
                .expression_attribute_values(":now", AttributeValue::N(now.to_string()))
                .expression_attribute_values(":holder", AttributeValue::S(holder.to_string()))
                .send()
                .await;

            match result {
                Ok(_) => {
                    info!(realm, holder, expires, "DynamoDB lease acquired");
                    Ok(true)
                }
                Err(sdk_err) => {
                    if sdk_err
                        .as_service_error()
                        .is_some_and(|e| e.is_conditional_check_failed_exception())
                    {
                        debug!(realm, holder, "DynamoDB lease conflict");
                        Ok(false)
                    } else {
                        Err(format!("DynamoDB lease acquire error: {sdk_err}"))
                    }
                }
            }
        }

        /// Release a compaction lease.
        ///
        /// Silently succeeds if the lease was already released or expired (idempotent).
        pub async fn release_lease(&self, realm: &str, holder: &str) -> Result<(), String> {
            match self
                .client
                .delete_item()
                .table_name(&self.locks_table)
                .key("pk", AttributeValue::S(format!("lease#{realm}")))
                .condition_expression("holder = :holder")
                .expression_attribute_values(":holder", AttributeValue::S(holder.to_string()))
                .send()
                .await
            {
                Ok(_) => {
                    info!(realm, holder, "DynamoDB lease released");
                    Ok(())
                }
                Err(sdk_err) => {
                    if sdk_err
                        .as_service_error()
                        .is_some_and(|e| e.is_conditional_check_failed_exception())
                    {
                        // Lease already released, expired, or held by someone else — idempotent.
                        debug!(realm, holder, "DynamoDB lease already released or expired");
                        Ok(())
                    } else {
                        Err(format!("DynamoDB lease release error: {sdk_err}"))
                    }
                }
            }
        }

        /// Store realm metadata (realm → owner node mapping).
        pub async fn assign_realm(&self, realm: &str, owner: &str) -> Result<(), String> {
            self.client
                .put_item()
                .table_name(&self.metadata_table)
                .item("pk", AttributeValue::S("realm".to_string()))
                .item("sk", AttributeValue::S(realm.to_string()))
                .item("owner", AttributeValue::S(owner.to_string()))
                .item(
                    "updated_at",
                    AttributeValue::N(now_epoch_secs().to_string()),
                )
                .send()
                .await
                .map_err(|e| format!("DynamoDB assign realm error: {e}"))?;
            Ok(())
        }

        /// Get the owner node for a realm.
        pub async fn realm_owner(&self, realm: &str) -> Result<Option<String>, String> {
            let result = self
                .client
                .get_item()
                .table_name(&self.metadata_table)
                .key("pk", AttributeValue::S("realm".to_string()))
                .key("sk", AttributeValue::S(realm.to_string()))
                .send()
                .await
                .map_err(|e| format!("DynamoDB get realm owner error: {e}"))?;

            Ok(result.item.and_then(|item| {
                item.get("owner")
                    .and_then(|v| v.as_s().ok())
                    .map(|s| s.to_string())
            }))
        }

        /// Register a node in the cluster.
        pub async fn register_node(&self, node_id: &str, addr: &str) -> Result<(), String> {
            self.client
                .put_item()
                .table_name(&self.metadata_table)
                .item("pk", AttributeValue::S("node".to_string()))
                .item("sk", AttributeValue::S(node_id.to_string()))
                .item("addr", AttributeValue::S(addr.to_string()))
                .item("heartbeat", AttributeValue::N(now_epoch_secs().to_string()))
                .send()
                .await
                .map_err(|e| format!("DynamoDB register node error: {e}"))?;
            Ok(())
        }

        /// Get all registered nodes (paginated — handles >1MB of results).
        pub async fn list_nodes(&self) -> Result<HashMap<String, String>, String> {
            let mut nodes = HashMap::new();
            let mut exclusive_start_key = None;

            loop {
                let mut query = self
                    .client
                    .query()
                    .table_name(&self.metadata_table)
                    .key_condition_expression("pk = :pk")
                    .expression_attribute_values(":pk", AttributeValue::S("node".to_string()));

                if let Some(ref start_key) = exclusive_start_key {
                    query = query.set_exclusive_start_key(Some(start_key.clone()));
                }

                let result = query
                    .send()
                    .await
                    .map_err(|e| format!("DynamoDB list nodes error: {e}"))?;

                if let Some(items) = result.items {
                    for item in items {
                        if let (Some(id), Some(addr)) = (
                            item.get("sk").and_then(|v| v.as_s().ok()),
                            item.get("addr").and_then(|v| v.as_s().ok()),
                        ) {
                            nodes.insert(id.clone(), addr.clone());
                        }
                    }
                }

                match result.last_evaluated_key {
                    Some(key) if !key.is_empty() => exclusive_start_key = Some(key),
                    _ => break,
                }
            }

            Ok(nodes)
        }

        /// Update heartbeat timestamp for a node (liveness check).
        pub async fn heartbeat(&self, node_id: &str) -> Result<(), String> {
            self.client
                .update_item()
                .table_name(&self.metadata_table)
                .key("pk", AttributeValue::S("node".to_string()))
                .key("sk", AttributeValue::S(node_id.to_string()))
                .update_expression("SET heartbeat = :hb")
                .expression_attribute_values(":hb", AttributeValue::N(now_epoch_secs().to_string()))
                .send()
                .await
                .map_err(|e| format!("DynamoDB heartbeat error: {e}"))?;
            Ok(())
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn dynamo_config_defaults() {
            let cfg: DynamoConfig = serde_json::from_str("{}").unwrap();
            assert_eq!(cfg.metadata_table, "hirn_metadata");
            assert_eq!(cfg.locks_table, "hirn_locks");
            assert!(cfg.region.is_none());
            assert!(cfg.endpoint_url.is_none());
        }
    }
}

// Re-export when feature is enabled.
#[cfg(feature = "serverless")]
pub use store::{DynamoConfig, DynamoMetadataStore};
