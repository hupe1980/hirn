use std::collections::HashMap;
use std::sync::Arc;

use lance_namespace::LanceNamespace;
use lance_namespace_impls::ConnectBuilder;

use crate::error::HirnDbError;

/// Configuration for connecting to a Lance namespace.
///
/// Supports local filesystem and cloud object stores (S3, GCS, Azure, OSS)
/// via `lance-namespace-impls` feature flags.
#[derive(Debug, Clone)]
pub struct NamespaceConfig {
    /// Root URI: local path, `s3://bucket/path`, `gs://bucket/path`,
    /// `az://container/path`, or `oss://bucket/path`.
    pub root: String,
    /// Additional properties passed to the namespace builder
    /// (e.g., `storage.region`, `storage.account_name`).
    pub properties: HashMap<String, String>,
}

impl NamespaceConfig {
    /// Create a new config for a local filesystem path.
    pub fn local(path: impl Into<String>) -> Self {
        Self {
            root: path.into(),
            properties: HashMap::new(),
        }
    }

    /// Create a new config with a root URI and optional properties.
    pub fn new(root: impl Into<String>) -> Self {
        Self {
            root: root.into(),
            properties: HashMap::new(),
        }
    }

    /// Add a property to the config.
    pub fn with_property(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.properties.insert(key.into(), value.into());
        self
    }

    /// Connect to the namespace and return a `LanceNamespace` handle.
    pub async fn connect(&self) -> Result<Arc<dyn LanceNamespace>, HirnDbError> {
        let mut builder = ConnectBuilder::new("dir").property("root", &self.root);

        for (k, v) in &self.properties {
            builder = builder.property(k, v);
        }

        let ns = builder
            .connect()
            .await
            .map_err(|e| HirnDbError::NamespaceError(e.to_string()))?;

        Ok(ns)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_config_creation() {
        let cfg = NamespaceConfig::local("/tmp/test");
        assert_eq!(cfg.root, "/tmp/test");
        assert!(cfg.properties.is_empty());
    }

    #[test]
    fn config_with_properties() {
        let cfg =
            NamespaceConfig::new("s3://bucket/data").with_property("storage.region", "us-east-1");
        assert_eq!(cfg.root, "s3://bucket/data");
        assert_eq!(
            cfg.properties.get("storage.region"),
            Some(&"us-east-1".to_string())
        );
    }
}
