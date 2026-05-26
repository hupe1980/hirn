//! Config-driven provider registry for AI traits.
//!
//! The [`ProviderRegistry`] holds named instances of the four core AI
//! traits — [`Embedder`], [`Tokenizer`], [`Reranker`], and [`LlmProvider`]
//! — and exposes a default + by-name lookup pattern.
//!
//! Providers can be configured via:
//! - **Programmatic API**: `register_embedder()`, `register_llm()`, etc.
//! - **Environment variables**: `from_env()` auto-discovers from env vars
//! - **TOML configuration**: `from_config()` / `from_toml()` for config-driven setup

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::RwLock;

use hirn_core::embed::{Embedder, LlmProvider, Reranker};
use hirn_core::tokenizer::{EstimatingTokenizer, Tokenizer};
use hirn_core::{HirnError, HirnResult};

/// Names of the default providers in each category.
#[derive(Debug, Clone, Default)]
pub struct ProviderDefaults {
    pub embedder: Option<String>,
    pub tokenizer: Option<String>,
    pub reranker: Option<String>,
    pub llm: Option<String>,
}

// ── TOML configuration types ─────────────────────────────────────────────

/// How an API key is specified in the TOML config.
///
/// Supports either a literal string or an environment variable reference:
///
/// ```toml
/// api_key = "sk-literal-key"          # literal
/// api_key = { env = "OPENAI_API_KEY" } # env var reference
/// ```
#[derive(Debug, Clone, serde::Deserialize, PartialEq)]
#[serde(untagged)]
pub enum ApiKeySource {
    /// Reference to an environment variable.
    Env {
        /// Name of the environment variable.
        env: String,
    },
    /// A literal API key string.
    Literal(String),
}

impl ApiKeySource {
    /// Resolve the API key to a string value.
    ///
    /// For `Env` variants, the environment variable is read.
    /// Returns an error if the variable is not set.
    pub fn resolve(&self) -> HirnResult<String> {
        match self {
            Self::Literal(key) => Ok(key.clone()),
            Self::Env { env } => std::env::var(env).map_err(|_| {
                HirnError::config(format!(
                    "environment variable '{env}' not set (required by provider config)"
                ))
            }),
        }
    }
}

/// Configuration for a single embedder provider.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct EmbedderConfig {
    /// Provider type: `"openai"`, `"ollama"`, `"pseudo"`.
    pub r#type: String,
    /// Model name (e.g. `"text-embedding-3-small"`).
    pub model: Option<String>,
    /// Embedding dimensions.
    pub dimensions: Option<usize>,
    /// API key (for remote providers).
    pub api_key: Option<ApiKeySource>,
    /// Base URL override.
    pub base_url: Option<String>,
}

/// Configuration for a single LLM provider.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct LlmConfig {
    /// Provider type: `"openai"`, `"ollama"`, `"anthropic"`, `"mock"`.
    pub r#type: String,
    /// Model name.
    pub model: Option<String>,
    /// API key (for remote providers).
    pub api_key: Option<ApiKeySource>,
    /// Base URL override.
    pub base_url: Option<String>,
}

/// Configuration for a single reranker provider.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct RerankerConfig {
    /// Provider type: `"cohere"`, `"cross-encoder"`, `"noop"`.
    pub r#type: String,
    /// Model name.
    pub model: Option<String>,
    /// API key (for remote providers).
    pub api_key: Option<ApiKeySource>,
    /// Base URL override.
    pub base_url: Option<String>,
}

/// Configuration for a single tokenizer provider.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct TokenizerConfig {
    /// Provider type: `"tiktoken"`, `"huggingface"`, `"estimating"`.
    pub r#type: String,
    /// Model name or identifier.
    pub model: Option<String>,
    /// Maximum token length.
    pub max_length: Option<usize>,
}

/// Which provider name to use as default for each category.
#[derive(Debug, Clone, Default, serde::Deserialize)]
pub struct DefaultsConfig {
    pub embedder: Option<String>,
    pub tokenizer: Option<String>,
    pub reranker: Option<String>,
    pub llm: Option<String>,
}

/// Top-level provider configuration, TOML-deserializable.
///
/// # Example
///
/// ```toml
/// [providers.embedder.openai]
/// type = "openai"
/// model = "text-embedding-3-small"
/// api_key = { env = "OPENAI_API_KEY" }
/// dimensions = 1536
///
/// [providers.llm.claude]
/// type = "anthropic"
/// model = "claude-sonnet-4-20250514"
/// api_key = { env = "ANTHROPIC_API_KEY" }
///
/// [providers.reranker.cohere]
/// type = "cohere"
/// model = "rerank-v3.5"
/// api_key = { env = "COHERE_API_KEY" }
///
/// [providers.tokenizer.default]
/// type = "estimating"
///
/// [defaults]
/// embedder = "openai"
/// llm = "claude"
/// reranker = "cohere"
/// tokenizer = "default"
/// ```
#[derive(Debug, Clone, Default, serde::Deserialize)]
pub struct ProviderConfig {
    /// Provider definitions grouped by category.
    #[serde(default)]
    pub providers: ProvidersSection,
    /// Which provider name to use as the default for each category.
    #[serde(default)]
    pub defaults: DefaultsConfig,
}

/// The `[providers]` section: maps category → name → config.
#[derive(Debug, Clone, Default, serde::Deserialize)]
pub struct ProvidersSection {
    #[serde(default)]
    pub embedder: HashMap<String, EmbedderConfig>,
    #[serde(default)]
    pub llm: HashMap<String, LlmConfig>,
    #[serde(default)]
    pub reranker: HashMap<String, RerankerConfig>,
    #[serde(default)]
    pub tokenizer: HashMap<String, TokenizerConfig>,
}

/// Central registry for AI providers, supporting runtime hot-swap.
///
/// Thread-safe: all state is behind `RwLock`/`Arc` so the registry can be
/// shared across `tokio` tasks.
///
/// # Example
///
/// ```rust
/// use hirn_engine::ProviderRegistry;
/// use hirn_provider::PseudoEmbedder;
/// use std::sync::Arc;
///
/// let mut reg = ProviderRegistry::new();
/// reg.register_embedder("pseudo", Arc::new(PseudoEmbedder::new(128)));
/// reg.set_default_embedder("pseudo").unwrap();
/// assert!(reg.embedder().is_some());
/// ```
pub struct ProviderRegistry {
    embedders: RwLock<HashMap<String, Arc<dyn Embedder>>>,
    tokenizers: RwLock<HashMap<String, Arc<dyn Tokenizer>>>,
    rerankers: RwLock<HashMap<String, Arc<dyn Reranker>>>,
    llms: RwLock<HashMap<String, Arc<dyn LlmProvider>>>,
    defaults: RwLock<ProviderDefaults>,
}

impl std::fmt::Debug for ProviderRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let defaults = self.defaults.read();
        f.debug_struct("ProviderRegistry")
            .field(
                "embedders",
                &self.embedders.read().keys().collect::<Vec<_>>(),
            )
            .field(
                "tokenizers",
                &self.tokenizers.read().keys().collect::<Vec<_>>(),
            )
            .field(
                "rerankers",
                &self.rerankers.read().keys().collect::<Vec<_>>(),
            )
            .field("llms", &self.llms.read().keys().collect::<Vec<_>>())
            .field("defaults", &*defaults)
            .finish()
    }
}

impl ProviderRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            embedders: RwLock::new(HashMap::new()),
            tokenizers: RwLock::new(HashMap::new()),
            rerankers: RwLock::new(HashMap::new()),
            llms: RwLock::new(HashMap::new()),
            defaults: RwLock::new(ProviderDefaults::default()),
        }
    }

    fn with_fallbacks() -> Self {
        let reg = Self::new();

        reg.register_embedder("pseudo", Arc::new(hirn_provider::PseudoEmbedder::new(384)));
        reg.register_tokenizer("estimating", Arc::new(EstimatingTokenizer));
        reg.register_reranker("noop", Arc::new(hirn_core::embed::NoopReranker));
        reg.register_llm(
            "mock",
            Arc::new(hirn_provider::MockLlmProvider::new("mock")),
        );

        let _ = reg.set_default_embedder("pseudo");
        let _ = reg.set_default_tokenizer("estimating");
        let _ = reg.set_default_reranker("noop");
        let _ = reg.set_default_llm("mock");

        #[cfg(feature = "tiktoken")]
        if let Ok(tokenizer) = hirn_provider::build_tokenizer("tiktoken", Some("cl100k_base"), None)
        {
            reg.register_tokenizer("tiktoken", tokenizer);
            let _ = reg.set_default_tokenizer("tiktoken");
        }

        reg
    }

    #[allow(dead_code)]
    fn default_embedder_is_unset_or_fallback(&self) -> bool {
        self.defaults
            .read()
            .embedder
            .as_deref()
            .is_none_or(|name| name == "pseudo")
    }

    #[allow(dead_code)]
    fn default_reranker_is_unset_or_fallback(&self) -> bool {
        self.defaults
            .read()
            .reranker
            .as_deref()
            .is_none_or(|name| name == "noop")
    }

    #[allow(dead_code)]
    fn default_llm_is_unset_or_fallback(&self) -> bool {
        self.defaults
            .read()
            .llm
            .as_deref()
            .is_none_or(|name| name == "mock")
    }

    #[allow(unused_variables)]
    fn populate_from_env(reg: &Self) {
        // Override with real providers based on env vars.
        #[cfg(feature = "openai")]
        if let Ok(key) = std::env::var("OPENAI_API_KEY") {
            Self::register_openai_from_key(
                reg,
                key,
                |api_key| {
                    hirn_provider::OpenAIEmbedder::new(api_key, "text-embedding-3-small", 1536)
                        .map(|embedder| Arc::new(embedder) as Arc<dyn Embedder>)
                },
                |api_key| {
                    hirn_provider::OpenAILlmProvider::new(api_key, "gpt-4o-mini")
                        .map(|provider| Arc::new(provider) as Arc<dyn LlmProvider>)
                },
            );
        }

        #[cfg(feature = "ollama")]
        {
            let host = std::env::var("OLLAMA_HOST")
                .unwrap_or_else(|_| "http://localhost:11434".to_owned());
            if std::env::var("OLLAMA_HOST").is_ok() {
                match hirn_provider::OllamaEmbedder::new("nomic-embed-text", 768) {
                    Ok(embedder) => match embedder.with_host(&host) {
                        Ok(embedder) => {
                            reg.register_embedder("ollama", Arc::new(embedder));
                            if reg.defaults.read().embedder.as_deref() != Some("openai") {
                                let _ = reg.set_default_embedder("ollama");
                            }
                        }
                        Err(err) => {
                            tracing::warn!(error = %err, provider = "ollama", "failed to validate optional ollama embedder host from environment");
                        }
                    },
                    Err(err) => {
                        tracing::warn!(error = %err, provider = "ollama", "failed to initialize optional ollama embedder from environment");
                    }
                }

                match hirn_provider::OllamaLlmProvider::new("llama3.1") {
                    Ok(provider) => match provider.with_host(&host) {
                        Ok(provider) => {
                            reg.register_llm("ollama", Arc::new(provider));
                            if reg.defaults.read().llm.as_deref() != Some("openai") {
                                let _ = reg.set_default_llm("ollama");
                            }
                        }
                        Err(err) => {
                            tracing::warn!(error = %err, provider = "ollama", "failed to validate optional ollama llm host from environment");
                        }
                    },
                    Err(err) => {
                        tracing::warn!(error = %err, provider = "ollama", "failed to initialize optional ollama llm from environment");
                    }
                }
            }
        }

        #[cfg(feature = "cohere")]
        match hirn_provider::CohereReranker::from_env() {
            Ok(Some(cohere_reranker)) => {
                reg.register_reranker("cohere", Arc::new(cohere_reranker));
                let _ = reg.set_default_reranker("cohere");
            }
            Ok(None) => {}
            Err(err) => {
                tracing::warn!(error = %err, provider = "cohere", "failed to initialize optional cohere reranker from environment");
            }
        }

        #[cfg(feature = "cohere")]
        match hirn_provider::CohereEmbedder::from_env() {
            Ok(Some(cohere_embedder)) => {
                reg.register_embedder("cohere", Arc::new(cohere_embedder));
                if reg.default_embedder_is_unset_or_fallback() {
                    let _ = reg.set_default_embedder("cohere");
                }
            }
            Ok(None) => {}
            Err(err) => {
                tracing::warn!(error = %err, provider = "cohere", "failed to initialize optional cohere embedder from environment");
            }
        }

        #[cfg(feature = "voyage")]
        match hirn_provider::VoyageEmbedder::from_env() {
            Ok(Some(voyage_embedder)) => {
                reg.register_embedder("voyage", Arc::new(voyage_embedder));
                if reg.default_embedder_is_unset_or_fallback() {
                    let _ = reg.set_default_embedder("voyage");
                }
            }
            Ok(None) => {}
            Err(err) => {
                tracing::warn!(error = %err, provider = "voyage", "failed to initialize optional voyage embedder from environment");
            }
        }

        #[cfg(feature = "cross-encoder")]
        if let Ok(cross_encoder) = hirn_provider::CrossEncoderReranker::default_model() {
            reg.register_reranker("cross-encoder", Arc::new(cross_encoder));
            if reg.default_reranker_is_unset_or_fallback() {
                let _ = reg.set_default_reranker("cross-encoder");
            }
        }

        #[cfg(feature = "anthropic")]
        if let Ok(key) = std::env::var("ANTHROPIC_API_KEY") {
            match hirn_provider::AnthropicProvider::new(key) {
                Ok(provider) => {
                    reg.register_llm("anthropic", Arc::new(provider));
                    if reg.default_llm_is_unset_or_fallback() {
                        let _ = reg.set_default_llm("anthropic");
                    }
                }
                Err(err) => {
                    tracing::warn!(error = %err, provider = "anthropic", "failed to initialize optional anthropic llm from environment");
                }
            }
        }

        #[cfg(feature = "hf-tokenizer")]
        if let Ok(model_id) = std::env::var("HF_TOKENIZER_MODEL") {
            if let Ok(hf_tok) = hirn_provider::HuggingFaceTokenizer::from_pretrained(&model_id) {
                reg.register_tokenizer("huggingface", Arc::new(hf_tok));
                let _ = reg.set_default_tokenizer("huggingface");
            }
        }
    }

    #[cfg(feature = "openai")]
    fn register_openai_from_key<FEmbed, FLlm>(
        reg: &Self,
        key: String,
        make_embedder: FEmbed,
        make_llm: FLlm,
    ) where
        FEmbed: FnOnce(String) -> HirnResult<Arc<dyn Embedder>>,
        FLlm: FnOnce(String) -> HirnResult<Arc<dyn LlmProvider>>,
    {
        match make_embedder(key.clone()) {
            Ok(embedder) => {
                reg.register_embedder("openai", embedder);
                let _ = reg.set_default_embedder("openai");
            }
            Err(err) => {
                tracing::warn!(error = %err, provider = "openai", "failed to initialize optional openai embedder from environment");
            }
        }

        match make_llm(key) {
            Ok(provider) => {
                reg.register_llm("openai", provider);
                let _ = reg.set_default_llm("openai");
            }
            Err(err) => {
                tracing::warn!(error = %err, provider = "openai", "failed to initialize optional openai llm from environment");
            }
        }
    }

    // ── Embedder ─────────────────────────────────────────────────────

    /// Register a named embedder.
    pub fn register_embedder(&self, name: &str, embedder: Arc<dyn Embedder>) {
        self.embedders.write().insert(name.to_owned(), embedder);
    }

    /// Set the default embedder name. Returns error if the name is not registered.
    pub fn set_default_embedder(&self, name: &str) -> HirnResult<()> {
        if !self.embedders.read().contains_key(name) {
            return Err(HirnError::config(format!(
                "embedder '{name}' not registered"
            )));
        }
        self.defaults.write().embedder = Some(name.to_owned());
        Ok(())
    }

    /// Get the default embedder, if one is configured.
    pub fn embedder(&self) -> Option<Arc<dyn Embedder>> {
        let defaults = self.defaults.read();
        let name = defaults.embedder.as_deref()?;
        self.embedders.read().get(name).cloned()
    }

    /// Look up an embedder by name.
    pub fn embedder_by_name(&self, name: &str) -> Option<Arc<dyn Embedder>> {
        self.embedders.read().get(name).cloned()
    }

    // ── Tokenizer ────────────────────────────────────────────────────

    /// Register a named tokenizer.
    pub fn register_tokenizer(&self, name: &str, tokenizer: Arc<dyn Tokenizer>) {
        self.tokenizers.write().insert(name.to_owned(), tokenizer);
    }

    /// Set the default tokenizer name.
    pub fn set_default_tokenizer(&self, name: &str) -> HirnResult<()> {
        if !self.tokenizers.read().contains_key(name) {
            return Err(HirnError::config(format!(
                "tokenizer '{name}' not registered"
            )));
        }
        self.defaults.write().tokenizer = Some(name.to_owned());
        Ok(())
    }

    /// Get the default tokenizer.
    pub fn tokenizer(&self) -> Option<Arc<dyn Tokenizer>> {
        let defaults = self.defaults.read();
        let name = defaults.tokenizer.as_deref()?;
        self.tokenizers.read().get(name).cloned()
    }

    /// Look up a tokenizer by name.
    pub fn tokenizer_by_name(&self, name: &str) -> Option<Arc<dyn Tokenizer>> {
        self.tokenizers.read().get(name).cloned()
    }

    // ── Reranker ─────────────────────────────────────────────────────

    /// Register a named reranker.
    pub fn register_reranker(&self, name: &str, reranker: Arc<dyn Reranker>) {
        self.rerankers.write().insert(name.to_owned(), reranker);
    }

    /// Set the default reranker name.
    pub fn set_default_reranker(&self, name: &str) -> HirnResult<()> {
        if !self.rerankers.read().contains_key(name) {
            return Err(HirnError::config(format!(
                "reranker '{name}' not registered"
            )));
        }
        self.defaults.write().reranker = Some(name.to_owned());
        Ok(())
    }

    /// Get the default reranker.
    pub fn reranker(&self) -> Option<Arc<dyn Reranker>> {
        let defaults = self.defaults.read();
        let name = defaults.reranker.as_deref()?;
        self.rerankers.read().get(name).cloned()
    }

    /// Look up a reranker by name.
    pub fn reranker_by_name(&self, name: &str) -> Option<Arc<dyn Reranker>> {
        self.rerankers.read().get(name).cloned()
    }

    // ── LLM ──────────────────────────────────────────────────────────

    /// Register a named LLM provider.
    pub fn register_llm(&self, name: &str, llm: Arc<dyn LlmProvider>) {
        self.llms.write().insert(name.to_owned(), llm);
    }

    /// Set the default LLM name.
    pub fn set_default_llm(&self, name: &str) -> HirnResult<()> {
        if !self.llms.read().contains_key(name) {
            return Err(HirnError::config(format!(
                "llm provider '{name}' not registered"
            )));
        }
        self.defaults.write().llm = Some(name.to_owned());
        Ok(())
    }

    /// Get the default LLM provider.
    pub fn llm(&self) -> Option<Arc<dyn LlmProvider>> {
        let defaults = self.defaults.read();
        let name = defaults.llm.as_deref()?;
        self.llms.read().get(name).cloned()
    }

    /// Look up an LLM provider by name.
    pub fn llm_by_name(&self, name: &str) -> Option<Arc<dyn LlmProvider>> {
        self.llms.read().get(name).cloned()
    }

    // ── Environment discovery ────────────────────────────────────────

    /// Auto-discover providers from environment variables.
    ///
    /// Recognized variables:
    /// - `OPENAI_API_KEY` → registers OpenAI embedder + LLM (if `openai` features enabled)
    /// - `OLLAMA_HOST` → registers Ollama embedder + LLM (if `ollama` features enabled)
    /// - `ANTHROPIC_API_KEY` → registers Anthropic LLM (if `anthropic` feature enabled)
    ///
    /// Falls back to `PseudoEmbedder` + provider-default tokenizer + `MockLlmProvider`
    /// when no keys are found.
    pub fn from_env() -> Self {
        let reg = Self::with_fallbacks();
        Self::populate_from_env(&reg);

        reg
    }

    /// Auto-discover providers from environment variables without registering
    /// pseudo/mock/noop fallbacks.
    pub fn from_env_strict() -> Self {
        let reg = Self::new();
        Self::populate_from_env(&reg);

        reg
    }

    // ── Config-driven construction ───────────────────────────────────

    /// Parse a TOML string into a [`ProviderConfig`] and build a registry.
    ///
    /// Environment variable references (`{ env = "VAR" }`) are resolved at
    /// call time.
    ///
    /// # Errors
    ///
    /// Returns an error if the TOML is invalid, a provider type is unknown,
    /// or an environment variable reference cannot be resolved.
    pub fn from_toml(toml_str: &str) -> HirnResult<Self> {
        let config: ProviderConfig = toml::from_str(toml_str)
            .map_err(|e| HirnError::config(format!("invalid provider TOML: {e}")))?;
        Self::from_config(&config)
    }

    /// Build a registry from a [`ProviderConfig`].
    ///
    /// Each provider entry is constructed according to its `type` field.
    /// Environment variable references are resolved at call time.
    /// Fallback providers (pseudo, estimating, noop, mock) are always
    /// registered; config entries override them.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - A provider `type` is unknown or not enabled via feature flag
    /// - A required field is missing (e.g. `api_key` for remote providers)
    /// - An environment variable reference cannot be resolved
    /// - A default name references a provider that was not configured
    pub fn from_config(config: &ProviderConfig) -> HirnResult<Self> {
        let reg = Self::with_fallbacks();

        // ── Embedders ────────────────────────────────────────────────
        for (name, cfg) in &config.providers.embedder {
            let embedder: Arc<dyn Embedder> = Self::build_embedder(name, cfg)?;
            reg.register_embedder(name, embedder);
        }

        // ── LLMs ─────────────────────────────────────────────────────
        for (name, cfg) in &config.providers.llm {
            let llm: Arc<dyn LlmProvider> = Self::build_llm(name, cfg)?;
            reg.register_llm(name, llm);
        }

        // ── Rerankers ────────────────────────────────────────────────
        for (name, cfg) in &config.providers.reranker {
            let reranker: Arc<dyn Reranker> = Self::build_reranker(name, cfg)?;
            reg.register_reranker(name, reranker);
        }

        // ── Tokenizers ───────────────────────────────────────────────
        for (name, cfg) in &config.providers.tokenizer {
            let tokenizer: Arc<dyn Tokenizer> = Self::build_tokenizer(name, cfg)?;
            reg.register_tokenizer(name, tokenizer);
        }

        // ── Defaults ─────────────────────────────────────────────────
        if let Some(ref name) = config.defaults.embedder {
            reg.set_default_embedder(name)?;
        }
        if let Some(ref name) = config.defaults.tokenizer {
            reg.set_default_tokenizer(name)?;
        }
        if let Some(ref name) = config.defaults.reranker {
            reg.set_default_reranker(name)?;
        }
        if let Some(ref name) = config.defaults.llm {
            reg.set_default_llm(name)?;
        }

        Ok(reg)
    }

    /// Apply a [`ProviderConfig`] on top of an existing registry.
    ///
    /// Providers from the config are registered (overriding any with the same
    /// name). Defaults from the config override existing defaults.
    pub fn apply_config(&self, config: &ProviderConfig) -> HirnResult<()> {
        for (name, cfg) in &config.providers.embedder {
            self.register_embedder(name, Self::build_embedder(name, cfg)?);
        }
        for (name, cfg) in &config.providers.llm {
            self.register_llm(name, Self::build_llm(name, cfg)?);
        }
        for (name, cfg) in &config.providers.reranker {
            self.register_reranker(name, Self::build_reranker(name, cfg)?);
        }
        for (name, cfg) in &config.providers.tokenizer {
            self.register_tokenizer(name, Self::build_tokenizer(name, cfg)?);
        }
        if let Some(ref name) = config.defaults.embedder {
            self.set_default_embedder(name)?;
        }
        if let Some(ref name) = config.defaults.tokenizer {
            self.set_default_tokenizer(name)?;
        }
        if let Some(ref name) = config.defaults.reranker {
            self.set_default_reranker(name)?;
        }
        if let Some(ref name) = config.defaults.llm {
            self.set_default_llm(name)?;
        }
        Ok(())
    }

    // ── Provider builders (private) ──────────────────────────────────

    #[cfg(feature = "openai")]
    fn build_openai_embedder_with<F>(
        name: &str,
        cfg: &EmbedderConfig,
        constructor: F,
    ) -> HirnResult<Arc<dyn Embedder>>
    where
        F: FnOnce(String, &str, usize) -> HirnResult<hirn_provider::OpenAIEmbedder>,
    {
        let api_key = cfg
            .api_key
            .as_ref()
            .ok_or_else(|| {
                HirnError::config(format!("embedder '{name}': 'api_key' required for openai"))
            })?
            .resolve()?;
        let model = cfg.model.as_deref().unwrap_or("text-embedding-3-small");
        let dims = cfg.dimensions.unwrap_or(1536);
        let mut embedder = constructor(api_key, model, dims).map_err(|err| {
            HirnError::config(format!(
                "embedder '{name}': failed to initialize openai client: {err}"
            ))
        })?;
        if let Some(ref url) = cfg.base_url {
            embedder = embedder.with_base_url(url).map_err(|err| {
                HirnError::config(format!("embedder '{name}': invalid base_url: {err}"))
            })?;
        }
        Ok(Arc::new(embedder))
    }

    fn build_embedder(name: &str, cfg: &EmbedderConfig) -> HirnResult<Arc<dyn Embedder>> {
        match cfg.r#type.as_str() {
            "pseudo" => {
                let dims = cfg.dimensions.unwrap_or(384);
                Ok(Arc::new(hirn_provider::PseudoEmbedder::new(dims)))
            }
            #[cfg(feature = "openai")]
            "openai" => Self::build_openai_embedder_with(name, cfg, |api_key, model, dims| {
                hirn_provider::OpenAIEmbedder::new(api_key, model, dims)
            }),
            #[cfg(feature = "ollama")]
            "ollama" => {
                let model = cfg.model.as_deref().unwrap_or("nomic-embed-text");
                let dims = cfg.dimensions.unwrap_or(768);
                let mut embedder =
                    hirn_provider::OllamaEmbedder::new(model, dims).map_err(|err| {
                        HirnError::config(format!(
                            "embedder '{name}': failed to initialize ollama client: {err}"
                        ))
                    })?;
                if let Some(ref url) = cfg.base_url {
                    embedder = embedder.with_host(url).map_err(|err| {
                        HirnError::config(format!("embedder '{name}': invalid base_url: {err}"))
                    })?;
                }
                Ok(Arc::new(embedder))
            }
            #[cfg(feature = "cohere")]
            "cohere" => {
                let api_key = cfg
                    .api_key
                    .as_ref()
                    .ok_or_else(|| {
                        HirnError::config(format!(
                            "embedder '{name}': 'api_key' required for cohere"
                        ))
                    })?
                    .resolve()?;
                let model = cfg.model.as_deref().unwrap_or("embed-english-v3.0");
                let dims = cfg.dimensions.unwrap_or(1024);
                let mut embedder = hirn_provider::CohereEmbedder::new(api_key, model, dims)
                    .map_err(|err| {
                        HirnError::config(format!(
                            "embedder '{name}': failed to initialize cohere client: {err}"
                        ))
                    })?;
                if let Some(ref url) = cfg.base_url {
                    embedder = embedder.with_base_url(url).map_err(|err| {
                        HirnError::config(format!("embedder '{name}': invalid base_url: {err}"))
                    })?;
                }
                Ok(Arc::new(embedder))
            }
            #[cfg(feature = "voyage")]
            "voyage" => {
                let api_key = cfg
                    .api_key
                    .as_ref()
                    .ok_or_else(|| {
                        HirnError::config(format!(
                            "embedder '{name}': 'api_key' required for voyage"
                        ))
                    })?
                    .resolve()?;
                let model = cfg.model.as_deref().unwrap_or("voyage-3");
                let dims = cfg.dimensions.unwrap_or(1024);
                let mut embedder = hirn_provider::VoyageEmbedder::new(api_key, model, dims)
                    .map_err(|err| {
                        HirnError::config(format!(
                            "embedder '{name}': failed to initialize voyage client: {err}"
                        ))
                    })?;
                if let Some(ref url) = cfg.base_url {
                    embedder = embedder.with_base_url(url).map_err(|err| {
                        HirnError::config(format!("embedder '{name}': invalid base_url: {err}"))
                    })?;
                }
                Ok(Arc::new(embedder))
            }
            other => Err(HirnError::config(format!(
                "embedder '{name}': unknown type '{other}'"
            ))),
        }
    }

    fn build_llm(name: &str, cfg: &LlmConfig) -> HirnResult<Arc<dyn LlmProvider>> {
        match cfg.r#type.as_str() {
            "mock" => Ok(Arc::new(hirn_provider::MockLlmProvider::new(name))),
            #[cfg(feature = "openai")]
            "openai" => {
                let api_key = cfg
                    .api_key
                    .as_ref()
                    .ok_or_else(|| {
                        HirnError::config(format!("llm '{name}': 'api_key' required for openai"))
                    })?
                    .resolve()?;
                let model = cfg.model.as_deref().unwrap_or("gpt-4o-mini");
                let mut provider =
                    hirn_provider::OpenAILlmProvider::new(api_key, model).map_err(|err| {
                        HirnError::config(format!(
                            "llm '{name}': failed to initialize openai client: {err}"
                        ))
                    })?;
                if let Some(ref url) = cfg.base_url {
                    provider = provider.with_base_url(url).map_err(|err| {
                        HirnError::config(format!("llm '{name}': invalid base_url: {err}"))
                    })?;
                }
                Ok(Arc::new(provider))
            }
            #[cfg(feature = "ollama")]
            "ollama" => {
                let model = cfg.model.as_deref().unwrap_or("llama3.1");
                let mut provider = hirn_provider::OllamaLlmProvider::new(model).map_err(|err| {
                    HirnError::config(format!(
                        "llm '{name}': failed to initialize ollama client: {err}"
                    ))
                })?;
                if let Some(ref url) = cfg.base_url {
                    provider = provider.with_host(url).map_err(|err| {
                        HirnError::config(format!("llm '{name}': invalid base_url: {err}"))
                    })?;
                }
                Ok(Arc::new(provider))
            }
            #[cfg(feature = "anthropic")]
            "anthropic" => {
                let api_key = cfg
                    .api_key
                    .as_ref()
                    .ok_or_else(|| {
                        HirnError::config(format!("llm '{name}': 'api_key' required for anthropic"))
                    })?
                    .resolve()?;
                let mut provider =
                    hirn_provider::AnthropicProvider::new(api_key).map_err(|err| {
                        HirnError::config(format!(
                            "llm '{name}': failed to initialize anthropic client: {err}"
                        ))
                    })?;
                if let Some(ref model) = cfg.model {
                    provider = provider.with_model(model);
                }
                if let Some(ref url) = cfg.base_url {
                    provider = provider.with_base_url(url).map_err(|err| {
                        HirnError::config(format!("llm '{name}': invalid base_url: {err}"))
                    })?;
                }
                Ok(Arc::new(provider))
            }
            other => Err(HirnError::config(format!(
                "llm '{name}': unknown type '{other}'"
            ))),
        }
    }

    fn build_reranker(name: &str, cfg: &RerankerConfig) -> HirnResult<Arc<dyn Reranker>> {
        match cfg.r#type.as_str() {
            "noop" => Ok(Arc::new(hirn_core::embed::NoopReranker)),
            #[cfg(feature = "cohere")]
            "cohere" => {
                let api_key = cfg
                    .api_key
                    .as_ref()
                    .ok_or_else(|| {
                        HirnError::config(format!(
                            "reranker '{name}': 'api_key' required for cohere"
                        ))
                    })?
                    .resolve()?;
                let mut reranker = hirn_provider::CohereReranker::new(api_key).map_err(|err| {
                    HirnError::config(format!(
                        "reranker '{name}': failed to initialize cohere client: {err}"
                    ))
                })?;
                if let Some(ref model) = cfg.model {
                    reranker = reranker.with_model(model);
                }
                if let Some(ref url) = cfg.base_url {
                    reranker = reranker.with_base_url(url).map_err(|err| {
                        HirnError::config(format!("reranker '{name}': invalid base_url: {err}"))
                    })?;
                }
                Ok(Arc::new(reranker))
            }
            #[cfg(feature = "cross-encoder")]
            "cross-encoder" => {
                let reranker =
                    hirn_provider::CrossEncoderReranker::default_model().map_err(|e| {
                        HirnError::config(format!(
                            "reranker '{name}': failed to load cross-encoder: {e}"
                        ))
                    })?;
                Ok(Arc::new(reranker))
            }
            other => Err(HirnError::config(format!(
                "reranker '{name}': unknown type '{other}'"
            ))),
        }
    }

    fn build_tokenizer(name: &str, cfg: &TokenizerConfig) -> HirnResult<Arc<dyn Tokenizer>> {
        hirn_provider::build_tokenizer(&cfg.r#type, cfg.model.as_deref(), cfg.max_length)
            .map_err(|e| HirnError::config(format!("tokenizer '{name}': {e}")))
    }
}

impl Default for ProviderRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// Satisfy Send + Sync requirement (parking_lot RwLock is Send + Sync).
// The trait objects are Send + Sync by trait bounds.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_and_lookup_embedder() {
        let reg = ProviderRegistry::new();
        reg.register_embedder("pseudo", Arc::new(hirn_provider::PseudoEmbedder::new(64)));
        assert!(reg.embedder_by_name("pseudo").is_some());
        assert!(reg.embedder_by_name("unknown").is_none());
    }

    #[test]
    fn default_embedder_requires_registration() {
        let reg = ProviderRegistry::new();
        assert!(reg.set_default_embedder("missing").is_err());
    }

    #[test]
    fn default_embedder_lookup() {
        let reg = ProviderRegistry::new();
        reg.register_embedder("pseudo", Arc::new(hirn_provider::PseudoEmbedder::new(64)));
        reg.set_default_embedder("pseudo").unwrap();
        assert!(reg.embedder().is_some());
    }

    #[test]
    fn no_default_embedder_returns_none() {
        let reg = ProviderRegistry::new();
        assert!(reg.embedder().is_none());
    }

    #[test]
    fn register_and_lookup_llm() {
        let reg = ProviderRegistry::new();
        reg.register_llm(
            "mock",
            Arc::new(hirn_provider::MockLlmProvider::new("test")),
        );
        assert!(reg.llm_by_name("mock").is_some());
    }

    #[test]
    fn hot_swap_embedder() {
        let reg = ProviderRegistry::new();
        let e1 = Arc::new(hirn_provider::PseudoEmbedder::new(64));
        let e2 = Arc::new(hirn_provider::PseudoEmbedder::new(128));
        reg.register_embedder("e", e1);
        reg.set_default_embedder("e").unwrap();
        assert_eq!(reg.embedder().unwrap().dimensions(), 64);
        // Hot-swap
        reg.register_embedder("e", e2);
        assert_eq!(reg.embedder().unwrap().dimensions(), 128);
    }

    #[test]
    fn from_env_creates_fallbacks() {
        // In CI/test without OPENAI_API_KEY, should get fallbacks.
        let reg = ProviderRegistry::from_env();
        assert!(reg.embedder().is_some());
        assert!(reg.tokenizer().is_some());
        assert!(reg.reranker().is_some());
        assert!(reg.llm().is_some());
    }

    #[test]
    fn from_env_strict_omits_fallback_embedder_when_no_real_embedder_is_configured() {
        if [
            "OPENAI_API_KEY",
            "OLLAMA_HOST",
            "COHERE_API_KEY",
            "VOYAGE_API_KEY",
        ]
        .iter()
        .any(|key| std::env::var(key).is_ok())
        {
            return;
        }

        let reg = ProviderRegistry::from_env_strict();
        assert!(reg.embedder().is_none());
    }

    #[test]
    fn registry_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ProviderRegistry>();
    }

    #[cfg(feature = "openai")]
    #[test]
    fn openai_auto_discovery_continues_when_embedder_init_fails() {
        let reg = ProviderRegistry::with_fallbacks();

        ProviderRegistry::register_openai_from_key(
            &reg,
            "sk-test".into(),
            |_api_key| Err(HirnError::provider("synthetic openai embedder failure")),
            |_api_key| Ok(Arc::new(hirn_provider::MockLlmProvider::new("openai"))),
        );

        assert_eq!(reg.defaults.read().embedder.as_deref(), Some("pseudo"));
        assert_eq!(reg.embedder().unwrap().dimensions(), 384);
        assert!(reg.embedder_by_name("openai").is_none());
        assert_eq!(reg.defaults.read().llm.as_deref(), Some("openai"));
        assert!(reg.llm_by_name("openai").is_some());
    }

    #[cfg(feature = "openai")]
    #[test]
    fn openai_config_constructor_failure_returns_structured_error() {
        let cfg = EmbedderConfig {
            r#type: "openai".into(),
            model: Some("text-embedding-3-small".into()),
            dimensions: Some(1536),
            api_key: Some(ApiKeySource::Literal("sk-test".into())),
            base_url: None,
        };

        let err = ProviderRegistry::build_openai_embedder_with(
            "broken-openai",
            &cfg,
            |_api_key, _model, _dims| Err(HirnError::provider("synthetic constructor failure")),
        );

        let err = match err {
            Ok(_) => panic!("expected constructor failure"),
            Err(err) => err,
        };

        match err {
            HirnError::InvalidInput(message) => {
                assert!(message.contains("embedder 'broken-openai'"));
                assert!(message.contains("failed to initialize openai client"));
                assert!(message.contains("synthetic constructor failure"));
            }
            other => panic!("expected invalid input, got {other:?}"),
        }
    }

    #[test]
    fn register_and_lookup_reranker() {
        let reg = ProviderRegistry::new();
        reg.register_reranker("noop", Arc::new(hirn_core::embed::NoopReranker));
        reg.set_default_reranker("noop").unwrap();
        assert!(reg.reranker().is_some());
    }

    #[test]
    fn register_and_lookup_tokenizer() {
        let reg = ProviderRegistry::new();
        reg.register_tokenizer("est", Arc::new(EstimatingTokenizer));
        reg.set_default_tokenizer("est").unwrap();
        assert!(reg.tokenizer().is_some());
    }

    // ── Config-driven tests ──────────────────────────────────────────

    #[test]
    fn from_toml_pseudo_and_estimating() {
        let toml = r#"
[providers.embedder.my_embed]
type = "pseudo"
dimensions = 256

[providers.tokenizer.my_tok]
type = "estimating"

[providers.llm.my_llm]
type = "mock"

[providers.reranker.my_reranker]
type = "noop"

[defaults]
embedder = "my_embed"
tokenizer = "my_tok"
llm = "my_llm"
reranker = "my_reranker"
"#;
        let reg = ProviderRegistry::from_toml(toml).unwrap();
        assert_eq!(reg.embedder().unwrap().dimensions(), 256);
        assert!(reg.tokenizer().is_some());
        assert!(reg.llm().is_some());
        assert!(reg.reranker().is_some());
    }

    #[test]
    fn from_toml_unknown_embedder_type_error() {
        let toml = r#"
[providers.embedder.bad]
type = "nonexistent_provider"
"#;
        let err = ProviderRegistry::from_toml(toml).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("unknown type") && msg.contains("nonexistent_provider"),
            "should mention unknown type: {msg}"
        );
    }

    #[test]
    fn from_toml_unknown_llm_type_error() {
        let toml = r#"
[providers.llm.bad]
type = "gpt-magic"
"#;
        let err = ProviderRegistry::from_toml(toml).unwrap_err();
        assert!(err.to_string().contains("unknown type"));
    }

    #[test]
    fn from_toml_unknown_reranker_type_error() {
        let toml = r#"
[providers.reranker.bad]
type = "magic-reranker"
"#;
        let err = ProviderRegistry::from_toml(toml).unwrap_err();
        assert!(err.to_string().contains("unknown type"));
    }

    #[test]
    fn from_toml_unknown_tokenizer_type_error() {
        let toml = r#"
[providers.tokenizer.bad]
type = "magic-tokenizer"
"#;
        let err = ProviderRegistry::from_toml(toml).unwrap_err();
        assert!(err.to_string().contains("unknown tokenizer type"));
    }

    #[test]
    fn from_toml_invalid_toml_syntax_error() {
        let toml = "this is not [valid toml";
        let err = ProviderRegistry::from_toml(toml).unwrap_err();
        assert!(
            err.to_string().contains("invalid provider TOML"),
            "error: {}",
            err,
        );
    }

    #[test]
    fn from_toml_env_var_literal_key() {
        // Test that literal API keys work in config (no env var needed).
        let toml = r#"
[providers.embedder.pseudo_env]
type = "pseudo"
dimensions = 128
"#;
        let reg = ProviderRegistry::from_toml(toml).unwrap();
        assert!(reg.embedder_by_name("pseudo_env").is_some());
    }

    #[test]
    fn missing_env_var_error() {
        // Use a var name that is very unlikely to be set.
        let source = ApiKeySource::Env {
            env: "HIRN_NONEXISTENT_VAR_42_TEST".into(),
        };
        let err = source.resolve().unwrap_err();
        assert!(
            err.to_string().contains("HIRN_NONEXISTENT_VAR_42_TEST"),
            "error should name the variable: {err}"
        );
    }

    #[test]
    fn api_key_source_literal_resolves() {
        let source = ApiKeySource::Literal("my-key".into());
        assert_eq!(source.resolve().unwrap(), "my-key");
    }

    #[test]
    fn api_key_source_env_resolves() {
        // Use HOME which is always set on macOS/Linux.
        let source = ApiKeySource::Env { env: "HOME".into() };
        let resolved = source.resolve().unwrap();
        assert!(
            !resolved.is_empty(),
            "HOME should resolve to a non-empty string"
        );
    }

    #[test]
    fn api_key_source_deserialize_literal() {
        #[derive(serde::Deserialize)]
        struct W {
            key: ApiKeySource,
        }
        let w: W = toml::from_str(r#"key = "my-literal-key""#).unwrap();
        assert_eq!(w.key, ApiKeySource::Literal("my-literal-key".into()));
    }

    #[test]
    fn api_key_source_deserialize_env() {
        #[derive(serde::Deserialize)]
        struct W {
            key: ApiKeySource,
        }
        let w: W = toml::from_str(r#"key = { env = "MY_VAR" }"#).unwrap();
        assert_eq!(
            w.key,
            ApiKeySource::Env {
                env: "MY_VAR".into()
            }
        );
    }

    #[test]
    fn from_toml_default_references_unregistered_provider_error() {
        let toml = r#"
[defaults]
embedder = "nonexistent"
"#;
        let err = ProviderRegistry::from_toml(toml).unwrap_err();
        assert!(err.to_string().contains("not registered"), "error: {}", err);
    }

    #[cfg(feature = "tiktoken")]
    #[test]
    fn from_toml_tiktoken_tokenizer() {
        let toml = r#"
[providers.tokenizer.tiktoken]
type = "tiktoken"
model = "cl100k_base"

[defaults]
tokenizer = "tiktoken"
"#;
        let reg = ProviderRegistry::from_toml(toml).unwrap();
        let tok = reg.tokenizer().unwrap();
        assert!(tok.count_tokens("hello world") > 0);
    }

    #[cfg(feature = "tiktoken")]
    #[test]
    fn from_toml_tiktoken_invalid_model_error() {
        let toml = r#"
[providers.tokenizer.bad]
type = "tiktoken"
model = "gpt-99-turbo"
"#;
        let err = ProviderRegistry::from_toml(toml).unwrap_err();
        assert!(err.to_string().contains("unknown tiktoken model"));
    }

    #[test]
    fn from_toml_empty_config_uses_fallbacks() {
        let reg = ProviderRegistry::from_toml("").unwrap();
        // Fallbacks should be registered.
        assert!(reg.embedder().is_some());
        assert!(reg.tokenizer().is_some());
        assert!(reg.reranker().is_some());
        assert!(reg.llm().is_some());
    }

    #[test]
    fn from_config_and_from_env_combined() {
        // from_env creates a registry with fallbacks.
        let reg = ProviderRegistry::from_env();
        assert!(reg.embedder().is_some());

        // Apply config on top — add a custom pseudo embedder.
        let config = ProviderConfig {
            providers: ProvidersSection {
                embedder: {
                    let mut m = HashMap::new();
                    m.insert(
                        "custom".into(),
                        EmbedderConfig {
                            r#type: "pseudo".into(),
                            model: None,
                            dimensions: Some(999),
                            api_key: None,
                            base_url: None,
                        },
                    );
                    m
                },
                ..Default::default()
            },
            defaults: DefaultsConfig {
                embedder: Some("custom".into()),
                ..Default::default()
            },
        };
        reg.apply_config(&config).unwrap();
        assert_eq!(reg.embedder().unwrap().dimensions(), 999);
    }

    #[test]
    fn from_toml_multiple_embedders() {
        let toml = r#"
[providers.embedder.small]
type = "pseudo"
dimensions = 128

[providers.embedder.large]
type = "pseudo"
dimensions = 2048

[defaults]
embedder = "large"
"#;
        let reg = ProviderRegistry::from_toml(toml).unwrap();
        assert_eq!(reg.embedder().unwrap().dimensions(), 2048);
        assert_eq!(reg.embedder_by_name("small").unwrap().dimensions(), 128);
    }

    #[test]
    fn provider_config_deserialize_full_example() {
        let toml = r#"
[providers.embedder.openai]
type = "openai"
model = "text-embedding-3-small"
api_key = { env = "OPENAI_API_KEY" }
dimensions = 1536

[providers.embedder.local]
type = "pseudo"
dimensions = 384

[providers.llm.claude]
type = "anthropic"
model = "claude-sonnet-4-20250514"
api_key = { env = "ANTHROPIC_API_KEY" }

[providers.llm.fallback]
type = "mock"

[providers.reranker.noop]
type = "noop"

[providers.tokenizer.default]
type = "estimating"

[providers.tokenizer.tiktoken]
type = "tiktoken"
model = "cl100k_base"

[defaults]
embedder = "local"
llm = "fallback"
reranker = "noop"
tokenizer = "default"
"#;
        // Parse only — don't resolve env vars (they may not be set).
        let config: ProviderConfig = toml::from_str(toml).unwrap();
        assert_eq!(config.providers.embedder.len(), 2);
        assert_eq!(config.providers.llm.len(), 2);
        assert_eq!(config.providers.reranker.len(), 1);
        assert_eq!(config.providers.tokenizer.len(), 2);
        assert_eq!(config.defaults.embedder.as_deref(), Some("local"));
        assert_eq!(config.defaults.llm.as_deref(), Some("fallback"));
    }
}
