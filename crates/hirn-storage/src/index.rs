use crate::store::{IndexConfig, IndexParams, IndexType};

impl IndexConfig {
    /// Create a vector index config (IVF-HNSW-SQ by default).
    pub fn vector(column: impl Into<String>) -> Self {
        Self {
            columns: vec![column.into()],
            index_type: IndexType::IvfHnswSq,
            params: IndexParams::default(),
            replace: true,
        }
    }

    /// Create a BM25 full-text search index config.
    pub fn bm25(columns: Vec<String>) -> Self {
        Self {
            columns,
            index_type: IndexType::Bm25,
            params: IndexParams::default(),
            replace: true,
        }
    }

    /// Create a BTree scalar index config.
    pub fn btree(column: impl Into<String>) -> Self {
        Self {
            columns: vec![column.into()],
            index_type: IndexType::BTree,
            params: IndexParams::default(),
            replace: true,
        }
    }

    /// Create a Bitmap scalar index config.
    pub fn bitmap(column: impl Into<String>) -> Self {
        Self {
            columns: vec![column.into()],
            index_type: IndexType::Bitmap,
            params: IndexParams::default(),
            replace: true,
        }
    }

    /// Set the index type.
    pub fn with_type(mut self, index_type: IndexType) -> Self {
        self.index_type = index_type;
        self
    }

    /// Set index parameters.
    pub fn with_params(mut self, params: IndexParams) -> Self {
        self.params = params;
        self
    }

    /// Whether to replace an existing index.
    pub fn with_replace(mut self, replace: bool) -> Self {
        self.replace = replace;
        self
    }
}
