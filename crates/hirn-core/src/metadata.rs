use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// Metadata value that supports both bincode and JSON serialization.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MetadataValue {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    String(String),
    /// F-79: Array of metadata values (e.g., tags, multi-value fields).
    List(Vec<Self>),
    /// F-79: Nested key-value map (e.g., structured sub-objects).
    Map(BTreeMap<String, Self>),
}

impl From<&str> for MetadataValue {
    fn from(s: &str) -> Self {
        Self::String(s.to_string())
    }
}

impl From<String> for MetadataValue {
    fn from(s: String) -> Self {
        Self::String(s)
    }
}

impl From<bool> for MetadataValue {
    fn from(b: bool) -> Self {
        Self::Bool(b)
    }
}

impl From<i64> for MetadataValue {
    fn from(n: i64) -> Self {
        Self::Int(n)
    }
}

impl TryFrom<f64> for MetadataValue {
    type Error = &'static str;

    fn try_from(n: f64) -> Result<Self, Self::Error> {
        if n.is_finite() {
            Ok(Self::Float(n))
        } else {
            Err("non-finite f64 cannot be stored as metadata")
        }
    }
}

impl MetadataValue {
    /// Create a float value. Returns `None` if the value is NaN or infinite.
    pub const fn float(n: f64) -> Option<Self> {
        if n.is_finite() {
            Some(Self::Float(n))
        } else {
            None
        }
    }
}

impl From<Vec<Self>> for MetadataValue {
    fn from(v: Vec<Self>) -> Self {
        Self::List(v)
    }
}

impl From<BTreeMap<String, Self>> for MetadataValue {
    fn from(m: BTreeMap<String, Self>) -> Self {
        Self::Map(m)
    }
}

/// A metadata map that supports bincode round-tripping.
pub type Metadata = BTreeMap<String, MetadataValue>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serde_round_trip() {
        let mut m = Metadata::new();
        m.insert("str".into(), MetadataValue::String("hello".into()));
        m.insert("num".into(), MetadataValue::Int(42));
        m.insert("bool".into(), MetadataValue::Bool(true));
        m.insert("null".into(), MetadataValue::Null);
        m.insert("float".into(), MetadataValue::Float(1.23));
        m.insert(
            "list".into(),
            MetadataValue::List(vec![
                MetadataValue::Int(1),
                MetadataValue::String("a".into()),
            ]),
        );
        let mut sub = BTreeMap::new();
        sub.insert("nested".into(), MetadataValue::Bool(true));
        m.insert("map".into(), MetadataValue::Map(sub));

        let bytes = bincode::serialize(&m).unwrap();
        let back: Metadata = bincode::deserialize(&bytes).unwrap();
        assert_eq!(m, back);
    }

    #[test]
    fn from_impls() {
        let _s: MetadataValue = "hello".into();
        let _b: MetadataValue = true.into();
        let _n: MetadataValue = 42i64.into();
        let _f: MetadataValue = MetadataValue::try_from(1.23f64).unwrap();
        assert!(MetadataValue::try_from(f64::NAN).is_err());
        assert!(MetadataValue::try_from(f64::INFINITY).is_err());
        let _l: MetadataValue = vec![MetadataValue::Int(1)].into();
    }
}
