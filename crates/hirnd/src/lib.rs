pub mod auth;
pub mod config;
pub mod convert;
pub mod coordination;
#[cfg(feature = "serverless")]
pub mod dynamo;
pub mod grpc;
pub mod http;
pub mod mcp;
pub mod raft;
pub mod realm;
pub mod throttle;
pub mod tls;
pub mod watch;

/// Default embedding dimensions. Used as fallback when no explicit value is
/// configured (matches common models like `text-embedding-3-small` at 768-d).
pub const DEFAULT_EMBEDDING_DIMS: usize = 768;

/// Generated protobuf/gRPC types.
pub mod proto {
    tonic::include_proto!("hirn.v1");
}
