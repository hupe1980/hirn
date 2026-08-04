use hirn_core::{HirnError, HirnResult};
use serde::de::DeserializeOwned;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::sync::Arc;

const MAX_PROVIDER_RESPONSE_BYTES: usize = 16 * 1024 * 1024;
const MAX_PROVIDER_ERROR_BYTES: usize = 64 * 1024;

#[derive(Debug)]
struct PublicProviderDnsResolver;

impl reqwest::dns::Resolve for PublicProviderDnsResolver {
    fn resolve(&self, name: reqwest::dns::Name) -> reqwest::dns::Resolving {
        let host = name.as_str().to_owned();
        Box::pin(async move {
            let addresses = tokio::net::lookup_host((host.as_str(), 0))
                .await?
                .collect::<Vec<_>>();
            validate_resolved_addresses(&host, &addresses)?;
            Ok(Box::new(addresses.into_iter()) as reqwest::dns::Addrs)
        })
    }
}

pub(crate) fn secure_provider_client_builder(
    builder: reqwest::ClientBuilder,
) -> reqwest::ClientBuilder {
    builder
        .redirect(reqwest::redirect::Policy::none())
        .no_proxy()
        .dns_resolver(Arc::new(PublicProviderDnsResolver))
}

fn validate_resolved_addresses(host: &str, addresses: &[SocketAddr]) -> std::io::Result<()> {
    if addresses.is_empty() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::AddrNotAvailable,
            format!("provider host '{host}' resolved to no addresses"),
        ));
    }

    let localhost = host.eq_ignore_ascii_case("localhost");
    if addresses.iter().any(|address| {
        (!localhost && address.ip().is_loopback()) || is_forbidden_provider_address(address.ip())
    }) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            format!(
                "provider host '{host}' resolved to a private, loopback, link-local, multicast, or unspecified address"
            ),
        ));
    }

    Ok(())
}

pub(crate) async fn read_error_response(mut response: reqwest::Response) -> String {
    let mut body = Vec::new();
    while let Ok(Some(chunk)) = response.chunk().await {
        let remaining = MAX_PROVIDER_ERROR_BYTES.saturating_sub(body.len());
        if remaining == 0 {
            break;
        }
        body.extend_from_slice(&chunk[..chunk.len().min(remaining)]);
    }
    String::from_utf8_lossy(&body).into_owned()
}

pub(crate) async fn decode_json_response<T: DeserializeOwned>(
    mut response: reqwest::Response,
) -> Result<T, String> {
    if response
        .content_length()
        .is_some_and(|length| length > MAX_PROVIDER_RESPONSE_BYTES as u64)
    {
        return Err(format!(
            "provider response exceeds {MAX_PROVIDER_RESPONSE_BYTES} byte limit"
        ));
    }

    let mut body = Vec::new();
    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|error| format!("failed to read provider response: {error}"))?
    {
        let next_len = body
            .len()
            .checked_add(chunk.len())
            .ok_or_else(|| "provider response size overflow".to_owned())?;
        if next_len > MAX_PROVIDER_RESPONSE_BYTES {
            return Err(format!(
                "provider response exceeds {MAX_PROVIDER_RESPONSE_BYTES} byte limit"
            ));
        }
        body.extend_from_slice(&chunk);
    }

    serde_json::from_slice(&body)
        .map_err(|error| format!("failed to decode provider JSON response: {error}"))
}

pub(crate) fn validate_secret_bearing_base_url(
    provider: &'static str,
    url: &str,
) -> HirnResult<()> {
    validate_provider_base_url(
        provider,
        url,
        "secret-bearing provider traffic requires HTTPS; only loopback HTTP is allowed for local development",
    )
}

#[cfg(any(test, feature = "ollama"))]
pub(crate) fn validate_privacy_bearing_base_url(
    provider: &'static str,
    url: &str,
) -> HirnResult<()> {
    validate_provider_base_url(
        provider,
        url,
        "privacy-bearing provider traffic requires HTTPS; only loopback HTTP is allowed for local development",
    )
}

fn validate_provider_base_url(
    provider: &'static str,
    url: &str,
    http_rejection_reason: &'static str,
) -> HirnResult<()> {
    let parsed = reqwest::Url::parse(url).map_err(|error| {
        invalid_base_url(provider, url, format!("must be an absolute URL: {error}"))
    })?;

    if let Some(host) = parsed.host_str()
        && let Ok(address) = host.trim_matches(['[', ']']).parse::<IpAddr>()
        && is_forbidden_provider_address(address)
    {
        return Err(invalid_base_url(
            provider,
            url,
            "provider endpoints cannot target private, link-local, multicast, or unspecified IP addresses",
        ));
    }

    match parsed.scheme() {
        "https" => Ok(()),
        "http" if is_loopback_http_endpoint(&parsed) => {
            tracing::warn!(
                provider,
                base_url = %url,
                "using plaintext loopback provider endpoint"
            );
            Ok(())
        }
        "http" => Err(invalid_base_url(provider, url, http_rejection_reason)),
        scheme => Err(invalid_base_url(
            provider,
            url,
            format!("unsupported URL scheme '{scheme}'; expected https"),
        )),
    }
}

fn invalid_base_url(provider: &'static str, value: &str, reason: impl Into<String>) -> HirnError {
    HirnError::InvalidConfig {
        field: format!("{provider}.base_url"),
        value: value.to_owned(),
        reason: reason.into(),
    }
}

fn is_loopback_http_endpoint(url: &reqwest::Url) -> bool {
    url.host_str().is_some_and(|host| {
        host.eq_ignore_ascii_case("localhost")
            || host
                .trim_matches(['[', ']'])
                .parse::<IpAddr>()
                .is_ok_and(|address| address.is_loopback())
    })
}

fn is_forbidden_provider_address(address: IpAddr) -> bool {
    match address {
        IpAddr::V4(address) => is_forbidden_provider_ipv4(address),
        IpAddr::V6(address) => is_forbidden_provider_ipv6(address),
    }
}

fn is_forbidden_provider_ipv4(address: Ipv4Addr) -> bool {
    if address.is_loopback() {
        return false;
    }
    let [first, second, ..] = address.octets();
    address.is_private()
        || address.is_link_local()
        || address.is_multicast()
        || address.is_unspecified()
        || address.is_broadcast()
        || (first == 100 && (64..=127).contains(&second))
        || (first == 198 && (18..=19).contains(&second))
}

fn is_forbidden_provider_ipv6(address: Ipv6Addr) -> bool {
    if address.is_loopback() {
        return false;
    }
    let first = address.segments()[0];
    address.is_unspecified()
        || address.is_multicast()
        || (first & 0xfe00) == 0xfc00
        || (first & 0xffc0) == 0xfe80
}

#[cfg(test)]
mod tests {
    use std::net::SocketAddr;

    use super::{
        validate_privacy_bearing_base_url, validate_resolved_addresses,
        validate_secret_bearing_base_url,
    };

    #[test]
    fn allows_https_base_url() {
        validate_secret_bearing_base_url("test", "https://api.example.com/v1").unwrap();
    }

    #[test]
    fn allows_loopback_http_base_url() {
        validate_secret_bearing_base_url("test", "http://127.0.0.1:8080/v1").unwrap();
        validate_secret_bearing_base_url("test", "http://[::1]:8080/v1").unwrap();
        validate_secret_bearing_base_url("test", "http://localhost:8080/v1").unwrap();
    }

    #[test]
    fn rejects_remote_plaintext_base_url() {
        let err = validate_secret_bearing_base_url("test", "http://example.com/v1")
            .expect_err("remote plaintext must be rejected");
        assert!(err.to_string().contains("requires HTTPS"));
    }

    #[test]
    fn rejects_remote_plaintext_privacy_bearing_base_url() {
        let err = validate_privacy_bearing_base_url("test", "http://example.com/v1")
            .expect_err("remote plaintext must be rejected");
        assert!(
            err.to_string()
                .contains("privacy-bearing provider traffic requires HTTPS")
        );
    }

    #[test]
    fn rejects_private_ip_even_over_https() {
        let err = validate_secret_bearing_base_url("test", "https://10.0.0.7/v1")
            .expect_err("private provider endpoint must be rejected");
        assert!(err.to_string().contains("private"));

        let err = validate_secret_bearing_base_url("test", "https://[fe80::1]/v1")
            .expect_err("link-local provider endpoint must be rejected");
        assert!(err.to_string().contains("link-local"));
    }

    #[test]
    fn rejects_dns_answers_containing_non_public_addresses() {
        let mixed = [
            "93.184.216.34:0".parse::<SocketAddr>().unwrap(),
            "10.0.0.7:0".parse::<SocketAddr>().unwrap(),
        ];
        let error = validate_resolved_addresses("api.example.com", &mixed).unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::PermissionDenied);

        let rebound = ["127.0.0.1:0".parse::<SocketAddr>().unwrap()];
        assert!(validate_resolved_addresses("api.example.com", &rebound).is_err());
        assert!(validate_resolved_addresses("localhost", &rebound).is_ok());
    }
}
