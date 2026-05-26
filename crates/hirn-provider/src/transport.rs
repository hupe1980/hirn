use hirn_core::{HirnError, HirnResult};

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
    matches!(
        url.host_str(),
        Some("localhost") | Some("127.0.0.1") | Some("::1")
    )
}

#[cfg(test)]
mod tests {
    use super::{validate_privacy_bearing_base_url, validate_secret_bearing_base_url};

    #[test]
    fn allows_https_base_url() {
        validate_secret_bearing_base_url("test", "https://api.example.com/v1").unwrap();
    }

    #[test]
    fn allows_loopback_http_base_url() {
        validate_secret_bearing_base_url("test", "http://127.0.0.1:8080/v1").unwrap();
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
}
