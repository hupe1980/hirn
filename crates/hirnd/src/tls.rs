use std::fs;
use std::io::BufReader;
use std::path::Path;
use std::sync::Arc;

use tokio_rustls::TlsAcceptor;

use crate::config::TlsConfig;

/// Load TLS certificate and key, returning a `TlsAcceptor`.
///
/// When `client_ca_path` is set in the config, mutual TLS is enabled:
/// the server requires clients to present a certificate signed by the given CA.
pub fn load_tls(config: &TlsConfig) -> Result<TlsAcceptor, Box<dyn std::error::Error>> {
    // Ensure a CryptoProvider is installed (idempotent if already set)
    let _ = rustls::crypto::ring::default_provider().install_default();

    let cert_pem = fs::read(&config.cert_path)?;
    let key_pem = fs::read(&config.key_path)?;

    let certs = rustls_pemfile::certs(&mut BufReader::new(cert_pem.as_slice()))
        .collect::<Result<Vec<_>, _>>()?;

    let key = rustls_pemfile::private_key(&mut BufReader::new(key_pem.as_slice()))?
        .ok_or("no private key found in key file")?;

    let tls_config = if let Some(ref ca_path) = config.client_ca_path {
        // mTLS: require client certificates signed by the given CA
        let ca_pem = fs::read(ca_path)?;
        let ca_certs = rustls_pemfile::certs(&mut BufReader::new(ca_pem.as_slice()))
            .collect::<Result<Vec<_>, _>>()?;

        let mut root_store = rustls::RootCertStore::empty();
        for cert in ca_certs {
            root_store.add(cert)?;
        }

        let verifier = rustls::server::WebPkiClientVerifier::builder(Arc::new(root_store))
            .build()
            .map_err(|e| format!("failed to build client cert verifier: {e}"))?;

        rustls::ServerConfig::builder()
            .with_client_cert_verifier(verifier)
            .with_single_cert(certs, key)?
    } else {
        rustls::ServerConfig::builder()
            .with_no_client_auth()
            .with_single_cert(certs, key)?
    };

    Ok(TlsAcceptor::from(Arc::new(tls_config)))
}

/// Generate a self-signed CA certificate and key, writing them to the given paths.
/// Returns the `rcgen::CertificateParams` for signing client certs in tests.
pub fn generate_self_signed_cert(
    cert_path: &Path,
    key_path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let subject_alt_names = vec!["localhost".to_owned(), "127.0.0.1".to_owned()];

    let rcgen::CertifiedKey { cert, key_pair } =
        rcgen::generate_simple_self_signed(subject_alt_names)?;

    let cert_pem = cert.pem();
    let key_pem = key_pair.serialize_pem();

    fs::write(cert_path, cert_pem)?;
    fs::write(key_path, key_pem)?;

    Ok(())
}

/// Generate a CA certificate for mTLS client authentication.
/// Returns the CA cert/key so they can be used to sign client certificates.
pub fn generate_ca_cert(
    ca_cert_path: &Path,
    ca_key_path: &Path,
) -> Result<rcgen::CertifiedKey, Box<dyn std::error::Error>> {
    use rcgen::{CertificateParams, IsCa, KeyUsagePurpose};

    let mut params = CertificateParams::default();
    params.is_ca = IsCa::Ca(rcgen::BasicConstraints::Unconstrained);
    params.key_usages.push(KeyUsagePurpose::KeyCertSign);
    params.key_usages.push(KeyUsagePurpose::CrlSign);
    params
        .distinguished_name
        .push(rcgen::DnType::CommonName, "hirn Test CA");

    let ca_key = rcgen::KeyPair::generate()?;
    let ca_cert = params.self_signed(&ca_key)?;

    fs::write(ca_cert_path, ca_cert.pem())?;
    fs::write(ca_key_path, ca_key.serialize_pem())?;

    Ok(rcgen::CertifiedKey {
        cert: ca_cert,
        key_pair: ca_key,
    })
}

/// Generate a client certificate signed by the given CA.
/// The CN is set to `cn`, which maps to a client identity in the auth config.
pub fn generate_client_cert(
    ca: &rcgen::CertifiedKey,
    cn: &str,
    cert_path: &Path,
    key_path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    use rcgen::{CertificateParams, ExtendedKeyUsagePurpose};

    let mut params = CertificateParams::default();
    params
        .distinguished_name
        .push(rcgen::DnType::CommonName, cn);
    params
        .extended_key_usages
        .push(ExtendedKeyUsagePurpose::ClientAuth);

    let client_key = rcgen::KeyPair::generate()?;
    let client_cert = params.signed_by(&client_key, &ca.cert, &ca.key_pair)?;

    fs::write(cert_path, client_cert.pem())?;
    fs::write(key_path, client_key.serialize_pem())?;

    Ok(())
}

/// Extract the Common Name (CN) from a DER-encoded X.509 certificate.
pub fn extract_cn(cert_der: &[u8]) -> Option<String> {
    use x509_parser::prelude::*;

    let (_, cert) = X509Certificate::from_der(cert_der).ok()?;
    cert.subject()
        .iter_common_name()
        .next()
        .and_then(|cn| cn.as_str().ok())
        .map(|s| s.to_owned())
}
