fn main() {
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").ok();
    if target_os.as_deref() == Some("macos")
        && std::env::var_os("PYO3_BUILD_EXTENSION_MODULE").is_some()
    {
        pyo3_build_config::add_extension_module_link_args();
    }
}
