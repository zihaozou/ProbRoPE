fn main() {
    let version = std::process::Command::new("git")
        .args(["describe", "--tags", "--always", "--dirty"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| std::env::var("CARGO_PKG_VERSION").unwrap_or_else(|_| "unknown".into()));

    println!("cargo:rustc-env=FS_RECORD_SOFTWARE_VERSION={version}");

    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_default();
    println!("cargo:rerun-if-changed={manifest_dir}/../../.git/HEAD");
    println!("cargo:rerun-if-changed={manifest_dir}/../../.git/index");
}
