use std::path::PathBuf;

fn main() {
    let root = std::env::var("SPINNAKER_ROOT")
        .unwrap_or_else(|_| r"C:\Program Files\Teledyne\Spinnaker".to_string());
    println!("cargo:rustc-link-search=native={root}\\lib64\\vs2015");
    println!("cargo:rustc-link-lib=SpinnakerC_v140");

    let bindings = bindgen::Builder::default()
        .header(format!("{root}\\include\\spinc\\SpinnakerC.h"))
        .clang_arg(format!("-I{root}\\include\\spinc"))
        .allowlist_function("spin.*")
        .allowlist_type("spin.*|_spin.*")
        .allowlist_var("SPINNAKER.*|MAX_BUFF_LEN")
        .generate()
        .expect("bindgen failed");
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    bindings.write_to_file(out.join("bindings.rs")).unwrap();
}
