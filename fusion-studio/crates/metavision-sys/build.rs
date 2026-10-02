fn main() {
    let root = std::env::var("METAVISION_ROOT")
        .unwrap_or_else(|_| r"C:\Program Files\Prophesee".to_string());
    cc::Build::new()
        .cpp(true)
        .file("shim/shim.cpp")
        .include(format!("{root}\\include"))
        .include(format!("{root}\\third_party\\include"))
        .flag("/std:c++17")
        .flag("/EHsc")
        .compile("mv_shim");
    println!("cargo:rustc-link-search=native={root}\\lib");
    for lib in ["metavision_hal", "metavision_sdk_base", "metavision_sdk_core", "metavision_sdk_stream"] {
        println!("cargo:rustc-link-lib={lib}");
    }
    println!("cargo:rerun-if-changed=shim/shim.cpp");
}
