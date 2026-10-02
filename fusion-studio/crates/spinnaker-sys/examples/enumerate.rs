fn main() {
    let sys = spinnaker_sys::wrapper::SpinSystem::new().expect("spinnaker system");
    println!("cameras: {}", sys.camera_count().expect("count"));
}
