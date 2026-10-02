
use metavision_sys::MvCamera;

const BIASES: &[&str] = &["bias_diff_on", "bias_diff_off", "bias_fo", "bias_hpf", "bias_refr"];

fn main() {
    let cam = match MvCamera::open_live() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("camera busy or not found, exiting gracefully: {e}");
            std::process::exit(0);
        }
    };

    match cam.geometry() {
        Ok((w, h)) => println!("geometry: {w}x{h}"),
        Err(e) => println!("geometry: <unavailable: {e}>"),
    }

    match cam.probe_facilities() {
        Ok(f) => {
            println!("facilities:");
            println!("  I_LL_Biases            : {}", f.ll_biases);
            println!("  I_ErcModule             : {}", f.erc);
            println!("  I_AntiFlickerModule     : {}", f.antiflicker);
            println!("  I_EventTrailFilterModule: {}", f.trail_filter);
            println!("  I_ROI                   : {}", f.roi);
            println!("  I_DigitalCrop           : {}", f.digital_crop);
        }
        Err(e) => {
            eprintln!("probe_facilities failed, exiting gracefully: {e}");
            std::process::exit(0);
        }
    }

    println!("\nbiases (current value, recommended range, allowed range):");
    for name in BIASES {
        let cur = cam.get_bias(name);
        let info = cam.bias_info(name);
        match (cur, info) {
            (Ok(v), Ok(i)) => println!(
                "  {name:<14} value={v}  recommended=[{}, {}]  allowed=[{}, {}]",
                i.min_recommended, i.max_recommended, i.min_allowed, i.max_allowed
            ),
            (cur, info) => println!("  {name:<14} <read failed: cur={cur:?} info={info:?}>"),
        }
    }
}
