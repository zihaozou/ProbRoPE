
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use metavision_sys::{evt3_raw, MvCamera};

static CB_BYTES: AtomicU64 = AtomicU64::new(0);
static CB_BATCHES: AtomicU64 = AtomicU64::new(0);

unsafe extern "C" fn on_raw(_data: *const u8, n: usize, _user: *mut c_void) {
    CB_BYTES.fetch_add(n as u64, Ordering::Relaxed);
    CB_BATCHES.fetch_add(1, Ordering::Relaxed);
}

fn main() {
    let tmp_dir = std::env::temp_dir().join("raw_logging_probe");
    std::fs::create_dir_all(&tmp_dir).expect("create temp dir");
    let raw_path = tmp_dir.join("probe.raw");
    let raw_path_str = raw_path.to_str().expect("temp path is valid UTF-8").to_string();
    let _ = std::fs::remove_file(&raw_path);

    println!("=== raw_logging_probe ===");
    println!("probe.raw path: {raw_path_str}");

    let cam = match MvCamera::open_live() {
        Ok(c) => c,
        Err(e) => {
            println!("BLOCKED: MvCamera::open_live() failed: {e}");
            std::process::exit(1);
        }
    };

    if let Err(e) = cam.enable_trigger_in() {
        println!("note: enable_trigger_in failed (non-fatal for this probe): {e}");
    }

    match cam.geometry() {
        Ok((w, h)) => println!("geometry: {w}x{h}"),
        Err(e) => println!("geometry: <unavailable: {e}>"),
    }

    unsafe {
        cam.set_raw_callback(on_raw, std::ptr::null_mut()).expect("set_raw_callback");
    }

    match cam.start_recording(&raw_path_str) {
        Ok(()) => println!("start_recording: ok"),
        Err(e) => {
            println!("start_recording: FAILED: {e}");
            println!("\nverdict: NO-GO -- start_recording itself failed with the raw callback registered");
            std::process::exit(0);
        }
    }

    cam.start().expect("cam.start");
    println!("streaming for 10s...");

    let t0 = Instant::now();
    let mut last_report = t0;
    while t0.elapsed() < Duration::from_secs(10) {
        std::thread::sleep(Duration::from_millis(200));
        if last_report.elapsed() >= Duration::from_secs(2) {
            last_report = Instant::now();
            let bytes = CB_BYTES.load(Ordering::Relaxed);
            let file_len = std::fs::metadata(&raw_path).map(|m| m.len()).unwrap_or(0);
            println!(
                "  t={:>4.1}s  callback_bytes={bytes:>10}  probe.raw_size={file_len:>10}",
                t0.elapsed().as_secs_f64()
            );
        }
    }

    if let Err(e) = cam.stop_recording() {
        println!("stop_recording: FAILED: {e}");
    } else {
        println!("stop_recording: ok");
    }
    cam.stop().expect("cam.stop");

    let cb_bytes = CB_BYTES.load(Ordering::Relaxed);
    let cb_batches = CB_BATCHES.load(Ordering::Relaxed);
    let file_len = std::fs::metadata(&raw_path).map(|m| m.len()).unwrap_or(0);

    println!("\n=== results ===");
    println!("callback: {cb_bytes} bytes across {cb_batches} batches");
    println!("probe.raw: {file_len} bytes");

    let header_result = std::fs::read(&raw_path).ok().and_then(|data| evt3_raw::parse_header(&data).ok());
    match &header_result {
        Some((header, offset)) => {
            println!(
                "header parsed: width={} height={} evt_version={:?} payload_offset={offset}",
                header.width, header.height, header.evt_version
            );
            let payload = &std::fs::read(&raw_path).unwrap()[*offset..];
            let mut words = Vec::with_capacity(payload.len() / 2);
            for pair in payload.chunks_exact(2) {
                words.push(u16::from_le_bytes([pair[0], pair[1]]));
            }
            match evt3_raw::time_shift_us(&words) {
                Some(shift) => println!("time_shift_us: {shift}"),
                None => println!("time_shift_us: <no TIME_HIGH word found in payload>"),
            }
        }
        None => println!("header parsed: FAILED (file missing, unreadable, or not a valid EVT3 header)"),
    }

    let crit_a = cb_bytes > 0;
    let crit_b = file_len > 0 && header_result.is_some();
    let ratio = if cb_bytes > 0 && file_len > 0 {
        let (a, b) = (cb_bytes as f64, file_len as f64);
        (a.max(b) / a.min(b), true)
    } else {
        (f64::INFINITY, false)
    };
    let crit_c = crit_a && crit_b && ratio.1 && ratio.0 <= 2.0;

    println!("\n=== criteria ===");
    println!("A. callback still receiving data (bytes > 0): {} (bytes={cb_bytes})", pass_fail(crit_a));
    println!("B. file non-empty and header parses: {} (file_len={file_len}, header_ok={})", pass_fail(crit_b), header_result.is_some());
    if ratio.1 {
        println!("C. file size vs callback bytes within 2x: {} (ratio={:.3})", pass_fail(crit_c), ratio.0);
    } else {
        println!("C. file size vs callback bytes within 2x: {} (one side is zero, ratio undefined)", pass_fail(crit_c));
    }

    let go = crit_a && crit_b && crit_c;
    println!("\nverdict: {}", if go { "GO -- SDK raw logging coexists with our raw callback" } else { "NO-GO -- see failing criteria above" });

    let cleanup_len = std::fs::metadata(&raw_path).map(|m| m.len()).unwrap_or(0);
    let _ = std::fs::remove_file(&raw_path);
    println!("\n(probe.raw deleted, was {cleanup_len} bytes)");
}

fn pass_fail(b: bool) -> &'static str {
    if b {
        "PASS"
    } else {
        "FAIL"
    }
}
