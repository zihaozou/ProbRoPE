
use std::env;
use std::ffi::c_void;
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use fs_recon::gpu_decode::{scan_triggers_cpu, DecodedEvent, GpuEventDecoder};
use metavision_sys::evt3_raw::Evt3RawFile;
use metavision_sys::MvCamera;

const DEFAULT_RAW: &str =
    r"C:\Users\98600\Desktop\calibration_dancer_revision\event\recording_2026-08-17_14-55-44.raw";
const DEFAULT_CHUNK_WORDS: usize = 16_384;
const SAMPLE_EVENTS: usize = 10_000_000;
const SAMPLE_STRIDE: usize = 100_000;
const SWEEP_PREFIX_WORDS: usize = 20_000_000;

static SDK_TOTAL: AtomicU64 = AtomicU64::new(0);
static SDK_TRIG_TOTAL: AtomicU64 = AtomicU64::new(0);
static SDK_LAST_T: AtomicI64 = AtomicI64::new(0);
static SDK_SAMPLE: Mutex<Vec<DecodedEvent>> = Mutex::new(Vec::new());

unsafe extern "C" fn on_cd(e: *const metavision_sys::MvEventCD, n: usize, _u: *mut c_void) {
    let evs = std::slice::from_raw_parts(e, n);
    let prev = SDK_TOTAL.fetch_add(n as u64, Ordering::Relaxed) as usize;
    if let Some(last) = evs.last() {
        SDK_LAST_T.store(last.t, Ordering::Relaxed);
    }
    if prev < SAMPLE_EVENTS {
        let mut guard = SDK_SAMPLE.lock().unwrap();
        for ev in evs {
            if guard.len() >= SAMPLE_EVENTS {
                break;
            }
            guard.push(DecodedEvent { x: ev.x, y: ev.y, p: ev.p as u8, t_us: ev.t });
        }
    }
}

unsafe extern "C" fn on_trig(_e: *const metavision_sys::MvEventTrigger, n: usize, _u: *mut c_void) {
    SDK_TRIG_TOTAL.fetch_add(n as u64, Ordering::Relaxed);
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let raw_path = args.get(1).cloned().unwrap_or_else(|| DEFAULT_RAW.to_string());
    let chunk_words: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(DEFAULT_CHUNK_WORDS);

    println!("=== Phase 5 GPU EVT3 decode spike: gpu_decode_bench ===");
    println!("raw file:    {raw_path}");
    println!("chunk_words: {chunk_words}\n");

    let t_load0 = Instant::now();
    let file = Evt3RawFile::open(&raw_path).unwrap_or_else(|e| panic!("failed to open {raw_path}: {e}"));
    let load_ms = t_load0.elapsed().as_secs_f64() * 1000.0;
    let n_words = file.words.len();
    let file_bytes = n_words * 2;
    println!(
        "header: {}x{} evt{} | {n_words} words ({:.1} MB raw payload) loaded from disk in {load_ms:.1} ms",
        file.header.width,
        file.header.height,
        file.header.evt_version,
        file_bytes as f64 / 1e6
    );

    let mut decoder = GpuEventDecoder::new().unwrap_or_else(|e| panic!("GpuEventDecoder::new failed: {e}"));
    let (gpu_events, gpu_triggers, stats) =
        decoder.decode(&file.words, chunk_words).unwrap_or_else(|e| panic!("gpu decode failed: {e}"));

    let kernel_ms = stats.pass1_kernel_ms + stats.pass2_kernel_ms;
    let mev_s_kernel_only = (stats.cd_events as f64 / 1e6) / (kernel_ms.max(1e-6) / 1000.0);
    let mev_s_end_to_end = (stats.cd_events as f64 / 1e6) / (stats.total_ms.max(1e-6) / 1000.0);

    println!("\n--- GPU decode (full file, chunk_words={chunk_words}, {} chunks) ---", stats.n_chunks);
    println!("{stats:#?}");
    println!("cd_events={} triggers={}", stats.cd_events, stats.triggers);
    println!(
        "  kernel-only : {mev_s_kernel_only:8.1} Mev/s  ({kernel_ms:8.1} ms = pass1 {:.1} + pass2 {:.1})",
        stats.pass1_kernel_ms, stats.pass2_kernel_ms
    );
    println!(
        "  end-to-end  : {mev_s_end_to_end:8.1} Mev/s  ({:8.1} ms total, incl. upload/host-scan/download)",
        stats.total_ms
    );

    println!("\n--- chunk_words micro-sweep (first {SWEEP_PREFIX_WORDS} words only) ---");
    let sweep_words = &file.words[..SWEEP_PREFIX_WORDS.min(n_words)];
    for &cw in &[1_024usize, 4_096, 16_384, 65_536, 262_144] {
        let (dev, _trig, s) = decoder.decode(sweep_words, cw).expect("sweep decode failed");
        let k_ms = s.pass1_kernel_ms + s.pass2_kernel_ms;
        let mev_k = (dev.len as f64 / 1e6) / (k_ms.max(1e-6) / 1000.0);
        let mev_e2e = (dev.len as f64 / 1e6) / (s.total_ms.max(1e-6) / 1000.0);
        println!(
            "  chunk_words={cw:>8}  n_chunks={:>7}  kernel-only={mev_k:8.1} Mev/s  end-to-end={mev_e2e:8.1} Mev/s",
            s.n_chunks
        );
    }

    println!("\n--- EXT_TRIGGER extraction: GPU (integrated in pass2) vs standalone CPU scan ---");
    let t0 = Instant::now();
    let cpu_triggers = scan_triggers_cpu(&file.words);
    let cpu_trig_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let cpu_mword_s = (n_words as f64 / 1e6) / (cpu_trig_ms.max(1e-6) / 1000.0);
    println!(
        "  GPU (pass2, integrated into full CD decode above): {} triggers -- marginal cost over CD decode alone is ~0 (same word scan, same kernel launch)",
        stats.triggers
    );
    println!("  CPU (scan_triggers_cpu, standalone, single-threaded): {} triggers in {cpu_trig_ms:.1} ms ({cpu_mword_s:.1} Mword/s)", cpu_triggers.len());
    assert_eq!(gpu_triggers.len(), cpu_triggers.len(), "gpu vs cpu trigger COUNT must match");
    assert_eq!(gpu_triggers, cpu_triggers, "gpu vs cpu decoded triggers must match exactly");
    println!("  PASS: GPU and CPU trigger decodes agree exactly ({} triggers)", gpu_triggers.len());

    println!("\n--- SDK CPU decode baseline (metavision-sys / Camera::from_file, same mechanism as examples/read_raw.rs) ---");
    let t_sdk0 = Instant::now();
    let cam = MvCamera::open_file(&raw_path, false).expect("MvCamera::open_file failed");
    let (sdk_w, sdk_h) = cam.geometry().expect("geometry failed");
    unsafe {
        cam.set_cd_callback(on_cd, std::ptr::null_mut()).expect("set_cd_callback failed");
        cam.set_trigger_callback(on_trig, std::ptr::null_mut()).expect("set_trigger_callback failed");
    }
    cam.start().expect("start failed");
    let mut prev = u64::MAX;
    loop {
        std::thread::sleep(Duration::from_millis(200));
        let n = SDK_TOTAL.load(Ordering::Relaxed);
        if n == prev {
            break;
        }
        prev = n;
    }
    cam.stop().expect("stop failed");
    let sdk_ms = t_sdk0.elapsed().as_secs_f64() * 1000.0;
    let sdk_total = SDK_TOTAL.load(Ordering::Relaxed) as usize;
    let sdk_trig_total = SDK_TRIG_TOTAL.load(Ordering::Relaxed) as usize;
    let sdk_mev_s = (sdk_total as f64 / 1e6) / (sdk_ms.max(1e-6) / 1000.0);
    println!("geometry from SDK: {sdk_w}x{sdk_h}  (header said {}x{})", file.header.width, file.header.height);
    println!("SDK decode: {sdk_total} CD events, {sdk_trig_total} triggers in {sdk_ms:.1} ms  ({sdk_mev_s:.1} Mev/s)");
    println!("last SDK CD timestamp: {} us", SDK_LAST_T.load(Ordering::Relaxed));

    println!("\n--- Throughput comparison (full file, {} words / {:.1} MB) ---", n_words, file_bytes as f64 / 1e6);
    println!("  GPU kernel-only : {mev_s_kernel_only:8.1} Mev/s");
    println!("  GPU end-to-end  : {mev_s_end_to_end:8.1} Mev/s");
    println!("  SDK CPU decode  : {sdk_mev_s:8.1} Mev/s");
    println!("  speedup (kernel-only vs SDK): {:.2}x", mev_s_kernel_only / sdk_mev_s.max(1e-9));
    println!("  speedup (end-to-end vs SDK) : {:.2}x", mev_s_end_to_end / sdk_mev_s.max(1e-9));

    let raw_upload_bytes = file_bytes;
    let decoded_upload_equiv_bytes = stats.cd_events * 16;
    println!("\n--- PCIe traffic comparison ---");
    println!(
        "  raw EVT3 upload (this decoder's actual H2D traffic): {:.1} MB  ({:.2} bytes/event)",
        raw_upload_bytes as f64 / 1e6,
        raw_upload_bytes as f64 / stats.cd_events.max(1) as f64
    );
    println!(
        "  decoded-upload equivalent (16 B/event, if decoding on CPU and uploading results instead): {:.1} MB",
        decoded_upload_equiv_bytes as f64 / 1e6
    );
    println!(
        "  reduction: {:.1}x less PCIe traffic uploading raw EVT3 words than pre-decoded events",
        decoded_upload_equiv_bytes as f64 / raw_upload_bytes as f64
    );

    println!("\n--- Correctness gate: GPU decode vs SDK decode ---");
    assert_eq!(stats.cd_events, sdk_total, "TOTAL CD event count must match the SDK exactly");
    println!("PASS: total CD event count matches exactly ({sdk_total})");
    assert_eq!(stats.triggers, sdk_trig_total, "TOTAL trigger count must match the SDK exactly");
    println!("PASS: total trigger count matches exactly ({sdk_trig_total})");

    let n_sample = SAMPLE_EVENTS.min(stats.cd_events).min(sdk_total);
    let gpu_sample = gpu_events
        .download_range(decoder.stream(), 0, n_sample)
        .unwrap_or_else(|e| panic!("download_range failed: {e}"));
    let sdk_sample = SDK_SAMPLE.lock().unwrap().clone();
    assert!(gpu_sample.len() >= n_sample, "gpu sample too short: {} < {n_sample}", gpu_sample.len());
    assert!(sdk_sample.len() >= n_sample, "sdk sample too short: {} < {n_sample}", sdk_sample.len());

    let mut offset: Option<i64> = None;
    let mut checked = 0usize;
    let mut idx = 0usize;
    while idx < n_sample {
        let g = gpu_sample[idx];
        let s = sdk_sample[idx];
        assert_eq!((g.x, g.y, g.p), (s.x, s.y, s.p), "event #{idx}: x/y/p mismatch (gpu={g:?} sdk={s:?})");
        let this_offset = g.t_us - s.t_us;
        match offset {
            None => offset = Some(this_offset),
            Some(o) => assert_eq!(
                this_offset, o,
                "event #{idx}: timestamp offset from the SDK is not constant (gpu={g:?} sdk={s:?}, expected offset {o})"
            ),
        }
        checked += 1;
        idx += SAMPLE_STRIDE;
    }
    println!(
        "PASS: {checked} sampled events (every {SAMPLE_STRIDE}-th of the first {n_sample}) match the SDK decode \
         exactly on x/y/p, with a constant timestamp offset of {} us from the SDK's time-shifted playback clock",
        offset.unwrap_or(0)
    );

    println!("\n=== ALL CHECKS PASSED ===");
}
