
use std::ffi::c_void;
use std::time::Instant;

use crossbeam_channel::{bounded, Sender};
use fs_recon::gpu_decode::{CarryState, DecodedEvent, DecodedTrigger, GpuEventDecoder};
use metavision_sys::evt3_raw::{bytes_to_words, time_shift_us, Evt3RawFile};
use metavision_sys::MvCamera;

const DEFAULT_RAW: &str =
    r"C:\Users\98600\Desktop\calibration_dancer_revision\event\recording_2026-08-17_14-55-44.raw";

fn fold_events(mut h: u64, events: &[DecodedEvent]) -> u64 {
    const P: u64 = 0x0000_0100_0000_01B3;
    for e in events {
        for v in [e.x as u64, e.y as u64, e.p as u64, e.t_us as u64] {
            h ^= v;
            h = h.wrapping_mul(P);
        }
    }
    h
}

enum RawMsg {
    Bytes(Vec<u8>),
    Eof,
}

struct Ctx {
    tx: Sender<RawMsg>,
}

unsafe extern "C" fn on_raw(data: *const u8, n: usize, user: *mut c_void) {
    let ctx = &*(user as *const Ctx);
    let bytes = std::slice::from_raw_parts(data, n).to_vec();
    let _ = ctx.tx.send(RawMsg::Bytes(bytes));
}

unsafe extern "C" fn on_status(is_eof: i32, user: *mut c_void) {
    if is_eof != 0 {
        let ctx = &*(user as *const Ctx);
        let _ = ctx.tx.send(RawMsg::Eof);
    }
}

fn main() {
    let raw_path = std::env::args().nth(1).unwrap_or_else(|| DEFAULT_RAW.to_string());
    println!("=== Phase 5 gpu_stream_check: live raw surface + streaming decode vs whole-file ===");
    println!("raw file: {raw_path}\n");

    let file = Evt3RawFile::open(&raw_path).unwrap_or_else(|e| panic!("failed to open {raw_path}: {e}"));
    let ref_words = file.words.len();
    let ref_bytes = ref_words * 2;
    let shift = time_shift_us(&file.words);
    println!("reference: {ref_words} words ({:.1} MB payload), SDK-equivalent time_shift_us = {shift:?}", ref_bytes as f64 / 1e6);

    let mut decoder = GpuEventDecoder::new().unwrap_or_else(|e| panic!("GpuEventDecoder::new failed: {e}"));
    let (ref_dev, ref_triggers, ref_stats) =
        decoder.decode(&file.words, 65_536).unwrap_or_else(|e| panic!("whole-file decode failed: {e}"));
    let mut ref_hash: u64 = 0xcbf29ce484222325;
    let mut off = 0usize;
    const WINDOW: usize = 8_000_000;
    while off < ref_dev.len {
        let chunk = decoder_download(&decoder, &ref_dev, off, WINDOW);
        ref_hash = fold_events(ref_hash, &chunk);
        off += chunk.len();
    }
    println!(
        "reference decode: {} CD events + {} triggers (kernel {:.1} ms), hash={ref_hash:#018x}\n",
        ref_stats.cd_events,
        ref_stats.triggers,
        ref_stats.pass1_kernel_ms + ref_stats.pass2_kernel_ms
    );
    drop(ref_dev);

    let (tx, rx) = bounded::<RawMsg>(64);
    let ctx = Box::leak(Box::new(Ctx { tx }));
    let user = ctx as *mut Ctx as *mut c_void;

    let t0 = Instant::now();
    let cam = MvCamera::open_file(&raw_path, false).expect("MvCamera::open_file failed");
    unsafe {
        cam.set_raw_callback(on_raw, user).expect("set_raw_callback failed");
        cam.set_status_callback(on_status, user).expect("set_status_callback failed");
    }
    cam.start().expect("start failed");

    let mut state = CarryState::default();
    let mut byte_carry: Option<u8> = None;
    let mut words: Vec<u16> = Vec::new();
    let mut stream_bytes = 0usize;
    let mut stream_words = 0usize;
    let mut stream_cd = 0usize;
    let mut stream_hash: u64 = 0xcbf29ce484222325;
    let mut stream_triggers: Vec<DecodedTrigger> = Vec::new();
    let mut n_buffers = 0usize;
    let mut header_bytes_seen = false;

    while let Ok(msg) = rx.recv() {
        match msg {
            RawMsg::Bytes(bytes) => {
                n_buffers += 1;
                stream_bytes += bytes.len();
                if n_buffers == 1 && bytes.first() == Some(&b'%') {
                    header_bytes_seen = true;
                }
                words.clear();
                bytes_to_words(&mut words, &mut byte_carry, &bytes);
                stream_words += words.len();
                if words.is_empty() {
                    continue;
                }
                let (dev, trig, _stats) = decoder
                    .decode_with_state(&words, 16_384, &mut state)
                    .unwrap_or_else(|e| panic!("streaming decode failed at buffer {n_buffers}: {e}"));
                stream_cd += dev.len;
                let mut off = 0usize;
                while off < dev.len {
                    let chunk = decoder_download(&decoder, &dev, off, WINDOW);
                    stream_hash = fold_events(stream_hash, &chunk);
                    off += chunk.len();
                }
                stream_triggers.extend(trig);
            }
            RawMsg::Eof => break,
        }
    }
    let _ = cam.stop();
    let stream_secs = t0.elapsed().as_secs_f64();

    println!("streaming pass: {n_buffers} raw buffers, {stream_bytes} bytes -> {stream_words} words in {stream_secs:.2}s");
    println!("streaming decode: {stream_cd} CD events + {} triggers, hash={stream_hash:#018x}", stream_triggers.len());
    println!("final carry-out clock: {} us (reference last event region)", state.time_us());

    assert!(!header_bytes_seen, "raw callback delivered text-header bytes -- header must not leak into the raw stream");
    assert_eq!(stream_bytes, ref_bytes, "raw callback byte total must equal the file's post-header payload");
    assert_eq!(stream_words, ref_words, "streamed word total must equal the file's word count");
    assert!(byte_carry.is_none(), "well-formed EVT3 stream must not end on a dangling odd byte");
    assert_eq!(stream_cd, ref_stats.cd_events, "streaming CD count must match whole-file decode");
    assert_eq!(stream_triggers.len(), ref_stats.triggers, "streaming trigger count must match whole-file decode");
    assert_eq!(stream_triggers, ref_triggers, "streaming triggers must match whole-file decode exactly");
    assert_eq!(stream_hash, ref_hash, "streaming event stream must be byte-identical to whole-file decode");

    println!("\n=== ALL CHECKS PASSED: streaming decode across SDK buffer boundaries is byte-identical ===");
}

fn decoder_download(
    decoder: &GpuEventDecoder,
    dev: &fs_recon::gpu_decode::DecodedEventsDevice,
    start: usize,
    count: usize,
) -> Vec<DecodedEvent> {
    dev.download_range(decoder.stream(), start, count).unwrap_or_else(|e| panic!("download_range failed: {e}"))
}
