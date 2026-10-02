use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use clap::Parser;
use crossbeam_channel::bounded;
use fs_core::bus::EvkMsg;
use fs_core::GrayImage;
use fs_recon::{CudaManifold, Normalization, Reconstructor};
use opencv::calib3d;
use opencv::core::{Mat, Size, Vector};
use opencv::imgcodecs;
use opencv::prelude::*;

use fs_app::sources::replay::spawn_replay_evk;

const DEFAULT_RAW: &str =
    r"C:\Users\98600\Desktop\calibration_dancer_revision\event\recording_2026-08-17_14-55-44.raw";
const DEFAULT_TIMESTAMPS: &str =
    r"C:\Users\98600\Desktop\calibration_dancer_revision\event\timestamps.txt";
const DEFAULT_E2CALIB: &str = r"C:\Users\98600\Desktop\calibration_dancer_revision\event\e2calib";
const DEFAULT_OUT: &str = "target/golden_out";

#[derive(Parser)]
struct Args {
    #[arg(long)]
    raw: Option<String>,
    #[arg(long)]
    timestamps: Option<String>,
    #[arg(long)]
    e2calib: Option<String>,
    #[arg(long)]
    out: Option<String>,
    #[arg(long, default_value_t = 2)]
    iters: u32,

    #[arg(long)]
    c_pos: Option<f32>,

    #[arg(long)]
    c_neg: Option<f32>,

    #[arg(long)]
    lambda: Option<f32>,

    #[arg(long)]
    manifold_alpha: Option<f32>,

    #[arg(long)]
    tonemap_scale: Option<f32>,
    #[arg(long)]
    tau_leak_us: Option<f32>,
    #[arg(long, value_enum)]
    normalization: Option<NormalizationArg>,
    #[arg(long)]
    percentile_lo: Option<f32>,
    #[arg(long)]
    percentile_hi: Option<f32>,
    #[arg(long)]
    median: Option<bool>,
    #[arg(long, default_value_t = false)]
    gpu_decode: bool,

    #[arg(long)]
    decimation_threshold: Option<usize>,
}

#[derive(Clone, Copy, clap::ValueEnum)]
enum NormalizationArg {
    Fixed,
    Percentile,

    TileLocal,
}

fn golden_filename(t_us: i64) -> String {
    format!("{:019}.png", t_us * 1000)
}

fn load_timestamps(path: &Path) -> Vec<i64> {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("read timestamps file {}: {e}", path.display()));
    let mut v: Vec<i64> = text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(|l| l.parse::<i64>().unwrap_or_else(|e| panic!("parse timestamp {l:?}: {e}")))
        .collect();
    v.sort_unstable();
    v
}


fn mat_view(img: &GrayImage) -> opencv::boxed_ref::BoxedRef<'_, Mat> {
    Mat::new_rows_cols_with_data(img.h as i32, img.w as i32, &img.data).expect("wrap GrayImage as Mat")
}

const DETECT_FLAGS: i32 = calib3d::CALIB_CB_NORMALIZE_IMAGE | calib3d::CALIB_CB_ACCURACY;

fn detect(mat: &impl opencv::core::ToInputArray, pattern: Size) -> bool {
    let mut corners = Mat::default();
    match calib3d::find_chessboard_corners_sb(mat, pattern, &mut corners, DETECT_FLAGS) {
        Ok(found) => found,
        Err(e) => {
            eprintln!("warn: find_chessboard_corners_sb error: {e}");
            false
        }
    }
}


fn calibrate_pattern_size(e2calib_dir: &Path, timestamps: &[i64]) -> Size {
    let n = timestamps.len();
    let mid = n / 2;
    let half = 10usize;
    let lo = mid.saturating_sub(half);
    let hi = (mid + half).min(n);
    let sample = &timestamps[lo..hi];

    let candidates = [Size::new(6, 7), Size::new(7, 6)];
    let mut hits = [0usize; 2];
    let mut tried = 0usize;
    for &t in sample {
        let path = e2calib_dir.join(golden_filename(t));
        let Ok(mat) = imgcodecs::imread(&path, imgcodecs::IMREAD_GRAYSCALE) else {
            eprintln!("warn: pattern calibration: failed to read {}", path.display());
            continue;
        };
        if mat.empty() {
            continue;
        }
        tried += 1;
        for (i, &pat) in candidates.iter().enumerate() {
            if detect(&mat, pat) {
                hits[i] += 1;
            }
        }
    }
    println!(
        "pattern calibration ({tried} samples): Size(6,7) [width=6,height=7] = {} hits, Size(7,6) [width=7,height=6] = {} hits",
        hits[0], hits[1]
    );
    let chosen = if hits[0] >= hits[1] { candidates[0] } else { candidates[1] };
    println!(
        "pattern calibration: using Size(width={}, height={})",
        chosen.width, chosen.height
    );
    chosen
}

struct Perf {
    mean_ms: f64,
    p50_ms: f64,
    p95_ms: f64,
}

fn percentiles(mut samples_ms: Vec<f64>) -> Perf {
    samples_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = samples_ms.len();
    let mean_ms = samples_ms.iter().sum::<f64>() / n as f64;
    let at = |q: f64| samples_ms[(((n - 1) as f64) * q).round() as usize];
    Perf { mean_ms, p50_ms: at(0.50), p95_ms: at(0.95) }
}

fn main() {
    let args = Args::parse();
    let raw = args.raw.unwrap_or_else(|| DEFAULT_RAW.to_string());
    let timestamps_path = args.timestamps.unwrap_or_else(|| DEFAULT_TIMESTAMPS.to_string());
    let e2calib_dir = PathBuf::from(args.e2calib.unwrap_or_else(|| DEFAULT_E2CALIB.to_string()));
    let out_dir = PathBuf::from(args.out.unwrap_or_else(|| DEFAULT_OUT.to_string()));
    std::fs::create_dir_all(&out_dir)
        .unwrap_or_else(|e| panic!("create out dir {}: {e}", out_dir.display()));

    let timestamps = load_timestamps(Path::new(&timestamps_path));
    println!("loaded {} render timestamps from {timestamps_path}", timestamps.len());

    let pattern = calibrate_pattern_size(&e2calib_dir, &timestamps);

    let (evk_tx, evk_rx) = bounded::<EvkMsg>(2048);
    let mut _keepalive = None;
    let (w, h) = if args.gpu_decode {
        println!("decode path: GPU (spawn_gpu_replay_evk)");
        let (_replay_handle, dims) = fs_app::sources::replay::spawn_gpu_replay_evk(raw.clone(), false, evk_tx);
        dims
    } else {
        let evk_snapshot =
            std::sync::Arc::new(std::sync::Mutex::new(fs_app::settings::EvkSettingsSnapshot::default()));
        let shutdown = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let rates = std::sync::Arc::new(fs_app::stream_rates::StreamRates::default());
        let (_replay_handle, cmd_tx, dims) = spawn_replay_evk(
            raw.clone(),
            false,
            evk_tx,
            evk_snapshot,
            rates,
            fs_app::transform::EventXform::default(),
            fs_app::transform::EventRecordTap::default(),
            shutdown,
        );
        _keepalive = Some(cmd_tx);
        dims
    };
    println!("recon dims from replay geometry: {w}x{h}");

    let mut recon = CudaManifold::new(w, h)
        .unwrap_or_else(|e| panic!("CudaManifold::new failed (requires a CUDA-capable GPU): {e}"));
    recon.iters = args.iters;
    const TUNED_TAU_LEAK_US: f32 = 2_000_000.0;
    const TUNED_TONEMAP_SCALE: f32 = 400.0;
    const TUNED_NORMALIZATION: NormalizationArg = NormalizationArg::TileLocal;
    const TUNED_PERCENTILE_LO: f32 = 0.001;
    const TUNED_PERCENTILE_HI: f32 = 0.999;
    const TUNED_MEDIAN_FILTER: bool = true;
    recon.tau_leak_us = TUNED_TAU_LEAK_US;
    recon.tonemap_scale = TUNED_TONEMAP_SCALE;
    recon.normalization = match args.normalization.unwrap_or(TUNED_NORMALIZATION) {
        NormalizationArg::Fixed => Normalization::Fixed,
        NormalizationArg::Percentile => Normalization::Percentile,
        NormalizationArg::TileLocal => Normalization::TileLocal,
    };
    recon.percentile_lo = TUNED_PERCENTILE_LO;
    recon.percentile_hi = TUNED_PERCENTILE_HI;
    recon.median_filter = args.median.unwrap_or(TUNED_MEDIAN_FILTER);
    if let Some(v) = args.c_pos { recon.c_pos = v; }
    if let Some(v) = args.c_neg { recon.c_neg = v; }
    if let Some(v) = args.lambda { recon.lambda = v; }
    if let Some(v) = args.manifold_alpha { recon.manifold_alpha = v; }
    if let Some(v) = args.tonemap_scale { recon.tonemap_scale = v; }
    if let Some(v) = args.tau_leak_us { recon.tau_leak_us = v; }
    if let Some(v) = args.percentile_lo { recon.percentile_lo = v; }
    if let Some(v) = args.percentile_hi { recon.percentile_hi = v; }
    if let Some(v) = args.decimation_threshold { recon.event_decimation_threshold = v; }
    println!(
        "CudaManifold params: c_pos={} c_neg={} lambda={} iters={} manifold_alpha={} tonemap_scale={} tau_leak_us={} normalization={:?} percentile_lo={} percentile_hi={} median_filter={} event_decimation_threshold={}",
        recon.c_pos, recon.c_neg, recon.lambda, recon.iters, recon.manifold_alpha, recon.tonemap_scale,
        recon.tau_leak_us, recon.normalization, recon.percentile_lo, recon.percentile_hi, recon.median_filter,
        recon.event_decimation_threshold
    );

    let mut img = GrayImage::new(w, h);
    let mut pending_ts = timestamps.iter().copied().peekable();
    let mut latest_event_t: i64 = i64::MIN;

    let mut total_events: u64 = 0;
    let mut push_dur = Duration::ZERO;
    let mut render_samples_ms: Vec<f64> = Vec::with_capacity(timestamps.len());
    let mut ours_detected = 0usize;
    let mut golden_detected = 0usize;
    let mut rendered = 0usize;
    let mut goldens_scored = 0usize;
    let mut max_batch_events: usize = 0;

    let pipeline_start = Instant::now();

    let render_and_score = |recon: &mut CudaManifold,
                                 img: &mut GrayImage,
                                 t: i64,
                                 render_samples_ms: &mut Vec<f64>,
                                 ours_detected: &mut usize,
                                 golden_detected: &mut usize,
                                 goldens_scored: &mut usize,
                                 rendered: &mut usize,
                                 max_batch_events: &mut usize| {
        let r0 = Instant::now();
        recon.render_at(t, img);
        render_samples_ms.push(r0.elapsed().as_secs_f64() * 1000.0);
        *rendered += 1;
        *max_batch_events = (*max_batch_events).max(recon.last_render_stats().events_this_render);

        let fname = golden_filename(t);
        let out_path = out_dir.join(&fname);
        {
            let mat = mat_view(img);
            imgcodecs::imwrite(&out_path, &mat, &Vector::<i32>::new())
                .unwrap_or_else(|e| panic!("imwrite {}: {e}", out_path.display()));
            if detect(&mat, pattern) {
                *ours_detected += 1;
            }
        }

        let golden_path = e2calib_dir.join(&fname);
        match imgcodecs::imread(&golden_path, imgcodecs::IMREAD_GRAYSCALE) {
            Ok(gmat) if !gmat.empty() => {
                *goldens_scored += 1;
                if detect(&gmat, pattern) {
                    *golden_detected += 1;
                }
            }
            Ok(_) => eprintln!("warn: empty golden image {}", golden_path.display()),
            Err(e) => eprintln!("warn: missing/unreadable golden {}: {e}", golden_path.display()),
        }
    };

    'replay: loop {
        match evk_rx.recv() {
            Ok(EvkMsg::Events(batch)) => {
                if let Some(last) = batch.events.last() {
                    latest_event_t = last.t_us;
                }
                total_events += batch.events.len() as u64;
                let t0 = Instant::now();
                recon.push_events(&batch);
                push_dur += t0.elapsed();
            }
            Ok(EvkMsg::Trigger(_)) => {}
            Ok(EvkMsg::Eof) => break 'replay,
            Err(_) => break 'replay,
        }

        while let Some(&t) = pending_ts.peek() {
            if t > latest_event_t {
                break;
            }
            pending_ts.next();
            render_and_score(
                &mut recon,
                &mut img,
                t,
                &mut render_samples_ms,
                &mut ours_detected,
                &mut golden_detected,
                &mut goldens_scored,
                &mut rendered,
                &mut max_batch_events,
            );
        }
    }

    while let Some(t) = pending_ts.next() {
        render_and_score(
            &mut recon,
            &mut img,
            t,
            &mut render_samples_ms,
            &mut ours_detected,
            &mut golden_detected,
            &mut goldens_scored,
            &mut rendered,
            &mut max_batch_events,
        );
    }

    let pipeline_elapsed = pipeline_start.elapsed();

    println!();
    println!("rendered {rendered}/{} timestamps, scored {goldens_scored} goldens", timestamps.len());
    println!("ours: {ours_detected}/{rendered} detected, e2calib: {golden_detected}/{goldens_scored} detected");

    let acceptance_pass = ours_detected as f64 >= (golden_detected as f64) / 2.0 && ours_detected >= 20;
    println!(
        "acceptance (X >= Y/2 AND X >= 20): {}",
        if acceptance_pass { "PASS" } else { "FAIL" }
    );

    let perf = percentiles(render_samples_ms);
    println!();
    println!(
        "render_at perf @ iters={}: mean={:.3}ms p50={:.3}ms p95={:.3}ms (n={rendered})",
        args.iters, perf.mean_ms, perf.p50_ms, perf.p95_ms
    );
    println!("perf gate (<33ms mean, 30Hz): {}", if perf.mean_ms < 33.0 { "PASS" } else { "FAIL" });

    let push_render_secs = push_dur.as_secs_f64() + perf.mean_ms * rendered as f64 / 1000.0;
    let mev_push_render = total_events as f64 / 1e6 / push_render_secs;
    let mev_end_to_end = total_events as f64 / 1e6 / pipeline_elapsed.as_secs_f64();
    println!();
    println!(
        "throughput: {total_events} events, push_events={:.3}s render_at={:.3}s => {:.2} Mev/s (push+render only)",
        push_dur.as_secs_f64(),
        perf.mean_ms * rendered as f64 / 1000.0,
        mev_push_render
    );
    println!(
        "throughput (end-to-end incl. SDK decode + channel): {:.2} Mev/s over {:.2}s wall",
        mev_end_to_end,
        pipeline_elapsed.as_secs_f64()
    );

    println!();
    println!(
        "peak single-render batch: {max_batch_events} events (event_decimation_threshold={} -- valve engaged this run: {})",
        recon.event_decimation_threshold,
        max_batch_events > recon.event_decimation_threshold
    );

    println!();
    println!("PNGs written to {}", out_dir.display());

    if !acceptance_pass {
        eprintln!("golden_dancer: acceptance criteria not met (see numbers above)");
    }
}
