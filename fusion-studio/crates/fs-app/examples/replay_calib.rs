
use std::sync::{Arc, Mutex};

use crossbeam_channel::bounded;
use fs_app::sources::replay::{spawn_replay_evk, spawn_replay_flir};
use fs_calib::board::{BoardConfig, BoardKind};
use fs_core::sync::{SyncConfig, SyncEngine};
use serde_json::Value;

const DEFAULT_RAW: &str = r"C:\Users\98600\Desktop\calibration_dancer_revision\event\recording_2026-08-17_14-55-44.raw";
const DEFAULT_AVI: &str = r"C:\Users\98600\Desktop\calibration_dancer_revision\rgb\rgb-08172026145538-0000.avi";
const DEFAULT_REFERENCE_JSON: &str = r"C:\Users\98600\Desktop\calibration_dancer_revision\tmpst8b6gpy_stereo_calibration.json";
const FPS: f64 = 2.0;

fn main() {
    let gpu_decode = std::env::args().any(|a| a == "--gpu-decode");
    let args: Vec<String> = std::env::args().skip(1).filter(|a| a != "--gpu-decode").collect();
    let (raw, avi, ref_json_path) = match args.len() {
        0 => (DEFAULT_RAW.to_string(), DEFAULT_AVI.to_string(), DEFAULT_REFERENCE_JSON.to_string()),
        3 => (args[0].clone(), args[1].clone(), args[2].clone()),
        _ => {
            eprintln!("usage: replay_calib <raw> <avi> <reference_json> [--gpu-decode]  (or no args to use the dancer defaults)");
            std::process::exit(2);
        }
    };

    println!("=== fs-app replay_calib (Task 3.5: full live pipeline incl. calib worker) ===");
    println!("raw:       {raw}");
    println!("avi:       {avi}");
    println!("reference: {ref_json_path}");
    println!("decode:    {}\n", if gpu_decode { "GPU (spawn_gpu_replay_evk)" } else { "SDK (spawn_replay_evk)" });

    let last_error = Arc::new(Mutex::new(None));
    let (evk_tx, evk_rx) = bounded(256);
    let (flir_tx, flir_rx) = bounded(8);
    let event_xform = fs_app::transform::EventXform::default();
    let event_tap = fs_app::transform::EventRecordTap::default();
    let mut _keepalive = None;
    let recon_dims = if gpu_decode {
        let (_h, dims) = fs_app::sources::replay::spawn_gpu_replay_evk(raw, true, evk_tx);
        dims
    } else {
        let evk_snapshot = Arc::new(Mutex::new(fs_app::settings::EvkSettingsSnapshot::default()));
        let shutdown = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let rates = Arc::new(fs_app::stream_rates::StreamRates::default());
        let (_h, cmd, dims) =
            spawn_replay_evk(raw, true, evk_tx, evk_snapshot, rates, event_xform.clone(), event_tap.clone(), shutdown);
        _keepalive = Some(cmd);
        dims
    };
    spawn_replay_flir(avi, FPS, true, flir_tx, last_error.clone());

    let engine = SyncEngine::new(SyncConfig {
        fps: FPS,
        gate_frac: 0.2,
        trigger_polarity: 1,
        max_pending: fs_app::app::REPLAY_MAX_PENDING,
    });
    let (_pipe_cmd_tx, pipe_cmd_rx) = bounded::<fs_app::pipeline::PipelineCmd>(1);
    let tuning = fs_app::settings::SharedTuning::new(FPS, 0);
    let shutdown = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let handles = fs_app::wiring::spawn_pipeline(
        engine,
        flir_rx,
        evk_rx,
        pipe_cmd_rx,
        tuning,
        recon_dims,
        last_error,
        None,
        event_xform,
        event_tap,
        shutdown,
        Arc::new(fs_app::stream_rates::StreamRates::default()),
    );

    {
        let mut c = handles.calib_state.lock().unwrap();
        c.enabled = true;
        c.board = BoardConfig {
            kind: BoardKind::Checkerboard,
            inner_cols: 7,
            inner_rows: 6,
            rect_w_mm: 82.0,
            rect_h_mm: 98.0,
        };
        if std::env::var("FS_CALIB_DETECT_SPEED").as_deref() == Ok("full") {
            c.detect_speed = fs_calib::board::DetectSpeed::Full;
            println!("FS_CALIB_DETECT_SPEED=full -- overriding detect_speed to Full for this run\n");
        }
    }

    println!("running to Eof (realtime replay @ {FPS} fps -- ~2 minutes for the dancer capture)...\n");
    handles.sync_loop_handle.join().expect("sync_loop thread panicked");
    handles.calib_loop_handle.join().expect("calib worker thread panicked");

    let c = handles.calib_state.lock().unwrap();

    println!("================ SESSION SUMMARY ================");
    println!(
        "synced frames seen: {} | recon-pairing mismatches: {} | cam0 detections: {} | cam1 detections: {} | both: {}",
        c.synced_seen, c.recon_pair_mismatches, c.det0_count, c.det1_count, c.both_count
    );
    println!(
        "detect ms/attempt: cam0={:.2} (n={}) cam1={:.2} (n={}) | prefilter rejects: cam0={} cam1={} | throttle skipped: {}",
        mean_ms(c.det0_ms_sum, c.det0_attempts),
        c.det0_attempts,
        mean_ms(c.det1_ms_sum, c.det1_attempts),
        c.det1_attempts,
        c.prefilter_rejects0,
        c.prefilter_rejects1,
        c.throttle_skipped
    );
    println!("background calib publishes applied: {} | still in flight at exit: {}", c.calib_published, c.calib_in_flight);
    if c.synced_seen == 0 {
        println!("WARNING: zero synced frames seen -- the sync engine likely never matched a trigger to a frame.");
    } else if c.det0_count == 0 && c.det1_count == 0 {
        println!(
            "WARNING: zero detections on both cameras despite {} synced frames -- check the recon-pairing tag \
             tolerance (recon_pair_mismatches={}) or the FLIR gray conversion (Bgr8 path) before assuming the \
             board itself is undetectable.",
            c.synced_seen, c.recon_pair_mismatches
        );
    }
    println!(
        "cam0: pool {}/{} rms={} converged={} fx={:.3} fy={:.3} cx={:.3} cy={:.3}",
        c.cam0.pool_len,
        c.cam0.cap,
        fmt_opt(c.cam0.rms),
        c.cam0.converged,
        c.cam0.fx,
        c.cam0.fy,
        c.cam0.cx,
        c.cam0.cy
    );
    println!(
        "cam1: pool {}/{} rms={} converged={} fx={:.3} fy={:.3} cx={:.3} cy={:.3}",
        c.cam1.pool_len,
        c.cam1.cap,
        fmt_opt(c.cam1.rms),
        c.cam1.converged,
        c.cam1.fx,
        c.cam1.fy,
        c.cam1.cx,
        c.cam1.cy
    );
    match (&c.stereo, &c.stereo_r) {
        (Some((rms, angle_deg, t)), Some(_)) => {
            println!(
                "stereo: rms={rms:.4}px angle(from identity)={angle_deg:.4}deg t={t:?}mm |t|={:.3}mm",
                vec3_norm(*t)
            );
        }
        _ => println!("stereo: NOT CALIBRATED"),
    }

    let ref_text = std::fs::read_to_string(&ref_json_path).unwrap_or_else(|e| panic!("failed to read reference json '{ref_json_path}': {e}"));
    let reference: Value = serde_json::from_str(&ref_text).expect("reference json must parse");

    println!("\n================ ACCEPTANCE vs REFERENCE ================");
    let mut all_pass = true;
    let intrinsics_threshold_pct = 5.0;
    let rotation_threshold_deg = 2.0;

    for (label, fx, fy, cx, cy, rms_opt) in [
        ("cam0", c.cam0.fx, c.cam0.fy, c.cam0.cx, c.cam0.cy, c.cam0.rms),
        ("cam1", c.cam1.fx, c.cam1.fy, c.cam1.cx, c.cam1.cy, c.cam1.rms),
    ] {
        if rms_opt.is_none() {
            println!("[FAIL] {label}: cannot compute deltas -- camera never calibrated");
            all_pass = false;
            continue;
        }
        let (fx_ref, fy_ref, cx_ref, cy_ref) = k_from_ref(&reference["cameras"][label]["K"]);
        for (name, est, refv) in [("fx", fx, fx_ref), ("fy", fy, fy_ref), ("cx", cx, cx_ref), ("cy", cy, cy_ref)] {
            let delta_pct = pct_delta(est, refv);
            let pass = delta_pct <= intrinsics_threshold_pct;
            all_pass &= pass;
            println!(
                "[{}] {label}.{name}: est={est:.3} ref={refv:.3} delta={delta_pct:.2}% (threshold {intrinsics_threshold_pct}%)",
                if pass { "PASS" } else { "FAIL" }
            );
        }
    }

    match &c.stereo_r {
        Some(r_est) => {
            let (_, _, t_est) = c.stereo.expect("stereo_r implies stereo");
            let r_ref = mat3_from_ref(&reference["extrinsics"]["R_cam0_to_cam1"]);
            let t_ref = vec3_from_ref(&reference["extrinsics"]["T_cam0_to_cam1"]);

            let rot_err_deg = rotation_angle_deg(r_est, &r_ref);
            let dir_err_deg = direction_angle_deg(t_est, t_ref);

            let rot_pass = rot_err_deg <= rotation_threshold_deg;
            all_pass &= rot_pass;
            println!(
                "[{}] stereo rotation angle(R_est, R_ref) = {rot_err_deg:.4}deg (threshold {rotation_threshold_deg}deg)",
                if rot_pass { "PASS" } else { "FAIL" }
            );

            println!(
                "[INFO] stereo translation direction angle(T_est, T_ref) = {dir_err_deg:.4}deg (not a pass/fail criterion) \
                 |T_est|={:.3}mm |T_ref|={:.3}mm",
                vec3_norm(t_est),
                vec3_norm(t_ref)
            );
        }
        None => {
            println!("[FAIL] stereo: cannot compute deltas -- stereo never calibrated");
            all_pass = false;
        }
    }

    println!("\n================ OVERALL: {} ================", if all_pass { "PASS" } else { "FAIL" });
    std::process::exit(if all_pass { 0 } else { 1 });
}

fn fmt_opt(v: Option<f64>) -> String {
    v.map(|x| format!("{x:.4}px")).unwrap_or_else(|| "-".to_string())
}

fn mean_ms(sum: f64, n: u64) -> f64 {
    if n == 0 {
        0.0
    } else {
        sum / n as f64
    }
}

fn pct_delta(est: f64, refv: f64) -> f64 {
    (est - refv).abs() / refv.abs() * 100.0
}

fn vec3_norm(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn direction_angle_deg(a: [f64; 3], b: [f64; 3]) -> f64 {
    let dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let cos_dir = (dot / (vec3_norm(a) * vec3_norm(b))).clamp(-1.0, 1.0);
    cos_dir.acos().to_degrees()
}


fn rotation_angle_deg(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> f64 {
    let mut trace = 0.0;
    for i in 0..3 {
        for k in 0..3 {
            trace += a[k][i] * b[k][i];
        }
    }
    let cos_angle = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0);
    cos_angle.acos().to_degrees()
}

fn k_from_ref(k: &Value) -> (f64, f64, f64, f64) {
    let fx = k[0][0].as_f64().expect("K[0][0]");
    let fy = k[1][1].as_f64().expect("K[1][1]");
    let cx = k[0][2].as_f64().expect("K[0][2]");
    let cy = k[1][2].as_f64().expect("K[1][2]");
    (fx, fy, cx, cy)
}

fn mat3_from_ref(v: &Value) -> [[f64; 3]; 3] {
    let mut m = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            m[i][j] = v[i][j].as_f64().expect("R entry");
        }
    }
    m
}

fn vec3_from_ref(v: &Value) -> [f64; 3] {
    let mut out = [0.0; 3];
    for (i, o) in out.iter_mut().enumerate() {
        *o = v[i].as_f64().expect("T entry");
    }
    out
}
