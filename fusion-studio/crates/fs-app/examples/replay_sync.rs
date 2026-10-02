use std::sync::{Arc, Mutex};

use crossbeam_channel::bounded;
use fs_core::sync::{SyncConfig, SyncEngine, SyncStats};

use fs_app::pipeline::{sync_loop, PipelineOut};
use fs_app::sources::replay::{spawn_replay_evk, spawn_replay_flir};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).filter(|a| a != "--gpu-decode").collect();
    let gpu_decode = std::env::args().any(|a| a == "--gpu-decode");
    let raw = args.first().cloned().expect("usage: replay_sync <raw> <avi> <fps> [--gpu-decode]");
    let avi = args.get(1).cloned().expect("avi path");
    let fps: f64 = args.get(2).expect("fps").parse().unwrap();

    let (evk_tx, evk_rx) = bounded(256);
    let (flir_tx, flir_rx) = bounded(8);
    let last_error = Arc::new(Mutex::new(None));
    let mut _keepalive = None;
    if gpu_decode {
        println!("decode path: GPU (spawn_gpu_replay_evk)");
        let (_h1, _dims) = fs_app::sources::replay::spawn_gpu_replay_evk(raw, true, evk_tx);
    } else {
        let evk_snapshot = Arc::new(Mutex::new(fs_app::settings::EvkSettingsSnapshot::default()));
        let shutdown = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let rates = Arc::new(fs_app::stream_rates::StreamRates::default());
        let (_h1, cmd, _recon_dims) = spawn_replay_evk(
            raw,
            true,
            evk_tx,
            evk_snapshot,
            rates,
            fs_app::transform::EventXform::default(),
            fs_app::transform::EventRecordTap::default(),
            shutdown,
        );
        _keepalive = Some(cmd);
    }
    let _h2 = spawn_replay_flir(avi, fps, true, flir_tx, last_error);

    let stats = Arc::new(Mutex::new(SyncStats::default()));
    let engine = SyncEngine::new(SyncConfig {
        fps,
        gate_frac: 0.2,
        trigger_polarity: 1,
        max_pending: fs_app::app::REPLAY_MAX_PENDING,
    });
    let (ev_out_tx, ev_out_rx) = bounded(1024);
    let ev_drain_rx = ev_out_rx.clone();
    std::thread::spawn(move || { while ev_drain_rx.recv().is_ok() {} });
    let transform = bounded::<Arc<fs_core::SyncedFrame>>(1024);
    let transform_drain_rx = transform.1.clone();
    std::thread::spawn(move || { while transform_drain_rx.recv().is_ok() {} });

    let out = PipelineOut {
        flir_preview: bounded(2),
        events: (ev_out_tx, ev_out_rx),
        transform,
        stats: stats.clone(),
        events_dropped: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        frames_dropped_pre_transform: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        health: Arc::new(Mutex::new(fs_core::health::SyncHealthMonitor::new().health())),
        clock: Arc::new(Mutex::new(None)),
    };
    let (_pipe_cmd_tx, pipe_cmd_rx) = bounded::<fs_app::pipeline::PipelineCmd>(1);
    let shutdown = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    sync_loop(flir_rx, evk_rx, pipe_cmd_rx, engine, out, shutdown);

    let s = *stats.lock().unwrap();
    println!("matched={} interpolated={} spurious={} drift_ppm={:.1} offset_us={:.0}",
             s.matched, s.interpolated, s.spurious, s.drift_ppm, s.offset_us);
}
