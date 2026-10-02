use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use crossbeam_channel::{bounded, Receiver, RecvTimeoutError, TrySendError};
use fs_core::bus::send_latest;
use fs_core::clock::ClockParams;
use fs_core::health::{SyncHealth, SyncHealthMonitor};
use fs_core::sync::{SyncEngine, SyncStats};
use fs_core::{EventBatch, GrayImage, RgbFrame, SyncedFrame};

use crate::settings::SharedTuning;
use crate::stream_rates::StreamRates;
use crate::{calib_worker, decode_stage, pipeline, preview, recon_thread, record_worker, transform};

pub struct PipelineHandles {
    pub flir_img: Receiver<egui::ColorImage>,
    pub recon_img: Receiver<egui::ColorImage>,
    pub stats: Arc<Mutex<SyncStats>>,

    pub health: Arc<Mutex<SyncHealth>>,

    pub clock: Arc<Mutex<Option<ClockParams>>>,
    pub paired: Arc<AtomicBool>,
    pub last_error: Arc<Mutex<Option<String>>>,
    pub calib_state: Arc<Mutex<calib_worker::CalibUiState>>,
    pub recon: recon_thread::ReconHandles,
    pub sync_loop_handle: JoinHandle<()>,

    pub calib_loop_handle: JoinHandle<()>,

    pub record: RecordChannel,
    pub record_state: Arc<Mutex<record_worker::RecordUiState>>,
    pub rates: Arc<StreamRates>,
    pub decode_path: &'static str,
    pub geom_maps: Arc<Mutex<Option<transform::CamMapsPair>>>,
    pub event_xform: transform::EventXform,
    pub event_tap: transform::EventRecordTap,

    pub frames_dropped_pre_transform: Arc<std::sync::atomic::AtomicU64>,

    pub host_downloads: Arc<std::sync::atomic::AtomicU64>,
}

pub struct RecordChannel {

    cmd: Option<crossbeam_channel::Sender<record_worker::RecordCmd>>,
    join: Option<JoinHandle<()>>,
}

impl RecordChannel {

    pub fn send(&self, cmd: record_worker::RecordCmd) -> Result<(), String> {
        let Some(tx) = self.cmd.as_ref() else {
            return Err("录制线程已经停止".to_string());
        };
        tx.try_send(cmd).map_err(|e| match e {
            TrySendError::Full(_) => "录制线程忙(命令还没处理完),请稍后再试".to_string(),
            TrySendError::Disconnected(_) => "录制线程已经退出".to_string(),
        })
    }


    pub fn shutdown(&mut self) {
        drop(self.cmd.take());
        if let Some(h) = self.join.take() {
            let _ = h.join();
        }
    }
}

impl Drop for RecordChannel {
    fn drop(&mut self) {
        self.shutdown();
    }
}

pub fn spawn_pipeline(
    engine: SyncEngine,
    flir_rx: crossbeam_channel::Receiver<fs_core::bus::FlirMsg>,
    evk_rx: crossbeam_channel::Receiver<fs_core::bus::EvkMsg>,
    pipeline_cmd_rx: crossbeam_channel::Receiver<pipeline::PipelineCmd>,
    tuning: SharedTuning,
    recon_dims: (u32, u32),
    last_error: Arc<Mutex<Option<String>>>,
    gpu_raw: Option<decode_stage::GpuRawIntake>,
    event_xform: transform::EventXform,
    event_tap: transform::EventRecordTap,
    shutdown: Arc<AtomicBool>,
    rates: Arc<StreamRates>,
) -> PipelineHandles {
    let stats = Arc::new(Mutex::new(SyncStats::default()));
    let health = Arc::new(Mutex::new(SyncHealthMonitor::new().health()));
    let clock = Arc::new(Mutex::new(None));
    let paired = Arc::new(AtomicBool::new(false));
    let events_dropped = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let frames_dropped_pre_transform = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let host_downloads = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let flir_preview = bounded(2);
    let events: (crossbeam_channel::Sender<EventBatch>, Receiver<EventBatch>) = bounded(512);
    let transform_in: (crossbeam_channel::Sender<Arc<SyncedFrame>>, Receiver<Arc<SyncedFrame>>) = bounded(64);
    let synced: (crossbeam_channel::Sender<Arc<SyncedFrame>>, Receiver<Arc<SyncedFrame>>) = bounded(64);
    let calib_synced: (crossbeam_channel::Sender<Arc<SyncedFrame>>, Receiver<Arc<SyncedFrame>>) = bounded(64);

    let flir_raw_rx = flir_preview.1.clone();
    let events_rx = events.1.clone();
    let pairing_synced_rx = synced.1.clone();
    let calib_rx = calib_synced.1.clone();

    let (record_tx, record_rx) = record_worker::record_channel(record_worker::RECORD_QUEUE_BYTES);
    let record_state = Arc::new(Mutex::new(record_worker::RecordUiState::default()));
    let (record_cmd, record_cmd_rx) = bounded::<record_worker::RecordCmd>(8);
    let record_join = std::thread::spawn({
        let record_state = record_state.clone();
        let clock = clock.clone();
        let taps = record_worker::RecordTaps {
            tap: event_tap.clone(),
            oob_dropped: event_xform.oob_dropped.clone(),
            frames_dropped_pre_transform: frames_dropped_pre_transform.clone(),
        };
        move || record_worker::run(record_cmd_rx, record_rx, record_state, clock, taps)
    });
    let record = RecordChannel { cmd: Some(record_cmd), join: Some(record_join) };

    let geom_maps: Arc<Mutex<Option<transform::CamMapsPair>>> = Arc::new(Mutex::new(None));
    transform::spawn_frame_transform(
        transform_in.1.clone(),
        transform::FrameFanout { synced: synced.clone(), calib: calib_synced.clone(), record: Some(record_tx) },
        geom_maps.clone(),
        shutdown.clone(),
    );

    let out = pipeline::PipelineOut {
        flir_preview: flir_preview.clone(),
        events: events.clone(),
        transform: transform_in.clone(),
        stats: stats.clone(),
        events_dropped: events_dropped.clone(),
        frames_dropped_pre_transform: frames_dropped_pre_transform.clone(),
        health: health.clone(),
        clock: clock.clone(),
    };
    let sync_loop_handle = {
        let shutdown = shutdown.clone();
        std::thread::spawn(move || pipeline::sync_loop(flir_rx, evk_rx, pipeline_cmd_rx, engine, out, shutdown))
    };

    let (conv_tx, conv_rx) = bounded::<RgbFrame>(2);
    let conv_rx_for_drop = conv_rx.clone();
    let (req_tx, req_rx) = bounded::<i64>(2);
    let req_rx_for_drop = req_rx.clone();
    {
        let paired = paired.clone();
        let tuning = tuning.clone();
        let rates = rates.clone();
        std::thread::spawn(move || loop {
            match pairing_synced_rx.recv_timeout(Duration::from_millis(500)) {
                Ok(s) => {
                    paired.store(true, Ordering::Relaxed);
                    rates.observe_frame(&s.frame);
                    let t = s.recon_time_us(tuning.exposure_us());
                    send_latest(&conv_tx, &conv_rx_for_drop, s.frame.clone());
                    send_latest(&req_tx, &req_rx_for_drop, t);
                }
                Err(RecvTimeoutError::Timeout) => {
                    paired.store(false, Ordering::Relaxed);
                    if let Some(f) = flir_raw_rx.try_iter().last() {
                        match transform::to_rgb8(&f) {
                            Ok(rgb) => {
                                send_latest(&conv_tx, &conv_rx_for_drop, rgb);
                            }
                            Err(e) => {
                                static FALLBACK_CONV_LOGGED: AtomicBool = AtomicBool::new(false);
                                if !FALLBACK_CONV_LOGGED.swap(true, Ordering::Relaxed) {
                                    eprintln!("preview-fallback: 原始帧转换失败({e});丢帧");
                                }
                            }
                        }
                    }
                }
                Err(RecvTimeoutError::Disconnected) => break,
            }
        });
    }

    let flir_img = bounded(2);
    std::thread::spawn({
        let (tx, tx_rx) = flir_img.clone();
        let last_error = last_error.clone();
        move || preview::run(conv_rx, tx, tx_rx, last_error)
    });

    let recon_img = bounded(2);
    let recon_gray: (crossbeam_channel::Sender<(GrayImage, i64)>, Receiver<(GrayImage, i64)>) = bounded(8);
    let recon_gray_for_calib = recon_gray.1.clone();

    let (w, h) = recon_dims;
    let (initial_recon, initial_choice, initial_error) = recon_thread::default_reconstructor(w, h);
    match &initial_error {
        Some(e) => eprintln!("recon: cuda-manifold init failed ({e}); falling back to accumulator (window_us=150000)"),
        None => println!("recon: using cuda-manifold (iters=2, tau_leak_us=2e6, tonemap_scale=400, TileLocal, median_filter=true), dims={w}x{h}"),
    }

    let (tx_swap, rx_swap) = bounded::<recon_thread::ReconSwap>(2);
    let (tx_params, rx_params) = bounded::<recon_thread::ParamsMsg>(16);
    let paired_only = Arc::new(AtomicBool::new(false));
    let auto_degrade = Arc::new(AtomicBool::new(true));
    let recon_stats = Arc::new(Mutex::new(recon_thread::ReconStats::default()));

    let decode_path: &'static str = if gpu_raw.is_some() { "GPU解码" } else { "CPU解码" };
    let intake = match gpu_raw {
        Some(g) => {
            let dev = bounded::<decode_stage::DecodedBatch>(8);
            let host = bounded::<EventBatch>(512);
            decode_stage::spawn_decode_stage(
                g.rx_raw,
                g.decoder,
                decode_stage::DecodeFanout {
                    dev_tx: dev.0,
                    dev_rx_for_drop: dev.1.clone(),
                    host_recon_tx: host.0,
                    host_recon_rx_for_drop: host.1.clone(),
                    tap: event_tap.clone(),
                    trig_tx: g.trig_tx,
                    xform: event_xform.clone(),
                    events_dropped: events_dropped.clone(),
                    host_downloads: host_downloads.clone(),
                },
                shutdown.clone(),
            );
            recon_thread::EvkIntake::GpuRaw(recon_thread::DecodedIntake { rx_dev: dev.1, rx_host: host.1 })
        }
        None => recon_thread::EvkIntake::Sdk(events_rx),
    };
    std::thread::spawn({
        let (tx, tx_rx) = recon_img.clone();
        let (gtx, gtx_rx) = recon_gray.clone();
        let outputs = recon_thread::ReconOutputs { img: (tx, tx_rx), gray: (gtx, gtx_rx), stats: recon_stats.clone() };
        let ctrl = recon_thread::ReconCtrl {
            rx_swap,
            rx_params,
            events_dropped: events_dropped.clone(),
            auto_degrade: auto_degrade.clone(),
            paired_only: paired_only.clone(),
            shutdown: shutdown.clone(),
        };
        let tuning = tuning.clone();
        move || recon_thread::run(intake, req_rx, ctrl, outputs, initial_recon, w, h, tuning)
    });

    let recon_handles = recon_thread::ReconHandles {
        tx_swap,
        tx_params,
        paired_only,
        auto_degrade,
        stats: recon_stats,
        initial_choice,
        initial_error,
    };

    let calib_state = Arc::new(Mutex::new(calib_worker::CalibUiState::default()));
    let calib_loop_handle = {
        let calib_state = calib_state.clone();
        let tuning = tuning.clone();
        std::thread::spawn(move || calib_worker::run(calib_rx, recon_gray_for_calib, calib_state, tuning, recon_dims))
    };

    PipelineHandles {
        flir_img: flir_img.1,
        recon_img: recon_img.1,
        stats,
        health,
        clock,
        paired,
        last_error,
        calib_state,
        recon: recon_handles,
        record,
        record_state,
        rates,
        sync_loop_handle,
        calib_loop_handle,
        decode_path,
        geom_maps,
        event_xform,
        event_tap,
        frames_dropped_pre_transform,
        host_downloads,
    }
}
