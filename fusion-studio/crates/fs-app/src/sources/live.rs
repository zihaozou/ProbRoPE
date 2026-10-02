use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crossbeam_channel::{bounded, Receiver, RecvTimeoutError, Sender};
use fs_core::bus::{send_latest, EvkMsg, EvkRawMsg, FlirMsg};
use fs_core::{PixelFormat, RgbFrame, TriggerEvent};
use fs_recon::CpuTriggerScanner;
use metavision_sys::evt3_raw::bytes_to_words;
use metavision_sys::MvCamera;
use spinnaker_sys::wrapper::{SpinCamera, SpinSystem};

use super::evk_thread::{self, EvkCmd};
use crate::pipeline::PipelineCmd;
use crate::settings::{
    AutoMode, EvkSettingsSnapshot, FlirSettingsSnapshot, FpsApplyState, SharedTuning, CONNECT_REVISION,
};

pub struct LiveConfig {
    pub fps: f64,
    pub exposure_us: i64,
    pub gain_db: f64,
    pub trigger_polarity: i8,
    pub shutdown: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

pub use super::flir_cmd::{coalesce_flir_cmds, FlirCmd};

pub const GRAB_TIMEOUT_MS: u64 = 2000;


pub const HANDSHAKE_TRIGGER_WAIT: Duration = Duration::from_secs(2);

const INTERPOSER_POLL: Duration = Duration::from_millis(50);


pub fn configure_flir(cam: &SpinCamera, cfg: &LiveConfig) -> Result<(), String> {
    cam.set_enum("AcquisitionMode", "Continuous")?;
    if let Err(e) = cam.set_enum("PixelFormat", "BayerRG8") {
        eprintln!("warn: PixelFormat BayerRG8 not set ({e}); verify camera default is 8-bit Bayer");
    }
    cam.set_enum("ExposureAuto", "Off")?;
    cam.set_float("ExposureTime", cfg.exposure_us as f64)?;
    cam.set_enum("GainAuto", "Off")?;
    cam.set_float("Gain", cfg.gain_db)?;
    cam.set_bool("AcquisitionFrameRateEnable", true)?;
    cam.set_float("AcquisitionFrameRate", cfg.fps)?;
    cam.set_enum("LineSelector", "Line1")?;
    cam.set_enum("LineMode", "Output")?;
    cam.set_enum("LineSource", "ExposureActive")?;
    Ok(())
}

pub struct LiveHandles {
    pub evk_cmd: Sender<EvkCmd>,

    pub flir_cmd: Sender<FlirCmd>,
}


pub struct LiveSettingsHooks {
    pub flir_snapshot: Arc<Mutex<FlirSettingsSnapshot>>,
    pub evk_snapshot: Arc<Mutex<EvkSettingsSnapshot>>,

    pub pipeline_cmd_tx: Sender<PipelineCmd>,

    pub tuning: SharedTuning,
}


#[derive(Clone)]
struct TrigTap {
    enabled: Arc<AtomicBool>,
    rising_seen: Arc<AtomicU64>,
    falling_seen: Arc<AtomicU64>,
}

fn handshake(
    cam: &SpinCamera,
    trig_seen_rx: &Receiver<TriggerEvent>,
    tap: &TrigTap,
    trigger_polarity: i8,
) -> Result<(i64, i64), String> {
    let rising0 = tap.rising_seen.load(Ordering::Relaxed);
    let falling0 = tap.falling_seen.load(Ordering::Relaxed);
    tap.enabled.store(true, Ordering::Relaxed);
    let result = (|| {
        if let Err(e) = cam.execute("TimestampReset") {
            eprintln!("warn: TimestampReset failed ({e}); RLS offset absorbs it");
        }
        std::thread::sleep(Duration::from_millis(200));
        while trig_seen_rx.try_recv().is_ok() {}
        cam.execute("TriggerSoftware")?;
        let img = cam.next_image(GRAB_TIMEOUT_MS)?;
        let t_flir_us = (img.t_ns / 1000) as i64;
        let tr = trig_seen_rx.recv_timeout(HANDSHAKE_TRIGGER_WAIT).map_err(|_| {
            let r = tap.rising_seen.load(Ordering::Relaxed) - rising0;
            let f = tap.falling_seen.load(Ordering::Relaxed) - falling0;
            format!(
                "handshake: no polarity-{trigger_polarity} trigger within 2s (saw {r} rising / {f} falling \
                 -- if falling>0, retry with --trigger-polarity 0; if both 0, check wiring/trigger-in enable)"
            )
        })?;
        Ok((t_flir_us, tr.t_us))
    })();
    tap.enabled.store(false, Ordering::Relaxed);
    result
}

struct AcquisitionGuard<F: FnMut()> {
    cleanup: F,
    disarmed: bool,
}

impl<F: FnMut()> AcquisitionGuard<F> {

    fn arm(cleanup: F) -> Self {
        AcquisitionGuard { cleanup, disarmed: false }
    }

    fn disarm(mut self) {
        self.disarmed = true;
    }
}

impl<F: FnMut()> Drop for AcquisitionGuard<F> {
    fn drop(&mut self) {
        if !self.disarmed {
            (self.cleanup)();
        }
    }
}


pub fn start_live(
    cfg: LiveConfig,
    evk_tx: Sender<EvkMsg>,
    flir_tx: Sender<FlirMsg>,
    hooks: LiveSettingsHooks,
    raw_out: Option<(Sender<EvkRawMsg>, Receiver<EvkRawMsg>)>,
    rates: Arc<crate::stream_rates::StreamRates>,
    xform: crate::transform::EventXform,
    record_tap: crate::transform::EventRecordTap,
) -> Result<(LiveHandles, (i64, i64), (u32, u32)), String> {
    let trigger_polarity = cfg.trigger_polarity;

    let evk = MvCamera::open_live()?;
    let geometry = match evk.geometry() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("warn: EVK4 geometry query failed ({e}); falling back to 1280x720");
            (1280, 720)
        }
    };
    evk.enable_trigger_in()?;
    let (evk_cmd_tx, evk_cmd_rx) = bounded::<EvkCmd>(16);
    let (trig_seen_tx, trig_seen_rx) = bounded::<TriggerEvent>(16);

    let forwarding = Arc::new(AtomicBool::new(false));
    let tap = TrigTap {
        enabled: Arc::new(AtomicBool::new(false)),
        rising_seen: Arc::new(AtomicU64::new(0)),
        falling_seen: Arc::new(AtomicU64::new(0)),
    };
    match raw_out {
        None => {
            let (tap_tx, tap_rx) = bounded::<EvkMsg>(1024);
            {
                let evk_snapshot = hooks.evk_snapshot.clone();
                let shutdown = cfg.shutdown.clone();
                let rates = rates.clone();
                std::thread::spawn(move || {
                    evk_thread::run(evk, tap_tx, evk_cmd_rx, evk_snapshot, rates, xform, record_tap, shutdown)
                });
            }
            let forwarding = forwarding.clone();
            let tap = tap.clone();
            let trig_seen_tx = trig_seen_tx.clone();
            let shutdown = cfg.shutdown.clone();
            std::thread::spawn(move || loop {
                if shutdown.load(Ordering::Relaxed) {
                    return;
                }
                match tap_rx.recv_timeout(INTERPOSER_POLL) {
                    Ok(msg) => {
                        if let EvkMsg::Trigger(t) = &msg {
                            if t.polarity == 1 {
                                tap.rising_seen.fetch_add(1, Ordering::Relaxed);
                            } else {
                                tap.falling_seen.fetch_add(1, Ordering::Relaxed);
                            }
                            if tap.enabled.load(Ordering::Relaxed) && t.polarity == trigger_polarity {
                                let _ = trig_seen_tx.try_send(*t);
                            }
                        }
                        if forwarding.load(Ordering::Relaxed) {
                            if evk_tx.send(msg).is_err() {
                                return;
                            }
                        }
                    }
                    Err(RecvTimeoutError::Timeout) => {}
                    Err(RecvTimeoutError::Disconnected) => return,
                }
            });
        }
        Some((raw_fwd_tx, raw_fwd_rx_for_drop)) => {
            let (raw_tap_tx, raw_tap_rx) = bounded::<EvkRawMsg>(256);
            let raw_tap_rx_for_drop = raw_tap_rx.clone();
            {
                let evk_snapshot = hooks.evk_snapshot.clone();
                let shutdown = cfg.shutdown.clone();
                std::thread::spawn(move || {
                    evk_thread::run_raw(evk, raw_tap_tx, raw_tap_rx_for_drop, evk_tx, evk_cmd_rx, evk_snapshot, rates, shutdown)
                });
            }
            let forwarding = forwarding.clone();
            let tap = tap.clone();
            let trig_seen_tx = trig_seen_tx.clone();
            let shutdown = cfg.shutdown.clone();
            std::thread::spawn(move || {
                let mut scanner = CpuTriggerScanner::default();
                let mut words: Vec<u16> = Vec::new();
                loop {
                    if shutdown.load(Ordering::Relaxed) {
                        return;
                    }
                    match raw_tap_rx.recv_timeout(INTERPOSER_POLL) {
                        Ok(EvkRawMsg::Bytes(bytes)) => {
                            debug_assert!(bytes.len() % 2 == 0, "EvkRawMsg::Bytes must be whole EVT3 words");
                            words.clear();
                            bytes_to_words(&mut words, &mut None, &bytes);
                            for tr in scanner.push(&words) {
                                if tr.polarity == 1 {
                                    tap.rising_seen.fetch_add(1, Ordering::Relaxed);
                                } else {
                                    tap.falling_seen.fetch_add(1, Ordering::Relaxed);
                                }
                                if tap.enabled.load(Ordering::Relaxed) && tr.polarity as i8 == trigger_polarity {
                                    let _ = trig_seen_tx
                                        .try_send(TriggerEvent { t_us: tr.t_us, polarity: tr.polarity as i8 });
                                }
                            }
                            if forwarding.load(Ordering::Relaxed) {
                                send_latest(&raw_fwd_tx, &raw_fwd_rx_for_drop, EvkRawMsg::Bytes(bytes));
                            }
                        }
                        Ok(EvkRawMsg::ResetState) => {
                            scanner.reset();
                            let _ = raw_fwd_tx.send(EvkRawMsg::ResetState);
                        }
                        Err(RecvTimeoutError::Timeout) => {}
                        Err(RecvTimeoutError::Disconnected) => return,
                    }
                }
            });
        }
    }

    let sys = SpinSystem::new()?;
    if sys.camera_count()? == 0 {
        return Err("no FLIR camera found".into());
    }
    let cam = sys.camera(0)?;
    configure_flir(&cam, &cfg)?;
    cam.set_enum("TriggerMode", "On")?;
    cam.set_enum("TriggerSource", "Software")?;
    cam.begin_acquisition()?;
    let acq_guard = AcquisitionGuard::arm(|| {
        let _ = cam.end_acquisition();
    });
    let seed = handshake(&cam, &trig_seen_rx, &tap, trigger_polarity)?;
    eprintln!(
        "handshake ok: t_flir={} t_evk={} offset={}",
        seed.0,
        seed.1,
        seed.1 - seed.0
    );
    forwarding.store(true, Ordering::Relaxed);

    {
        let snap = read_flir_snapshot(&cam, &cfg);
        hooks.tuning.set_fps(snap.fps);
        hooks.tuning.set_exposure_us(snap.exposure_us as i64);
        *hooks.flir_snapshot.lock().unwrap() = snap;
    }

    cam.set_enum("TriggerMode", "Off")?;
    let (flir_cmd_tx, flir_cmd_rx) = bounded::<FlirCmd>(16);
    let ctx = FlirThreadCtx {
        snapshot: hooks.flir_snapshot,
        pipeline_cmd_tx: hooks.pipeline_cmd_tx,
        tuning: hooks.tuning,
        trig_seen_rx,
        tap,
        trigger_polarity,
        shutdown: cfg.shutdown.clone(),
    };
    acq_guard.disarm();
    std::thread::spawn(move || {
        let _sys = sys;
        let mut seq: u64 = 0;
        let mut last_poll = Instant::now();
        loop {
            if ctx.shutdown.load(Ordering::Relaxed) {
                break;
            }
            let mut queued = Vec::new();
            while let Ok(cmd) = flir_cmd_rx.try_recv() {
                queued.push(cmd);
            }
            if ctx.shutdown.load(Ordering::Relaxed) {
                break;
            }
            for cmd in coalesce_flir_cmds(queued) {
                apply_flir_cmd(&cam, cmd, &ctx);
            }
            if last_poll.elapsed() >= Duration::from_secs(2) {
                poll_flir_readback(&cam, &ctx);
                last_poll = Instant::now();
            }
            match cam.next_image(GRAB_TIMEOUT_MS) {
                Ok(img) => {
                    let frame = RgbFrame {
                        seq,
                        t_cam_us: (img.t_ns / 1000) as i64,
                        w: img.w,
                        h: img.h,
                        data: img.data,
                        format: PixelFormat::Bayer8,
                    };
                    seq += 1;
                    if flir_tx.send(FlirMsg::Frame(frame)).is_err() {
                        break;
                    }
                }
                Err(e) => {
                    if e.contains("spinError -1011") || e.contains("incomplete image") {
                        continue;
                    }
                    eprintln!("flir grab error: {e}");
                    let _ = flir_tx.send(FlirMsg::Eof);
                    break;
                }
            }
        }
        let _ = cam.end_acquisition();
    });

    Ok((LiveHandles { evk_cmd: evk_cmd_tx, flir_cmd: flir_cmd_tx }, seed, geometry))
}

struct FlirThreadCtx {
    snapshot: Arc<Mutex<FlirSettingsSnapshot>>,
    pipeline_cmd_tx: Sender<PipelineCmd>,
    tuning: SharedTuning,
    trig_seen_rx: Receiver<TriggerEvent>,
    tap: TrigTap,
    trigger_polarity: i8,
    shutdown: Arc<AtomicBool>,
}

fn read_flir_snapshot(cam: &SpinCamera, cfg: &LiveConfig) -> FlirSettingsSnapshot {
    use crate::settings::{FLIR_EXPOSURE_RANGE_US, FLIR_FPS_RANGE, FLIR_GAIN_RANGE_DB};
    FlirSettingsSnapshot {
        populated: true,
        exposure_us: cam.get_float("ExposureTime").unwrap_or(cfg.exposure_us as f64),
        exposure_range_us: cam.float_range("ExposureTime").unwrap_or(FLIR_EXPOSURE_RANGE_US),
        exposure_auto: read_auto(cam, "ExposureAuto"),
        gain_db: cam.get_float("Gain").unwrap_or(cfg.gain_db),
        gain_range_db: cam.float_range("Gain").unwrap_or(FLIR_GAIN_RANGE_DB),
        gain_auto: read_auto(cam, "GainAuto"),
        fps: cam.get_float("AcquisitionFrameRate").unwrap_or(cfg.fps),
        fps_range: cam.float_range("AcquisitionFrameRate").unwrap_or(FLIR_FPS_RANGE),
        isp_enable: cam.get_bool("IspEnable").unwrap_or(false),
        temperature_c: cam.get_float("DeviceTemperature").ok(),
        fps_apply: FpsApplyState::Idle,
        last_error: None,
        revision: CONNECT_REVISION,
    }
}

fn read_auto(cam: &SpinCamera, node: &str) -> AutoMode {
    cam.get_enum(node).ok().and_then(|s| AutoMode::from_node_entry(&s)).unwrap_or(AutoMode::Off)
}

fn poll_flir_readback(cam: &SpinCamera, ctx: &FlirThreadCtx) {
    let temp = cam.get_float("DeviceTemperature").ok();
    let (exp_auto_on, gain_auto_on) = {
        let s = ctx.snapshot.lock().unwrap();
        (s.exposure_auto != AutoMode::Off, s.gain_auto != AutoMode::Off)
    };
    let exp = if exp_auto_on { cam.get_float("ExposureTime").ok().map(|v| (v, read_auto(cam, "ExposureAuto"))) } else { None };
    let gain = if gain_auto_on { cam.get_float("Gain").ok().map(|v| (v, read_auto(cam, "GainAuto"))) } else { None };

    if let Some((v, _)) = exp {
        ctx.tuning.set_exposure_us(v as i64);
    }
    let mut s = ctx.snapshot.lock().unwrap();
    if let Some(t) = temp {
        s.temperature_c = Some(t);
    }
    if let Some((v, auto)) = exp {
        s.exposure_us = v;
        s.exposure_auto = auto;
    }
    if let Some((v, auto)) = gain {
        s.gain_db = v;
        s.gain_auto = auto;
    }
}


fn apply_flir_cmd(cam: &SpinCamera, cmd: FlirCmd, ctx: &FlirThreadCtx) {
    match cmd {
        FlirCmd::SetExposureUs(v) => {
            let res = cam.set_float("ExposureTime", v);
            let actual = cam.get_float("ExposureTime").ok();
            if let Some(a) = actual {
                ctx.tuning.set_exposure_us(a as i64);
            }
            finish_apply(ctx, res, |s| {
                if let Some(a) = actual {
                    s.exposure_us = a;
                }
            });
        }
        FlirCmd::SetExposureAuto(mode) => {
            let res = cam.set_enum("ExposureAuto", mode.node_entry());
            let auto = read_auto(cam, "ExposureAuto");
            let actual = cam.get_float("ExposureTime").ok();
            if let Some(a) = actual {
                ctx.tuning.set_exposure_us(a as i64);
            }
            finish_apply(ctx, res, |s| {
                s.exposure_auto = auto;
                if let Some(a) = actual {
                    s.exposure_us = a;
                }
            });
        }
        FlirCmd::SetGainDb(v) => {
            let res = cam.set_float("Gain", v);
            let actual = cam.get_float("Gain").ok();
            finish_apply(ctx, res, |s| {
                if let Some(a) = actual {
                    s.gain_db = a;
                }
            });
        }
        FlirCmd::SetGainAuto(mode) => {
            let res = cam.set_enum("GainAuto", mode.node_entry());
            let auto = read_auto(cam, "GainAuto");
            let actual = cam.get_float("Gain").ok();
            finish_apply(ctx, res, |s| {
                s.gain_auto = auto;
                if let Some(a) = actual {
                    s.gain_db = a;
                }
            });
        }
        FlirCmd::SetIspEnable(b) => {
            let res = (|| {
                cam.end_acquisition()?;
                let write = cam.set_bool("IspEnable", b);
                let resume = cam.begin_acquisition();
                write?;
                resume
            })();
            let actual = cam.get_bool("IspEnable").ok();
            finish_apply(ctx, res, |s| {
                if let Some(a) = actual {
                    s.isp_enable = a;
                }
            });
        }
        FlirCmd::ApplyFps(new_fps) => resync(cam, ctx, Some(new_fps)),
        FlirCmd::Resync => resync(cam, ctx, None),
    }
}

fn finish_apply(ctx: &FlirThreadCtx, res: Result<(), String>, update: impl FnOnce(&mut FlirSettingsSnapshot)) {
    let mut s = ctx.snapshot.lock().unwrap();
    match res {
        Ok(()) => s.last_error = None,
        Err(e) => s.last_error = Some(e),
    }
    update(&mut s);
    s.revision += 1;
}

fn resync(cam: &SpinCamera, ctx: &FlirThreadCtx, fps_to_write: Option<f64>) {
    let old_fps = ctx.tuning.fps();
    let mut clock_reset = false;
    let mut accepted_fps: Option<f64> = None;
    let result = {
        let clock_reset = &mut clock_reset;
        let accepted_fps = &mut accepted_fps;
        (move || -> Result<(f64, (i64, i64)), String> {
            cam.end_acquisition()?;
            let actual_fps = if let Some(new_fps) = fps_to_write {
                cam.set_float("AcquisitionFrameRate", new_fps)?;
                cam.get_float("AcquisitionFrameRate").unwrap_or(new_fps)
            } else {
                ctx.tuning.fps()
            };
            *accepted_fps = Some(actual_fps);
            cam.set_enum("TriggerMode", "On")?;
            cam.set_enum("TriggerSource", "Software")?;
            cam.begin_acquisition()?;
            *clock_reset = true;
            let seed = handshake(cam, &ctx.trig_seen_rx, &ctx.tap, ctx.trigger_polarity)?;
            Ok((actual_fps, seed))
        })()
    };

    match result {
        Ok((actual_fps, seed)) => {
            let _ = ctx.pipeline_cmd_tx.send(PipelineCmd::Reseed { fps: actual_fps, seed });
            ctx.tuning.set_fps(actual_fps);
            let resume = cam.set_enum("TriggerMode", "Off");
            let mut s = ctx.snapshot.lock().unwrap();
            match resume {
                Ok(()) => {
                    s.fps = actual_fps;
                    if fps_to_write.is_some() {
                        s.fps_apply = FpsApplyState::Idle;
                    }
                    s.last_error = None;
                    if fps_to_write.is_some() {
                        eprintln!(
                            "fps apply ok: fps={actual_fps} seed=({}, {}) offset={}",
                            seed.0,
                            seed.1,
                            seed.1 - seed.0
                        );
                    } else {
                        eprintln!(
                            "resync ok: fps={actual_fps} seed=({}, {}) offset={}",
                            seed.0,
                            seed.1,
                            seed.1 - seed.0
                        );
                    }
                }
                Err(e) => {
                    let msg = format!("re-handshake ok but free-run resume failed: {e}");
                    if fps_to_write.is_some() {
                        s.fps_apply = FpsApplyState::Failed(msg);
                    } else {
                        eprintln!("resync: re-handshake ok but free-run resume failed: {e}");
                        s.last_error = Some(msg);
                    }
                }
            }
            s.revision += 1;
        }
        Err(e) if clock_reset => {
            let _ = ctx.pipeline_cmd_tx.send(PipelineCmd::Desync);
            let fps_now = accepted_fps.unwrap_or(old_fps);
            ctx.tuning.set_fps(fps_now);
            let _ = cam.set_enum("TriggerMode", "Off");
            let mut s = ctx.snapshot.lock().unwrap();
            s.fps = fps_now;
            if fps_to_write.is_some() {
                s.fps_apply = FpsApplyState::Failed(format!(
                    "{e} — paired sync LOST (the FLIR clock was already reset); previews continue \
                     unpaired until a successful Apply re-handshakes. \
                     配对同步已丢失(FLIR 时钟已重置):预览退回未配对模式,重新 Apply 成功后才会恢复。"
                ));
            } else {
                s.last_error = Some(format!(
                    "resync failed: {e} — paired sync LOST (the FLIR clock was already reset); previews \
                     continue unpaired until the next successful re-handshake. \
                     配对同步已丢失(FLIR 时钟已重置):预览退回未配对模式,下一次重新握手成功后才会恢复。"
                ));
            }
            s.revision += 1;
            if fps_to_write.is_some() {
                eprintln!("fps apply failed AFTER clock reset -- pipeline desynced until a successful Apply");
            } else {
                eprintln!("resync failed AFTER clock reset -- pipeline desynced until a successful re-handshake");
            }
        }
        Err(e) => {
            if fps_to_write.is_some() {
                let _ = cam.set_float("AcquisitionFrameRate", old_fps);
            }
            let _ = cam.set_enum("TriggerMode", "Off");
            let _ = cam.begin_acquisition();
            let mut s = ctx.snapshot.lock().unwrap();
            if fps_to_write.is_some() {
                s.fps_apply = FpsApplyState::Failed(e);
            } else {
                s.last_error = Some(e);
            }
            s.revision += 1;
            if fps_to_write.is_some() {
                eprintln!("fps apply failed before clock reset; rolled back to fps={old_fps}");
            } else {
                eprintln!("resync failed before clock reset; fps unchanged at {old_fps}");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::AcquisitionGuard;



    #[test]
    fn armed_drop_runs_cleanup_once() {
        use std::cell::Cell;
        let calls = Cell::new(0);
        {
            let _g = AcquisitionGuard::arm(|| calls.set(calls.get() + 1));
            assert_eq!(calls.get(), 0, "arm 本身不触发清理");
        }
        assert_eq!(calls.get(), 1, "半路夭折必须清理一次已开始的采集");
    }

    #[test]
    fn disarmed_drop_does_not_run_cleanup() {
        use std::cell::Cell;
        let calls = Cell::new(0);
        AcquisitionGuard::arm(|| calls.set(calls.get() + 1)).disarm();
        assert_eq!(calls.get(), 0, "撤防之后 Drop 不应该再跑清理动作");
    }

    #[test]
    fn armed_drop_runs_cleanup_on_panic_unwind() {
        use std::panic::{catch_unwind, AssertUnwindSafe};
        use std::sync::atomic::{AtomicU32, Ordering};

        let calls = AtomicU32::new(0);
        let result = catch_unwind(AssertUnwindSafe(|| {
            let _g = AcquisitionGuard::arm(|| {
                calls.fetch_add(1, Ordering::Relaxed);
            });
            panic!("simulated failure between begin_acquisition and disarm");
        }));
        assert!(result.is_err(), "panic must propagate out of catch_unwind");
        assert_eq!(calls.load(Ordering::Relaxed), 1, "展开经过 armed guard 时必须清理一次");
    }
}
