use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crossbeam_channel::{Receiver, RecvTimeoutError};
use fs_calib::background::BackgroundSession;
use fs_calib::board::{self, BoardConfig, BoardKind, DetectOpts, DetectSpeed};
use fs_calib::engine::{CalibSession, IntrinsicsEngine};
use fs_calib::export::{to_native_json, to_stereo_calibration_v1};
use fs_core::{GrayImage, PixelFormat, RgbFrame, SyncedFrame};
use opencv::core::Mat;
use opencv::imgproc;
use opencv::prelude::*;

const POOL_CAP: usize = 40;

const RECON_TAG_TOLERANCE_US: i64 = 1000;

const RECON_WAIT_TIMEOUT_CAP_MS: u64 = 300;

fn recon_wait_budget_ms(fps: f64) -> u64 {
    if !(fps > 0.0) {
        return RECON_WAIT_TIMEOUT_CAP_MS;
    }
    let frame_interval_ms = 1000.0 / fps;
    ((2.0 * frame_interval_ms).round() as u64).clamp(1, RECON_WAIT_TIMEOUT_CAP_MS)
}


const MIN_BOARD_AREA_FRAC: f32 = 0.02;

const FINAL_DRAIN_TIMEOUT: Duration = Duration::from_secs(30);

pub const GEOMETRY_PAUSED_BUG: &str = "几何处理开启中,标定已暂停——这是一个 bug,请报告";

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ResetKind {
    Stereo,
    Camera(usize),
    All,
}

#[derive(Clone)]
pub struct CamPanel {
    pub pool_len: usize,
    pub cap: usize,
    pub rms: Option<f64>,
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
    pub converged: bool,
    pub coverage: [[bool; 6]; 6],
    pub history: Vec<[f64; 5]>,
}

impl Default for CamPanel {
    fn default() -> Self {
        CamPanel {
            pool_len: 0,
            cap: POOL_CAP,
            rms: None,
            fx: 0.0,
            fy: 0.0,
            cx: 0.0,
            cy: 0.0,
            converged: false,
            coverage: [[false; 6]; 6],
            history: Vec::new(),
        }
    }
}


pub struct CalibUiState {

    pub enabled: bool,
    pub frozen: bool,

    pub geometry_active: bool,
    pub geometry_fault: Option<&'static str>,
    pub board: BoardConfig,

    pub detect_speed: DetectSpeed,
    pub max_detect_hz: f64,
    pub cam0: CamPanel,
    pub cam1: CamPanel,
    pub stereo: Option<(f64, f64, [f64; 3])>,
    pub stereo_r: Option<[[f64; 3]; 3]>,
    pub last_detect0: Option<Vec<(f32, f32)>>,
    pub last_detect0_t: Option<i64>,
    pub last_detect1: Option<Vec<(f32, f32)>>,
    pub last_detect1_t: Option<i64>,

    pub last_detect0_at: Option<Instant>,
    pub last_detect1_at: Option<Instant>,

    pub last_detect0_accepted: bool,
    pub last_detect1_accepted: bool,
    pub calib_in_flight: bool,

    pub calib_published: u64,

    pub session_exists: bool,
    pub reset_request: Option<ResetKind>,
    pub export_request: Option<String>,
    pub export_result: Option<Result<String, String>>,

    pub apply_request: bool,
    pub apply_result: Option<Result<serde_json::Value, String>>,

    pub synced_seen: u64,
    pub recon_pair_mismatches: u64,
    pub det0_count: u64,
    pub det1_count: u64,
    pub both_count: u64,
    pub throttle_skipped: u64,
    pub prefilter_rejects0: u64,
    pub prefilter_rejects1: u64,

    pub det0_attempts: u64,
    pub det1_attempts: u64,
    pub det0_ms_sum: f64,
    pub det1_ms_sum: f64,
}

impl Default for CalibUiState {
    fn default() -> Self {
        CalibUiState {
            enabled: false,
            frozen: false,
            geometry_active: false,
            geometry_fault: None,
            board: BoardConfig {
                kind: BoardKind::Checkerboard,
                inner_cols: 7,
                inner_rows: 6,
                rect_w_mm: 82.0,
                rect_h_mm: 98.0,
            },
            detect_speed: DetectSpeed::default(),
            max_detect_hz: 8.0,
            cam0: CamPanel::default(),
            cam1: CamPanel::default(),
            stereo: None,
            stereo_r: None,
            last_detect0: None,
            last_detect0_t: None,
            last_detect1: None,
            last_detect1_t: None,
            last_detect0_at: None,
            last_detect1_at: None,
            last_detect0_accepted: false,
            last_detect1_accepted: false,
            calib_in_flight: false,
            calib_published: 0,
            session_exists: false,
            reset_request: None,
            export_request: None,
            export_result: None,
            apply_request: false,
            apply_result: None,
            synced_seen: 0,
            recon_pair_mismatches: 0,
            det0_count: 0,
            det1_count: 0,
            both_count: 0,
            throttle_skipped: 0,
            prefilter_rejects0: 0,
            prefilter_rejects1: 0,
            det0_attempts: 0,
            det1_attempts: 0,
            det0_ms_sum: 0.0,
            det1_ms_sum: 0.0,
        }
    }
}


pub fn run(
    rx_synced: Receiver<Arc<SyncedFrame>>,
    rx_recon: Receiver<(GrayImage, i64)>,
    state: Arc<Mutex<CalibUiState>>,
    tuning: crate::settings::SharedTuning,
    recon_dims: (u32, u32),
) {
    let mut session: Option<BackgroundSession> = None;
    let stats_diag = std::env::var("FS_RECON_STATS").is_ok();
    let mut iter_count: u64 = 0;
    let mut prev_det0 = false;
    let mut prev_det1 = false;
    let mut last_detect_attempt: Option<Instant> = None;

    loop {
        let enabled = state.lock().unwrap().enabled;
        if !enabled {
            let _ = rx_synced.recv_timeout(Duration::from_millis(50));
            while rx_recon.try_recv().is_ok() {}
            handle_reset_export(&state, &mut session);
            continue;
        }

        let mut sf = match rx_synced.recv_timeout(Duration::from_millis(50)) {
            Ok(sf) => sf,
            Err(RecvTimeoutError::Timeout) => {
                if state.lock().unwrap().frozen {
                    continue;
                }
                pump_session(&state, &mut session);
                handle_reset_export(&state, &mut session);
                continue;
            }
            Err(RecvTimeoutError::Disconnected) => {
                if let Some(sess) = session.as_mut() {
                    let deadline = Instant::now() + FINAL_DRAIN_TIMEOUT;
                    loop {
                        let applied = sess.pump();
                        if applied > 0 {
                            state.lock().unwrap().calib_published += applied as u64;
                            snapshot(&state, sess);
                            continue;
                        }
                        if !sess.busy() {
                            let tail = sess.pump();
                            if tail == 0 {
                                break;
                            }
                            state.lock().unwrap().calib_published += tail as u64;
                            snapshot(&state, sess);
                            continue;
                        }
                        if Instant::now() > deadline {
                            eprintln!("calib worker: final drain timed out with a calibration still in flight");
                            break;
                        }
                        std::thread::sleep(Duration::from_millis(10));
                    }
                    snapshot(&state, sess);
                }
                break;
            }
        };
        while let Ok(newer) = rx_synced.try_recv() {
            sf = newer;
        }
        let iter_t0 = Instant::now();
        let (frozen, geometry_active) = {
            let mut s = state.lock().unwrap();
            s.synced_seen += 1;
            (s.frozen, s.geometry_active)
        };
        if frozen {
            continue;
        }
        if geometry_active {
            state.lock().unwrap().geometry_fault = Some(GEOMETRY_PAUSED_BUG);
            continue;
        }

        let max_detect_hz = state.lock().unwrap().max_detect_hz;
        if max_detect_hz > 0.0 {
            let min_interval = Duration::from_secs_f64(1.0 / max_detect_hz);
            if let Some(last) = last_detect_attempt {
                if iter_t0.duration_since(last) < min_interval {
                    state.lock().unwrap().throttle_skipped += 1;
                    pump_session(&state, &mut session);
                    handle_reset_export(&state, &mut session);
                    continue;
                }
            }
        }
        last_detect_attempt = Some(iter_t0);

        let wait_budget_ms = recon_wait_budget_ms(tuning.fps());
        let want_t = sf.recon_time_us(tuning.exposure_us());

        let (gray1, wait_diag) = wait_for_matching_recon(&rx_recon, want_t, wait_budget_ms);
        if gray1.is_none() {
            state.lock().unwrap().recon_pair_mismatches += 1;
        }

        let Some(gray0) = flir_to_gray(&sf.frame) else {
            pump_session(&state, &mut session);
            handle_reset_export(&state, &mut session);
            continue;
        };
        let (w0, h0) = (sf.frame.w, sf.frame.h);
        let (w1, h1) = recon_dims;

        if session.is_none() {
            let board = state.lock().unwrap().board.clone();
            session = Some(BackgroundSession::new(board, POOL_CAP, (w0, h0), (w1, h1)));
        }
        let sess = session.as_mut().expect("just constructed above when None");
        let board_cfg = sess.session().cfg.clone();

        let speed = state.lock().unwrap().detect_speed;
        let use_prefilter = std::env::var_os("FS_CALIB_PREFILTER").is_some();
        let opts0 = DetectOpts { speed, skip_prefilter: !use_prefilter || prev_det0 };
        let opts1 = DetectOpts { speed, skip_prefilter: !use_prefilter || prev_det1 };

        let detect_scope_t0 = Instant::now();
        let (det0_out, det0_ms, det1_slot) = if let Some(g1) = gray1.as_ref() {
            std::thread::scope(|scope| {
                let cam0 = scope.spawn(|| {
                    let t = Instant::now();
                    let out = board::detect_opts(&board_cfg, &gray0, w0, h0, opts0);
                    (out, t.elapsed().as_secs_f64() * 1000.0)
                });
                let t1 = Instant::now();
                let out1 = board::detect_opts(&board_cfg, &g1.data, w1, h1, opts1);
                let ms1 = t1.elapsed().as_secs_f64() * 1000.0;
                let (out0, ms0) = cam0.join().expect("cam0 detect thread panicked");
                (out0, ms0, Some((out1, ms1)))
            })
        } else {
            let t = Instant::now();
            let out0 = board::detect_opts(&board_cfg, &gray0, w0, h0, opts0);
            (out0, t.elapsed().as_secs_f64() * 1000.0, None)
        };
        let detect_scope_ms = detect_scope_t0.elapsed().as_secs_f64() * 1000.0;

        prev_det0 = det0_out.detection.is_some();
        let prefiltered0 = det0_out.prefiltered;
        let det0 = det0_out.detection.filter(|d| d.board_area_frac >= MIN_BOARD_AREA_FRAC);

        let (det1, det1_ms, prefiltered1) = match det1_slot {
            Some((out1, ms1)) => {
                prev_det1 = out1.detection.is_some();
                let prefiltered1 = out1.prefiltered;
                let det1 = out1.detection.filter(|d| d.board_area_frac >= MIN_BOARD_AREA_FRAC);
                (det1, Some(ms1), prefiltered1)
            }
            None => (None, None, false),
        };

        {
            let mut s = state.lock().unwrap();
            s.det0_attempts += 1;
            s.det0_ms_sum += det0_ms;
            if prefiltered0 {
                s.prefilter_rejects0 += 1;
            }
            if let Some(ms1) = det1_ms {
                s.det1_attempts += 1;
                s.det1_ms_sum += ms1;
            }
            if prefiltered1 {
                s.prefilter_rejects1 += 1;
            }
            if det0.is_some() {
                s.det0_count += 1;
            }
            if det1.is_some() {
                s.det1_count += 1;
            }
            if det0.is_some() && det1.is_some() {
                s.both_count += 1;
            }
            if let Some(d0) = &det0 {
                s.last_detect0 = Some(d0.corners.clone());
                s.last_detect0_t = Some(want_t);
                s.last_detect0_at = Some(Instant::now());
            }
            if let Some(d1) = &det1 {
                s.last_detect1 = Some(d1.corners.clone());
                s.last_detect1_t = Some(want_t);
                s.last_detect1_at = Some(Instant::now());
            }
        }

        let had_det0 = det0.is_some();
        let had_det1 = det1.is_some();
        let pool0_before = sess.session().cam0.pool_len();
        let pool1_before = sess.session().cam1.pool_len();

        let observe_t0 = Instant::now();
        sess.observe(det0, det1);
        let observe_ms = observe_t0.elapsed().as_secs_f64() * 1000.0;
        let pool0_after = sess.session().cam0.pool_len();
        let pool1_after = sess.session().cam1.pool_len();

        if had_det0 {
            state.lock().unwrap().last_detect0_accepted = pool0_after != pool0_before;
        }
        if had_det1 {
            state.lock().unwrap().last_detect1_accepted = pool1_after != pool1_before;
        }

        let applied = sess.pump();
        if applied > 0 {
            state.lock().unwrap().calib_published += applied as u64;
        }

        snapshot(&state, sess);
        handle_reset_export(&state, &mut session);

        iter_count += 1;
        if stats_diag && iter_count % 10 == 0 {
            let iter_ms = iter_t0.elapsed().as_secs_f64() * 1000.0;
            let serial_sum_ms = det0_ms + det1_ms.unwrap_or(0.0);
            eprintln!(
                "[calib-stats] #{iter_count} iter={iter_ms:.1}ms want_t={want_t} matched={} wait_seen={} wait_closest_delta_us={:?} \
                 det0_ms={det0_ms:.1} det1_ms={:?} detect_scope_ms={detect_scope_ms:.1} serial_sum_ms={serial_sum_ms:.1} \
                 speedup={:.2}x observe_ms={observe_ms:.1}",
                gray1.is_some(),
                wait_diag.seen,
                wait_diag.closest_abs_delta_us,
                det1_ms,
                if detect_scope_ms > 0.0 { serial_sum_ms / detect_scope_ms } else { 1.0 }
            );
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct WaitDiag {
    seen: u32,
    closest_abs_delta_us: Option<i64>,
}


fn wait_for_matching_recon(rx_recon: &Receiver<(GrayImage, i64)>, want_t: i64, timeout_ms: u64) -> (Option<GrayImage>, WaitDiag) {
    let deadline = Instant::now() + Duration::from_millis(timeout_ms);
    let mut diag = WaitDiag::default();
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return (None, diag);
        }
        match rx_recon.recv_timeout(remaining) {
            Ok((g, t)) if (t - want_t).abs() <= RECON_TAG_TOLERANCE_US => return (Some(g), diag),
            Ok((_, t)) => {
                diag.seen += 1;
                let delta = (t - want_t).abs();
                diag.closest_abs_delta_us = Some(diag.closest_abs_delta_us.map_or(delta, |d| d.min(delta)));
            }
            Err(_) => return (None, diag),
        }
    }
}

fn pump_session(state: &Arc<Mutex<CalibUiState>>, session: &mut Option<BackgroundSession>) {
    let Some(sess) = session.as_mut() else { return };
    let applied = sess.pump();
    if applied > 0 {
        state.lock().unwrap().calib_published += applied as u64;
        snapshot(state, sess);
    } else {
        state.lock().unwrap().calib_in_flight = sess.busy();
    }
}


fn snapshot(state: &Arc<Mutex<CalibUiState>>, sess: &BackgroundSession) {
    let inner = sess.session();
    let cam0 = cam_panel_from(&inner.cam0);
    let cam1 = cam_panel_from(&inner.cam1);
    let stereo = inner.stereo.state.as_ref().map(|st| (st.rms, rotation_angle_deg(&st.r), st.t));
    let stereo_r = inner.stereo.state.as_ref().map(|st| st.r);
    let busy = sess.busy();
    let mut s = state.lock().unwrap();
    s.cam0 = cam0;
    s.cam1 = cam1;
    s.stereo = stereo;
    s.stereo_r = stereo_r;
    s.calib_in_flight = busy;
    s.session_exists = true;
}

fn cam_panel_from(engine: &IntrinsicsEngine) -> CamPanel {
    let (rms, fx, fy, cx, cy) = match &engine.state {
        Some(c) => (Some(c.rms), c.k[0][0], c.k[1][1], c.k[0][2], c.k[1][2]),
        None => (None, 0.0, 0.0, 0.0, 0.0),
    };
    CamPanel {
        pool_len: engine.pool_len(),
        cap: POOL_CAP,
        rms,
        fx,
        fy,
        cx,
        cy,
        converged: engine.converged(),
        coverage: engine.coverage_mask(),
        history: engine.history().cloned().collect(),
    }
}


fn rotation_angle_deg(r: &[[f64; 3]; 3]) -> f64 {
    let trace = r[0][0] + r[1][1] + r[2][2];
    ((trace - 1.0) / 2.0).clamp(-1.0, 1.0).acos().to_degrees()
}


fn handle_reset_export(state: &Arc<Mutex<CalibUiState>>, session: &mut Option<BackgroundSession>) {
    let (reset, export, apply) = {
        let mut s = state.lock().unwrap();
        (s.reset_request.take(), s.export_request.take(), std::mem::take(&mut s.apply_request))
    };
    if let Some(kind) = reset {
        match kind {
            ResetKind::Stereo => {
                if let Some(sess) = session.as_mut() {
                    sess.reset_stereo();
                    snapshot(state, sess);
                }
            }
            ResetKind::Camera(idx) => {
                if let Some(sess) = session.as_mut() {
                    sess.reset_camera(idx);
                    snapshot(state, sess);
                }
            }
            ResetKind::All => {
                *session = None;
                let mut s = state.lock().unwrap();
                s.cam0 = CamPanel::default();
                s.cam1 = CamPanel::default();
                s.stereo = None;
                s.stereo_r = None;
                s.calib_in_flight = false;
                s.session_exists = false;
                s.enabled = false;
            }
        }
    }
    if let Some(dir) = export {
        let result = do_export(session.as_ref().map(|s| s.session()), &dir);
        state.lock().unwrap().export_result = Some(result);
    }
    if apply {
        let result = match session.as_ref().map(|s| s.session()) {
            Some(sess) => to_stereo_calibration_v1(sess),
            None => Err("还没有标定会话(未观测到任何帧)".to_string()),
        };
        state.lock().unwrap().apply_result = Some(result);
    }
}

fn do_export(session: Option<&CalibSession>, dir: &str) -> Result<String, String> {
    let session = session.ok_or_else(|| "no calibration session yet (nothing observed)".to_string())?;
    let path = std::path::Path::new(dir);
    std::fs::create_dir_all(path).map_err(|e| format!("failed to create '{dir}': {e}"))?;

    let native = to_native_json(session);
    let native_path = path.join("calib.json");
    std::fs::write(&native_path, serde_json::to_string_pretty(&native).expect("json serialize"))
        .map_err(|e| format!("failed to write {}: {e}", native_path.display()))?;

    let v1 = to_stereo_calibration_v1(session)?;
    let v1_path = path.join("stereo_calibration.v1.json");
    std::fs::write(&v1_path, serde_json::to_string_pretty(&v1).expect("json serialize"))
        .map_err(|e| format!("failed to write {}: {e}", v1_path.display()))?;

    Ok(format!("wrote {} and {}", native_path.display(), v1_path.display()))
}

fn flir_to_gray(f: &RgbFrame) -> Option<Vec<u8>> {
    static NON_RGB8_LOGGED: AtomicBool = AtomicBool::new(false);
    if f.format != PixelFormat::Rgb8 {
        if !NON_RGB8_LOGGED.swap(true, Ordering::Relaxed) {
            eprintln!("calib: 非 Rgb8 帧({:?})到达标定 tap —— 上游变换级 bug;丢帧", f.format);
        }
        return None;
    }
    let need = (f.w * f.h * 3) as usize;
    if f.data.len() < need {
        return None;
    }
    let flat = Mat::new_rows_cols_with_data(f.h as i32, (f.w * 3) as i32, &f.data[..need]).ok()?;
    let rgb = flat.reshape(3, f.h as i32).ok()?;
    let mut dst = Mat::default();
    imgproc::cvt_color_def(&rgb, &mut dst, imgproc::COLOR_RGB2GRAY).ok()?;
    dst.data_bytes().ok().map(|b| b.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frozen_worker_consumes_input_without_touching_the_pool() {
        use crossbeam_channel::bounded;
        use fs_core::{PixelFormat, RgbFrame, SyncSource, SyncedFrame};

        let (tx_synced, rx_synced) = bounded::<Arc<SyncedFrame>>(4);
        let (_tx_recon, rx_recon) = bounded::<(GrayImage, i64)>(4);
        let state = Arc::new(Mutex::new(CalibUiState::default()));
        {
            let mut s = state.lock().unwrap();
            s.enabled = true;
            s.frozen = true;
        }
        let tuning = crate::settings::SharedTuning::new(30.0, 5000);
        let st = state.clone();
        let h = std::thread::spawn(move || run(rx_synced, rx_recon, st, tuning, (1280, 720)));

        tx_synced
            .send(Arc::new(SyncedFrame {
                frame: RgbFrame {
                    seq: 0,
                    t_cam_us: 0,
                    w: 4,
                    h: 4,
                    data: vec![0u8; 48],
                    format: PixelFormat::Rgb8,
                },
                t_evk_us: 3000,
                source: SyncSource::Matched,
            }))
            .unwrap();
        std::thread::sleep(std::time::Duration::from_millis(300));

        let s = state.lock().unwrap();
        assert_eq!(s.cam0.pool_len, 0, "冻结期间不得有任何帧进入 pool");
        assert!(!s.session_exists, "冻结期间不得建立标定会话");
        assert!(s.last_detect0.is_none(), "冻结期间不得产生检测 overlay");
        assert!(s.synced_seen >= 1, "worker 应当已经收到并计数了这一帧");
        drop(s);
        drop(tx_synced);
        h.join().unwrap();
    }

    #[test]
    fn frozen_worker_ignores_requests_on_the_idle_timeout_path() {
        use crossbeam_channel::bounded;

        let (tx_synced, rx_synced) = bounded::<Arc<SyncedFrame>>(4);
        let (_tx_recon, rx_recon) = bounded::<(GrayImage, i64)>(4);
        let state = Arc::new(Mutex::new(CalibUiState::default()));
        {
            let mut s = state.lock().unwrap();
            s.enabled = true;
            s.frozen = true;
            s.export_request = Some("unused/export/dir".to_string());
        }
        let tuning = crate::settings::SharedTuning::new(30.0, 5000);
        let st = state.clone();
        let h = std::thread::spawn(move || run(rx_synced, rx_recon, st, tuning, (1280, 720)));

        std::thread::sleep(std::time::Duration::from_millis(220));

        let s = state.lock().unwrap();
        assert!(s.export_request.is_some(), "冻结期间连空闲 timeout 分支也不得消费 export_request");
        assert!(s.export_result.is_none(), "冻结期间不得产生 export 结果");
        drop(s);
        drop(tx_synced);
        h.join().unwrap();
    }


    const TEST_WAIT_CEILING: Duration = Duration::from_secs(20);


    fn wait_until(state: &Arc<Mutex<CalibUiState>>, timeout: Duration, pred: impl Fn(&CalibUiState) -> bool) -> bool {
        let deadline = Instant::now() + timeout;
        loop {
            if pred(&state.lock().unwrap()) {
                return true;
            }
            if Instant::now() > deadline {
                return false;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
    }


    struct WorkerGuard {
        state: Arc<Mutex<CalibUiState>>,
        tx_synced: Option<crossbeam_channel::Sender<Arc<SyncedFrame>>>,
        handle: Option<std::thread::JoinHandle<()>>,
    }

    impl WorkerGuard {
        fn send(&self, frame: Arc<SyncedFrame>) {
            self.tx_synced.as_ref().expect("guard already dropped").send(frame).unwrap();
        }
    }

    impl Drop for WorkerGuard {
        fn drop(&mut self) {
            self.state.lock().unwrap().enabled = true;
            self.tx_synced.take();
            if let Some(h) = self.handle.take() {
                let _ = h.join();
            }
        }
    }


    #[test]
    fn calib_worker_refuses_frames_while_geometry_active() {
        use crossbeam_channel::bounded;
        use fs_core::{PixelFormat, RgbFrame, SyncSource, SyncedFrame};

        let (tx_synced, rx_synced) = bounded::<Arc<SyncedFrame>>(4);
        let (_tx_recon, rx_recon) = bounded::<(GrayImage, i64)>(4);
        let state = Arc::new(Mutex::new(CalibUiState::default()));
        {
            let mut s = state.lock().unwrap();
            s.enabled = true;
            s.geometry_active = true;
        }
        let tuning = crate::settings::SharedTuning::new(30.0, 5000);
        let st = state.clone();
        let h = std::thread::spawn(move || run(rx_synced, rx_recon, st, tuning, (1280, 720)));
        let guard = WorkerGuard { state: state.clone(), tx_synced: Some(tx_synced), handle: Some(h) };

        guard.send(Arc::new(SyncedFrame {
            frame: RgbFrame { seq: 0, t_cam_us: 0, w: 4, h: 4, data: vec![0u8; 48], format: PixelFormat::Rgb8 },
            t_evk_us: 3000,
            source: SyncSource::Matched,
        }));

        assert!(
            wait_until(&state, TEST_WAIT_CEILING, |s| s.geometry_fault.is_some()),
            "几何 On 时的观测必须触发面板红字"
        );
        let s = state.lock().unwrap();
        assert_eq!(s.geometry_fault, Some(GEOMETRY_PAUSED_BUG), "红字必须是那句可报告的 bug 文案");
        assert!(!s.session_exists, "几何 On 时不得建立标定会话");
        assert_eq!(s.cam0.pool_len, 0, "几何 On 时不得有任何帧进入 pool");
        assert!(s.last_detect0.is_none(), "几何 On 时不得产生检测 overlay");
        assert!(s.synced_seen >= 1, "worker 应当已经收到并计数了这一帧 —— 否则上面的断言是假阳性");
    }

    #[test]
    fn paused_reset_all_is_serviced() {
        use crossbeam_channel::bounded;
        use fs_core::{PixelFormat, RgbFrame, SyncSource, SyncedFrame};

        let (tx_synced, rx_synced) = bounded::<Arc<SyncedFrame>>(4);
        let (_tx_recon, rx_recon) = bounded::<(GrayImage, i64)>(4);
        let state = Arc::new(Mutex::new(CalibUiState::default()));
        state.lock().unwrap().enabled = true;
        let tuning = crate::settings::SharedTuning::new(30.0, 5000);
        let st = state.clone();
        let h = std::thread::spawn(move || run(rx_synced, rx_recon, st, tuning, (1280, 720)));
        let guard = WorkerGuard { state: state.clone(), tx_synced: Some(tx_synced), handle: Some(h) };

        guard.send(Arc::new(SyncedFrame {
            frame: RgbFrame { seq: 0, t_cam_us: 0, w: 4, h: 4, data: vec![0u8; 48], format: PixelFormat::Rgb8 },
            t_evk_us: 3000,
            source: SyncSource::Matched,
        }));
        assert!(wait_until(&state, TEST_WAIT_CEILING, |s| s.session_exists), "第一次观测应当建立 session");

        {
            let mut s = state.lock().unwrap();
            s.enabled = false;
            s.export_request = Some("\0settle-probe".to_string());
        }
        assert!(
            wait_until(&state, TEST_WAIT_CEILING, |s| s.export_result.is_some()),
            "settle 探针应当被消费，证明 worker 已经带着 enabled=false 回到循环顶部"
        );

        state.lock().unwrap().reset_request = Some(ResetKind::All);
        assert!(
            wait_until(&state, TEST_WAIT_CEILING, |s| !s.session_exists),
            "暂停期间的 Reset all 必须被消费，而不是被 `!enabled` 分支吞掉"
        );
        assert!(state.lock().unwrap().reset_request.is_none(), "请求消费后应当清空");
    }

    #[test]
    fn running_reset_all_pauses_and_the_board_stays_editable_across_more_frames() {
        use crossbeam_channel::bounded;
        use fs_core::{PixelFormat, RgbFrame, SyncSource, SyncedFrame};

        let (tx_synced, rx_synced) = bounded::<Arc<SyncedFrame>>(4);
        let (_tx_recon, rx_recon) = bounded::<(GrayImage, i64)>(4);
        let state = Arc::new(Mutex::new(CalibUiState::default()));
        state.lock().unwrap().enabled = true;
        let tuning = crate::settings::SharedTuning::new(30.0, 5000);
        let st = state.clone();
        let h = std::thread::spawn(move || run(rx_synced, rx_recon, st, tuning, (1280, 720)));
        let guard = WorkerGuard { state: state.clone(), tx_synced: Some(tx_synced), handle: Some(h) };

        let frame = || {
            Arc::new(SyncedFrame {
                frame: RgbFrame { seq: 0, t_cam_us: 0, w: 4, h: 4, data: vec![0u8; 48], format: PixelFormat::Rgb8 },
                t_evk_us: 3000,
                source: SyncSource::Matched,
            })
        };

        guard.send(frame());
        assert!(wait_until(&state, TEST_WAIT_CEILING, |s| s.session_exists), "第一次观测应当建立 session");

        state.lock().unwrap().reset_request = Some(ResetKind::All);
        assert!(
            wait_until(&state, TEST_WAIT_CEILING, |s| !s.session_exists),
            "运行中的 Reset all 也必须被消费"
        );
        assert!(!state.lock().unwrap().enabled, "Reset all 应当把标定也一并暂停");

        assert!(
            !wait_until(&state, Duration::from_millis(300), |s| s.session_exists),
            "暂停后 session 不应在后续帧上被重建 -- 否则面板会在人还没反应过来时就重新灰掉"
        );
    }

    #[test]
    fn paused_export_is_serviced() {
        use crossbeam_channel::bounded;

        let (tx_synced, rx_synced) = bounded::<Arc<SyncedFrame>>(4);
        let (_tx_recon, rx_recon) = bounded::<(GrayImage, i64)>(4);
        let state = Arc::new(Mutex::new(CalibUiState::default()));
        {
            let mut s = state.lock().unwrap();
            s.enabled = false;
            s.export_request = Some("unused/export/dir".to_string());
        }
        let tuning = crate::settings::SharedTuning::new(30.0, 5000);
        let st = state.clone();
        let h = std::thread::spawn(move || run(rx_synced, rx_recon, st, tuning, (1280, 720)));
        let _guard = WorkerGuard { state: state.clone(), tx_synced: Some(tx_synced), handle: Some(h) };

        assert!(
            wait_until(&state, TEST_WAIT_CEILING, |s| s.export_result.is_some()),
            "暂停期间的 Export 请求必须被消费并产生结果"
        );
        assert!(
            state.lock().unwrap().export_request.is_none(),
            "结果出现时请求必然已被取走(取走发生在更早的一次加锁里)"
        );
    }


    #[test]
    fn apply_request_is_serviced_and_reports_uncalibrated() {
        use crossbeam_channel::bounded;
        use fs_core::{PixelFormat, RgbFrame, SyncSource, SyncedFrame};

        let (tx_synced, rx_synced) = bounded::<Arc<SyncedFrame>>(4);
        let (_tx_recon, rx_recon) = bounded::<(GrayImage, i64)>(4);
        let state = Arc::new(Mutex::new(CalibUiState::default()));
        state.lock().unwrap().enabled = true;
        let tuning = crate::settings::SharedTuning::new(30.0, 5000);
        let st = state.clone();
        let h = std::thread::spawn(move || run(rx_synced, rx_recon, st, tuning, (1280, 720)));
        let guard = WorkerGuard { state: state.clone(), tx_synced: Some(tx_synced), handle: Some(h) };

        guard.send(Arc::new(SyncedFrame {
            frame: RgbFrame { seq: 0, t_cam_us: 0, w: 4, h: 4, data: vec![0u8; 48], format: PixelFormat::Rgb8 },
            t_evk_us: 3000,
            source: SyncSource::Matched,
        }));
        assert!(wait_until(&state, TEST_WAIT_CEILING, |s| s.session_exists), "第一次观测应当建立 session");

        state.lock().unwrap().apply_request = true;
        assert!(
            wait_until(&state, TEST_WAIT_CEILING, |s| s.apply_result.is_some()),
            "apply 请求必须像 export 一样在共用服务点被消费"
        );
        let r = state.lock().unwrap().apply_result.take().unwrap();
        assert!(r.is_err(), "未解出的会话必须返回 Err 而不是编造文档");
        assert!(!state.lock().unwrap().apply_request, "请求消费后应当清空");
    }


    #[test]
    fn apply_request_is_not_serviced_while_frozen() {
        use crossbeam_channel::bounded;

        let (tx_synced, rx_synced) = bounded::<Arc<SyncedFrame>>(4);
        let (_tx_recon, rx_recon) = bounded::<(GrayImage, i64)>(4);
        let state = Arc::new(Mutex::new(CalibUiState::default()));
        {
            let mut s = state.lock().unwrap();
            s.enabled = true;
            s.frozen = true;
            s.apply_request = true;
        }
        let tuning = crate::settings::SharedTuning::new(30.0, 5000);
        let st = state.clone();
        let h = std::thread::spawn(move || run(rx_synced, rx_recon, st, tuning, (1280, 720)));

        std::thread::sleep(std::time::Duration::from_millis(220));

        let s = state.lock().unwrap();
        assert!(s.apply_request, "冻结期间连空闲 timeout 分支也不得消费 apply_request");
        assert!(s.apply_result.is_none(), "冻结期间不得产生 apply 结果");
        drop(s);
        drop(tx_synced);
        h.join().unwrap();
    }
}
