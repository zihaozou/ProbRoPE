use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crossbeam_channel::bounded;
use fs_core::health::SyncState;
use fs_core::sync::{SyncConfig, SyncEngine};

use crate::session::{poll_connect_result, AbortGuard, ConnectRx, ConnectState, Session};
use crate::settings::{EvkSettingsSnapshot, FlirSettingsSnapshot, SharedTuning};
use crate::sources::live::FlirCmd;
use crate::stream_rates::StreamRates;
use crate::sync_policy::Action;
use crate::ui_gate::{PanelGate, Tab};
use crate::wiring::spawn_pipeline;
use crate::{calib_worker, decode_stage, pipeline, record_worker, sources, transform, ui_calib, ui_recon, ui_record, ui_settings};


pub const RAW_CHANNEL_CAP: usize = 512;


pub const REPLAY_MAX_PENDING: usize = 240;

fn connect_live(
    attempt: u32,
    fps: f64,
    exposure_us: i64,
    gain_db: f64,
    trigger_polarity: i8,
) -> Result<Session, String> {
    let shutdown = Arc::new(AtomicBool::new(false));
    let abort = AbortGuard::arm(shutdown.clone());
    let (evk_tx, evk_rx) = bounded(256);
    let (flir_tx, flir_rx) = bounded(8);
    let tuning = SharedTuning::new(fps, exposure_us);
    let rates = Arc::new(StreamRates::default());
    let event_xform = transform::EventXform::default();
    let event_tap = transform::EventRecordTap::default();
    let flir_snapshot = Arc::new(Mutex::new(FlirSettingsSnapshot::default()));
    let evk_snapshot = Arc::new(Mutex::new(EvkSettingsSnapshot::default()));
    let (pipe_cmd_tx, pipe_cmd_rx) = bounded::<pipeline::PipelineCmd>(4);

    let (raw_wiring, gpu_raw) = match fs_recon::GpuEventDecoder::new() {
        Ok(decoder) => {
            println!("connect #{attempt}: GPU解码 (raw EVT3 -> GpuEventDecoder; SDK CPU decode bypassed)");
            let (raw_tx, raw_rx) = bounded::<fs_core::bus::EvkRawMsg>(RAW_CHANNEL_CAP);
            let raw_rx_for_drop = raw_rx.clone();
            (
                Some((raw_tx, raw_rx_for_drop)),
                Some(decode_stage::GpuRawIntake { rx_raw: raw_rx, trig_tx: evk_tx.clone(), decoder }),
            )
        }
        Err(e) => {
            eprintln!("connect #{attempt}: CPU解码 (GPU decoder unavailable: {e}); using SDK decoded-events path");
            (None, None)
        }
    };

    let (handles, (seed_flir, seed_evk), recon_dims) = sources::live::start_live(
        sources::live::LiveConfig { fps, exposure_us, gain_db, trigger_polarity, shutdown: shutdown.clone() },
        evk_tx,
        flir_tx,
        sources::live::LiveSettingsHooks {
            flir_snapshot: flir_snapshot.clone(),
            evk_snapshot: evk_snapshot.clone(),
            pipeline_cmd_tx: pipe_cmd_tx,
            tuning: tuning.clone(),
        },
        raw_wiring,
        rates.clone(),
        event_xform.clone(),
        event_tap.clone(),
    )?;
    let mut engine =
        SyncEngine::new(SyncConfig { fps: tuning.fps(), gate_frac: 0.2, trigger_polarity, max_pending: 120 });
    engine.seed(seed_flir, seed_evk);
    let last_error: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
    let settings_ui = ui_settings::SettingsHandles {
        flir: Some(ui_settings::FlirSettingsUi { cmd_tx: handles.flir_cmd.clone(), snapshot: flir_snapshot }),
        evk: Some(ui_settings::EvkSettingsUi { cmd_tx: handles.evk_cmd.clone(), snapshot: evk_snapshot }),
    };
    let pipeline = spawn_pipeline(
        engine,
        flir_rx,
        evk_rx,
        pipe_cmd_rx,
        tuning.clone(),
        recon_dims,
        last_error,
        gpu_raw,
        event_xform,
        event_tap,
        shutdown.clone(),
        rates,
    );

    abort.disarm();
    Ok(Session::new(pipeline, Some(handles), tuning, settings_ui, recon_dims, shutdown))
}

pub fn run_live(fps: f64, exposure_us: i64, gain_db: f64, trigger_polarity: i8) -> anyhow::Result<()> {
    let app = StudioApp::new_live(fps, exposure_us, gain_db, trigger_polarity);
    eframe::run_native(
        "fusion-studio",
        eframe::NativeOptions::default(),
        Box::new(|cc| {
            install_cjk_font(&cc.egui_ctx);
            Ok(Box::new(app))
        }),
    )
    .map_err(|e| anyhow::anyhow!("{e}"))
}

fn connect_replay(raw: String, avi: String, fps: f64) -> Session {
    let shutdown = Arc::new(AtomicBool::new(false));
    let (evk_tx, evk_rx) = bounded(256);
    let (flir_tx, flir_rx) = bounded(8);
    let last_error: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
    let tuning = SharedTuning::new(fps, 0);
    let evk_snapshot = Arc::new(Mutex::new(EvkSettingsSnapshot::default()));
    let rates = Arc::new(StreamRates::default());
    let event_xform = transform::EventXform::default();
    let event_tap = transform::EventRecordTap::default();
    let (_evk_handle, evk_cmd, recon_dims) = sources::replay::spawn_replay_evk(
        raw,
        true,
        evk_tx,
        evk_snapshot.clone(),
        rates.clone(),
        event_xform.clone(),
        event_tap.clone(),
        shutdown.clone(),
    );
    sources::replay::spawn_replay_flir(avi, fps, true, flir_tx, last_error.clone());
    let engine =
        SyncEngine::new(SyncConfig { fps, gate_frac: 0.2, trigger_polarity: 1, max_pending: REPLAY_MAX_PENDING });
    let (_pipe_cmd_tx, pipe_cmd_rx) = bounded::<pipeline::PipelineCmd>(1);
    let settings_ui = ui_settings::SettingsHandles {
        flir: None,
        evk: Some(ui_settings::EvkSettingsUi { cmd_tx: evk_cmd, snapshot: evk_snapshot }),
    };
    let pipeline = spawn_pipeline(
        engine,
        flir_rx,
        evk_rx,
        pipe_cmd_rx,
        tuning.clone(),
        recon_dims,
        last_error,
        None,
        event_xform,
        event_tap,
        shutdown.clone(),
        rates,
    );
    Session::new(pipeline, None, tuning, settings_ui, recon_dims, shutdown)
}

pub fn run_replay(raw: String, avi: String, fps: f64) -> anyhow::Result<()> {
    let app = StudioApp {
        session: Some(connect_replay(raw, avi, fps)),
        connect: ConnectState::Connected,
        connect_rx: None,
        live_params: None,
        connect_attempts: 0,
        disconnected_at: None,
        flir_tex: None,
        recon_tex: None,
        export_path: "calib_out".to_string(),
        active_calib: None,
        calib_load_path: "calib_out/stereo_calibration.v1.json".to_string(),
        active_calib_msg: None,
        geom: None,
        geom_target: (0, 0),
        geom_msg: None,
        show_coverage: false,
        tab: Tab::Camera,
        record_panel: ui_record::RecordPanelState::default(),
        policy: crate::sync_policy::SyncPolicy::new(),
    };
    eframe::run_native(
        "fusion-studio",
        eframe::NativeOptions::default(),
        Box::new(|cc| {
            install_cjk_font(&cc.egui_ctx);
            Ok(Box::new(app))
        }),
    )
    .map_err(|e| anyhow::anyhow!("{e}"))
}


const DEVICE_RELEASE_GRACE: Duration = Duration::from_millis(
    sources::live::GRAB_TIMEOUT_MS + sources::live::HANDSHAKE_TRIGGER_WAIT.as_millis() as u64 + 1500,
);


#[derive(Clone, Copy)]
struct LiveParams {
    fps: f64,
    exposure_us: i64,
    gain_db: f64,
    trigger_polarity: i8,
}


struct GeomActive {
    mode: fs_calib::GeomMode,
    post_doc: serde_json::Value,
    out_size: (u32, u32),
}

fn geom_mode_label(mode: fs_calib::GeomMode) -> String {
    match mode {
        fs_calib::GeomMode::Off => "原始几何".to_string(),
        fs_calib::GeomMode::Undistort => "去畸变 Undistort".to_string(),
        fs_calib::GeomMode::SharedK { target } => format!("共享内参 Shared-K {}×{}", target.0, target.1),
    }
}


fn build_recon_at(choice: crate::recon_thread::ReconChoice, dims: (u32, u32)) -> Result<Box<dyn fs_recon::Reconstructor>, String> {
    match choice {
        crate::recon_thread::ReconChoice::CudaManifold => crate::recon_thread::build_cuda_manifold(dims.0, dims.1)
            .map(|m| Box::new(m) as Box<dyn fs_recon::Reconstructor>)
            .map_err(|e| format!("按 {}×{} 重建 cuda-manifold 失败:{e}", dims.0, dims.1)),
        crate::recon_thread::ReconChoice::Accumulator => {
            Ok(Box::new(crate::recon_thread::build_accumulator(dims.0, dims.1)))
        }
    }
}

struct StudioApp {

    session: Option<Session>,
    connect: ConnectState,

    connect_rx: ConnectRx<Session>,

    live_params: Option<LiveParams>,

    connect_attempts: u32,

    disconnected_at: Option<Instant>,
    flir_tex: Option<egui::TextureHandle>,
    recon_tex: Option<egui::TextureHandle>,

    export_path: String,
    active_calib: Option<fs_calib::ActiveCalibration>,

    calib_load_path: String,

    active_calib_msg: Option<Result<String, String>>,
    geom: Option<GeomActive>,
    geom_target: (u32, u32),

    geom_msg: Option<Result<String, String>>,
    show_coverage: bool,

    tab: Tab,

    record_panel: ui_record::RecordPanelState,
    policy: crate::sync_policy::SyncPolicy,
}

impl StudioApp {
    fn new_live(fps: f64, exposure_us: i64, gain_db: f64, trigger_polarity: i8) -> Self {
        let mut app = StudioApp {
            session: None,
            connect: ConnectState::Disconnected,
            connect_rx: None,
            live_params: Some(LiveParams { fps, exposure_us, gain_db, trigger_polarity }),
            connect_attempts: 0,
            disconnected_at: None,
            flir_tex: None,
            recon_tex: None,
            export_path: "calib_out".to_string(),
            active_calib: None,
            calib_load_path: "calib_out/stereo_calibration.v1.json".to_string(),
            active_calib_msg: None,
            geom: None,
            geom_target: (0, 0),
            geom_msg: None,
            show_coverage: false,
            tab: Tab::Camera,
            record_panel: ui_record::RecordPanelState::default(),
            policy: crate::sync_policy::SyncPolicy::new(),
        };
        app.start_connect();
        app
    }


    fn releasing(&self) -> bool {
        self.disconnected_at.is_some_and(|t| t.elapsed() < DEVICE_RELEASE_GRACE)
    }

    fn start_connect(&mut self) {
        let Some(p) = self.live_params else { return };
        if !self.connect.can_start() || self.releasing() {
            return;
        }
        self.connect_attempts += 1;
        let attempt = self.connect_attempts;
        self.disconnected_at = None;
        let (tx, rx) = bounded(1);
        std::thread::spawn(move || {
            let r = std::panic::catch_unwind(|| {
                connect_live(attempt, p.fps, p.exposure_us, p.gain_db, p.trigger_polarity)
            })
            .unwrap_or_else(|e| Err(format!("连接线程 panic: {}", panic_message(e))));
            let _ = tx.send(r);
        });
        self.connect_rx = Some(rx);
        self.connect = ConnectState::Connecting;
    }

    fn poll_connect(&mut self) {
        if let Some(s) = poll_connect_result(&mut self.connect_rx, &mut self.connect) {
            self.session = Some(s);
        }
    }

    fn disconnect(&mut self) {
        self.session = None;
        self.connect = ConnectState::Disconnected;
        self.connect_rx = None;
        self.disconnected_at = Some(Instant::now());
        self.flir_tex = None;
        self.recon_tex = None;
        if self.geom.take().is_some() {
            self.geom_msg = Some(Ok("已断开 —— 几何处理已回到关闭(重连后需手动重新开启)".to_string()));
        }
    }


    fn camera_tab(&mut self, ui: &mut egui::Ui) {
        if self.live_params.is_none() {
            ui.label("回放模式,无设备");
        } else {
            if let Some(e) = self.connect.error() {
                ui.colored_label(egui::Color32::RED, format!("连接失败: {e}"));
            }
            let connected = self.connect == ConnectState::Connected;
            let releasing = self.releasing();
            let label = if releasing { "断开中…" } else { self.connect.button_label() };
            let clickable = !releasing && (self.connect.can_start() || connected);
            if ui.add_enabled(clickable, egui::Button::new(label)).clicked() {
                if connected {
                    self.disconnect();
                } else {
                    self.start_connect();
                }
            }
        }
        ui.separator();

        let Some(s) = &mut self.session else {
            ui.weak("未连接相机");
            return;
        };

        ui.heading("设备 Device");
        let recording = s.pipeline.record_state.lock().unwrap().recording;
        ui_settings::settings_panel(ui, &mut s.settings_panel, recording);

        ui.separator();
        ui.heading("同步 Sync");
        {
            let h = *s.health.lock().unwrap();
            let (color, label) = if h.desynced {
                (egui::Color32::GRAY, "已失效")
            } else {
                match h.state {
                    SyncState::Healthy => (egui::Color32::GREEN, "健康"),
                    SyncState::Degraded => (egui::Color32::YELLOW, "劣化"),
                    SyncState::Lost => (egui::Color32::RED, "已丢失"),
                }
            };
            let rate = match h.match_rate_window {
                Some(r) => format!("{:.0}%", r * 100.0),
                None => "—".to_string(),
            };
            ui.horizontal(|ui| {
                let (rect, _) = ui.allocate_exact_size(egui::vec2(10.0, 10.0), egui::Sense::hover());
                ui.painter().circle_filled(rect.center(), 5.0, color);
                ui.label(format!("{label} · 滑窗匹配率 {rate}"));
                if ui.button("重新同步").clicked() {
                    let now_ms = (ui.input(|i| i.time) * 1000.0) as u64;
                    self.policy.manual_resync(now_ms);
                    if let Some(live) = s.live.as_ref() {
                        let _ = live.flir_cmd.try_send(FlirCmd::Resync);
                    }
                }
            });
        }
        let st = *s.pipeline.stats.lock().unwrap();
        let total = st.matched + st.interpolated;
        let rate = if total > 0 { 100.0 * st.matched as f64 / total as f64 } else { 0.0 };
        ui.label(format!("matched: {}", st.matched));
        ui.label(format!("interpolated: {}", st.interpolated));
        ui.label(format!("spurious: {}", st.spurious));
        ui.label(format!("match rate: {rate:.1}%"));
        ui.label(format!("drift: {:.1} ppm", st.drift_ppm));
        ui.label(format!("offset: {:.0} µs", st.offset_us));
        let mode = if s.pipeline.paired.load(Ordering::Relaxed) { "paired" } else { "free-run" };
        ui.label(format!("preview: {mode}"));
        ui.label(format!("解码: {}", s.pipeline.decode_path));

        if let Some(err) = s.pipeline.last_error.lock().unwrap().clone() {
            ui.separator();
            ui.colored_label(egui::Color32::RED, format!("error: {err}"));
        }

        ui.separator();
        egui::CollapsingHeader::new("重建 Reconstruction").default_open(false).show(ui, |ui| {
            ui_recon::recon_panel(ui, &mut s.recon_panel);
        });
    }


    fn calibrate_tab(&mut self, ui: &mut egui::Ui) -> bool {
        let Some(s) = &self.session else {
            ui.weak("未连接相机");
            return false;
        };
        ui_calib::calibrate_panel(
            ui,
            &s.pipeline.calib_state,
            &mut self.export_path,
            &mut self.show_coverage,
            self.geom.is_some(),
        )
    }


    fn source_dims(&self) -> Option<((u32, u32), (u32, u32))> {
        if self.geom.is_some() {
            let a = self.active_calib.as_ref()?;
            return Some((a.cam0.image_size, a.cam1.image_size));
        }
        let d = |t: &Option<egui::TextureHandle>| {
            t.as_ref().map(|t| {
                let s = t.size();
                (s[0] as u32, s[1] as u32)
            })
        };
        Some((d(&self.flir_tex)?, d(&self.recon_tex)?))
    }


    fn finish_calib_apply(&mut self, r: Result<serde_json::Value, String>) {
        let dims = self.source_dims();
        if let Err(e) = self.force_geom_off_for_slot_change() {
            self.active_calib_msg = Some(Err(e));
            return;
        }
        let msg = (|| {
            let a = fs_calib::active::load_stereo_calibration_v1(r?, fs_calib::CalibSource::LiveSession)?;
            let (d0, d1) = dims.ok_or("还没有预览帧,无法校验尺寸")?;
            a.validate_against(d0, d1)?;
            Ok(a)
        })()
        .map(|a: fs_calib::ActiveCalibration| {
            let msg = format!("已应用本次会话结果(stereo rms {:.3}px)", a.stereo_rms);
            self.active_calib = Some(a);
            msg
        });
        let ok = msg.is_ok();
        self.active_calib_msg = Some(msg);
        if ok {
            self.reset_geom_target_default();
        }
    }

    fn load_calibration(&mut self) {
        let path = self.calib_load_path.trim().to_string();
        let dims = self.source_dims();
        if let Err(e) = self.force_geom_off_for_slot_change() {
            self.active_calib_msg = Some(Err(e));
            return;
        }
        let msg = (|| {
            let text = std::fs::read_to_string(&path).map_err(|e| format!("读取 {path} 失败:{e}"))?;
            let doc: serde_json::Value =
                serde_json::from_str(&text).map_err(|e| format!("{path} 不是合法 JSON:{e}"))?;
            let a = fs_calib::active::load_stereo_calibration_v1(doc, fs_calib::CalibSource::Loaded(path.clone()))?;
            let (d0, d1) = dims.ok_or("还没有预览帧,无法校验尺寸")?;
            a.validate_against(d0, d1)?;
            Ok(a)
        })()
        .map(|a: fs_calib::ActiveCalibration| {
            let msg = format!("已加载 {path}(stereo rms {:.3}px)", a.stereo_rms);
            self.active_calib = Some(a);
            msg
        });
        let ok = msg.is_ok();
        self.active_calib_msg = Some(msg);
        if ok {
            self.reset_geom_target_default();
        }
    }

    fn force_geom_off_for_slot_change(&mut self) -> Result<(), String> {
        if self.geom.is_none() {
            return Ok(());
        }
        self.set_geom_mode(fs_calib::GeomMode::Off);
        if self.geom.is_none() {
            if !matches!(self.geom_msg, Some(Err(_))) {
                self.geom_msg = Some(Ok("更换标定 —— 几何处理已自动关闭(映射表基于旧标定)".to_string()));
            }
            Ok(())
        } else {
            let why = match &self.geom_msg {
                Some(Err(e)) => e.clone(),
                _ => "原因未知".to_string(),
            };
            Err(format!("几何处理开启中且无法关闭({why})—— 未更换标定"))
        }
    }


    fn reset_geom_target_default(&mut self) {
        if let Some(a) = &self.active_calib {
            let ((w0, h0), (w1, h1)) = (a.cam0.image_size, a.cam1.image_size);
            self.geom_target = (w0.max(w1), h0.max(h1));
        }
    }

    fn geom_gate_now(&self) -> Option<&'static str> {
        let Some(s) = &self.session else {
            return Some("未连接 —— 几何处理随连接会话生效");
        };
        let calib_session = {
            let c = s.pipeline.calib_state.lock().unwrap();
            c.enabled || c.session_exists
        };
        let recording =
            s.pipeline.record_state.lock().unwrap().recording || self.record_panel.start_pending();
        crate::ui_gate::geom_gate(self.active_calib.is_some(), calib_session, recording)
    }

    fn geometry_section_ui(&mut self, ui: &mut egui::Ui) -> Option<fs_calib::GeomMode> {
        let gate = self.geom_gate_now();
        let current = self.geom.as_ref().map(|g| g.mode).unwrap_or(fs_calib::GeomMode::Off);
        ui_calib::geometry_section(ui, current, &mut self.geom_target, gate, self.geom_msg.as_ref())
    }

    fn set_geom_mode(&mut self, want: fs_calib::GeomMode) {
        use fs_calib::GeomMode;
        let current = self.geom.as_ref().map(|g| g.mode).unwrap_or(GeomMode::Off);
        if want == current {
            return;
        }
        if let Some(reason) = self.geom_gate_now() {
            self.geom_msg = Some(Err(reason.to_string()));
            return;
        }
        let Some(s) = &mut self.session else {
            self.geom_msg = Some(Err("未连接 —— 几何处理随连接会话生效".to_string()));
            return;
        };

        if want == GeomMode::Off {
            let recon = match build_recon_at(s.recon_panel.choice, s.recon_dims) {
                Ok(r) => r,
                Err(e) => {
                    self.geom_msg = Some(Err(format!("关闭几何处理被拒绝:{e}")));
                    return;
                }
            };
            *s.pipeline.geom_maps.lock().unwrap_or_else(|p| p.into_inner()) = None;
            s.pipeline.event_xform.set(None);
            if let Err(e) = s.pipeline.recon.tx_swap.send(crate::recon_thread::ReconSwap { recon, dims: s.recon_dims }) {
                self.geom_msg = Some(Err(format!("重建线程未响应画幅切换:{e}")));
            } else {
                self.geom_msg = Some(Ok("几何处理已关闭(原始几何)".to_string()));
            }
            s.recon_panel.set_recon_dims(s.recon_dims);
            s.pipeline.calib_state.lock().unwrap().geometry_active = false;
            self.geom = None;
            return;
        }

        let Some(active) = &self.active_calib else {
            self.geom_msg = Some(Err("无当前标定 —— 无法推导映射表".to_string()));
            return;
        };
        let (m0, m1) = match fs_calib::derive(active, want) {
            Ok(v) => v,
            Err(e) => {
                self.geom_msg = Some(Err(format!("切换失败(保持{}):{e}", geom_mode_label(current))));
                return;
            }
        };
        let post_doc = match fs_calib::post_op_document(active, want, (&m0, &m1)) {
            Ok(v) => v,
            Err(e) => {
                self.geom_msg = Some(Err(format!("切换失败(保持{}):操作后文档生成失败:{e}", geom_mode_label(current))));
                return;
            }
        };
        let out = m1.out_size;
        let recon = match build_recon_at(s.recon_panel.choice, out) {
            Ok(r) => r,
            Err(e) => {
                self.geom_msg = Some(Err(format!("切换失败(保持{}):{e}", geom_mode_label(current))));
                return;
            }
        };
        let lut = transform::EventLut::from_cam_maps(&m1, active.cam1.image_size);
        let pair: transform::CamMapsPair = (
            transform::CamXform { src_size: active.cam0.image_size, maps: m0 },
            transform::CamXform { src_size: active.cam1.image_size, maps: m1 },
        );

        *s.pipeline.geom_maps.lock().unwrap_or_else(|p| p.into_inner()) = Some(pair);
        s.pipeline.event_xform.set(Some(lut));
        if let Err(e) = s.pipeline.recon.tx_swap.send(crate::recon_thread::ReconSwap { recon, dims: out }) {
            self.geom_msg = Some(Err(format!("重建线程未响应画幅切换:{e}")));
        } else {
            self.geom_msg =
                Some(Ok(format!("几何处理已切换:{}(事件画幅 {}×{})", geom_mode_label(want), out.0, out.1)));
        }
        s.recon_panel.set_recon_dims(out);
        s.pipeline.calib_state.lock().unwrap().geometry_active = true;
        self.geom = Some(GeomActive { mode: want, post_doc, out_size: out });
    }

    fn active_calib_ui(&mut self, ui: &mut egui::Ui) -> bool {
        let status = match &self.active_calib {
            None => "当前标定:无".to_string(),
            Some(a) => {
                let ((w0, h0), (w1, h1)) = (a.cam0.image_size, a.cam1.image_size);
                match &a.source {
                    fs_calib::CalibSource::Loaded(p) => {
                        format!("当前标定:文件 {p} · rms {:.3}px · {w0}x{h0} / {w1}x{h1}", a.stereo_rms)
                    }
                    fs_calib::CalibSource::LiveSession => {
                        format!("当前标定:本次会话 · rms {:.3}px · {w0}x{h0} / {w1}x{h1}", a.stereo_rms)
                    }
                }
            }
        };
        let can_validate = self.source_dims().is_some();
        ui_calib::active_calib_section(
            ui,
            &status,
            &mut self.calib_load_path,
            can_validate,
            self.session.as_ref().map(|s| &s.pipeline.calib_state),
            self.active_calib_msg.as_ref(),
        )
    }

    fn record_tab(&mut self, ui: &mut egui::Ui) {
        let Some(s) = &self.session else {
            ui.weak("未连接相机");
            return;
        };
        let geom_mode = self.geom.as_ref().map(|g| g.mode).unwrap_or(fs_calib::GeomMode::Off);
        let calib_note = match &self.active_calib {
            None => "无标定(take 将不含 calibration.json)".to_string(),
            Some(a) => {
                let src = match &a.source {
                    fs_calib::CalibSource::Loaded(p) => format!("附带标定:文件 {p}"),
                    fs_calib::CalibSource::LiveSession => "附带标定:本次会话".to_string(),
                };
                if self.geom.is_some() {
                    format!("{src}(操作后参数)")
                } else {
                    src
                }
            }
        };
        let calibration_source = self.active_calib.as_ref().map(|a| match &a.source {
            fs_calib::CalibSource::Loaded(p) => format!("loaded:{p}"),
            fs_calib::CalibSource::LiveSession => "live_session".to_string(),
        });
        let geometry_op = match geom_mode {
            fs_calib::GeomMode::Off => serde_json::json!({"mode": "off"}),
            fs_calib::GeomMode::Undistort => serde_json::json!({"mode": "undistort"}),
            fs_calib::GeomMode::SharedK { target } => {
                serde_json::json!({"mode": "shared_k", "target_size": [target.0, target.1]})
            }
        };
        let geometry_desc = match self.geom.as_ref() {
            None => geom_mode_label(fs_calib::GeomMode::Off),
            Some(g) => geom_mode_label(g.mode),
        };
        ui_record::record_panel(
            ui,
            &mut self.record_panel,
            ui_record::RecordHandles {
                record: &s.pipeline.record,
                record_state: &s.pipeline.record_state,
                rates: &s.pipeline.rates,
                tuning: &s.tuning,
                settings: &s.settings_panel,
                calib: &s.pipeline.calib_state,
                calibration_json: match &self.geom {
                    Some(g) => Some(&g.post_doc),
                    None => self.active_calib.as_ref().map(|a| &a.document),
                },
                calib_note: &calib_note,
                geometry_op,
                calibration_source,
                event_size: self.geom.as_ref().map(|g| g.out_size).unwrap_or(s.recon_dims),
                geometry_desc: &geometry_desc,
                geometry: geom_mode,
                has_active_calib: self.active_calib.is_some(),
            },
        );
    }

    fn tab_rail(&mut self, ui: &mut egui::Ui, frozen: bool) {
        const RAIL_WIDTH: f32 = 28.0;
        const V_PAD: f32 = 8.0;
        const CORNER: u8 = 6;
        let font = egui::TextStyle::Body.resolve(ui.style());
        let line_h = font.size + 6.0;

        ui.spacing_mut().item_spacing.y = 2.0;
        for t in Tab::ALL {
            let blocked = !PanelGate::evaluate(t, &self.connect, frozen).is_open();
            let mut glyphs: Vec<char> = t.label().chars().collect();
            if blocked {
                glyphs.insert(0, '⚠');
            }
            let h = V_PAD * 2.0 + line_h * glyphs.len() as f32;
            let (rect, resp) = ui.allocate_exact_size(egui::vec2(RAIL_WIDTH, h), egui::Sense::click());
            if resp.clicked() {
                self.tab = t;
            }
            let selected = self.tab == t;
            let visuals = ui.visuals();
            let fill = if selected {
                visuals.panel_fill
            } else if resp.hovered() {
                visuals.widgets.hovered.weak_bg_fill
            } else {
                visuals.extreme_bg_color
            };
            let mut paint_rect = rect;
            if selected {
                paint_rect.max.x += 1.0;
            }
            let corner = egui::CornerRadius { nw: CORNER, sw: CORNER, ne: 0, se: 0 };
            ui.painter().rect_filled(paint_rect, corner, fill);

            let text_color = visuals.text_color();
            let warn_color = visuals.warn_fg_color;
            let cx = rect.center().x;
            let mut y = rect.top() + V_PAD + line_h / 2.0;
            for ch in &glyphs {
                let is_warn_glyph = blocked && *ch == '⚠';
                ui.painter().text(
                    egui::pos2(cx, y),
                    egui::Align2::CENTER_CENTER,
                    ch,
                    font.clone(),
                    if is_warn_glyph { warn_color } else { text_color },
                );
                y += line_h;
            }
        }
    }

    fn sync_banner(&self) -> Option<(egui::Color32, &'static str)> {
        let s = self.session.as_ref()?;
        if self.policy.frozen() {
            return Some((egui::Color32::from_rgb(200, 60, 60), "同步已丢失 —— 标定与录制已冻结"));
        }
        let has_camera = s.live.is_some();
        if !has_camera {
            return None;
        }
        if self.policy.gave_up() {
            return Some((
                egui::Color32::from_rgb(200, 60, 60),
                "自动恢复失败 3 次,请检查 trigger 线后手动重新同步",
            ));
        }
        let h = *s.health.lock().unwrap();
        if h.state == SyncState::Lost {
            return Some((egui::Color32::from_rgb(200, 160, 40), "正在自动重新同步…"));
        }
        None
    }
}

fn activity_from(
    calib_state: &Mutex<calib_worker::CalibUiState>,
    record_state: &Mutex<record_worker::RecordUiState>,
) -> crate::sync_policy::Activity {
    crate::sync_policy::Activity {
        calibrating: calib_state.lock().unwrap().enabled,
        recording: record_state.lock().unwrap().recording,
    }
}

fn panic_message(e: Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = e.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = e.downcast_ref::<String>() {
        s.clone()
    } else {
        "(不可读的 panic 载荷)".to_string()
    }
}


fn install_cjk_font(ctx: &egui::Context) {
    const CANDIDATES: &[&str] = &[
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\msyhl.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
    ];
    for path in CANDIDATES {
        let Ok(bytes) = std::fs::read(path) else { continue };
        let mut fonts = egui::FontDefinitions::default();
        fonts.font_data.insert("cjk".into(), egui::FontData::from_owned(bytes).into());
        for family in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
            fonts.families.entry(family).or_default().push("cjk".into());
        }
        ctx.set_fonts(fonts);
        return;
    }
    eprintln!("warn: no system CJK font found; Chinese UI labels will render as boxes");
}

impl eframe::App for StudioApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.poll_connect();
        if let Some(s) = &self.session {
            if let Ok(img) = s.pipeline.flir_img.try_recv() {
                self.flir_tex = Some(ctx.load_texture("flir", img, egui::TextureOptions::LINEAR));
            }
            if let Ok(img) = s.pipeline.recon_img.try_recv() {
                self.recon_tex = Some(ctx.load_texture("recon", img, egui::TextureOptions::LINEAR));
            }
        }

        let apply = self.session.as_ref().and_then(|s| s.pipeline.calib_state.lock().unwrap().apply_result.take());
        if let Some(r) = apply {
            self.finish_calib_apply(r);
        }

        if let Some(s) = &self.session {
            s.pipeline.calib_state.lock().unwrap().geometry_active = self.geom.is_some();
        }

        if let Some(s) = &self.session {
            let h = *s.health.lock().unwrap();
            let a = activity_from(&s.pipeline.calib_state, &s.pipeline.record_state);
            let now_ms = (ui.input(|i| i.time) * 1000.0) as u64;
            match self.policy.decide(h, a, now_ms) {
                Action::AutoResync => {
                    if let Some(live) = s.live.as_ref() {
                        let _ = live.flir_cmd.try_send(FlirCmd::Resync);
                    }
                }
                Action::Freeze => {
                    s.pipeline.calib_state.lock().unwrap().frozen = self.policy.frozen();
                    if a.recording {
                        if let Err(e) = s.pipeline.record.send(record_worker::RecordCmd::Stop {
                            reason: record_worker::StopReason::SyncLost,
                        }) {
                            let msg = format!(
                                "失步后自动停止录制失败({e})—— 请到「相机」tab 重新同步;恢复后若仍在录制,请立即手动停止,失步之后的数据不可信"
                            );
                            eprintln!("record: {msg}");
                            self.record_panel.panel_error = Some(msg);
                        }
                    }
                }
                Action::Unfreeze => {
                    s.pipeline.calib_state.lock().unwrap().frozen = self.policy.frozen();
                }
                Action::GiveUp | Action::Nothing => {}
            }
        }

        if let Some((color, msg)) = self.sync_banner() {
            egui::Panel::top("sync_banner").show(ui, |ui| {
                ui.colored_label(color, msg);
            });
        }

        egui::Panel::right("panel").min_size(300.0).show(ui, |ui| {
            let frozen = self.policy.frozen();
            ui.horizontal_top(|ui| {
                ui.vertical(|ui| self.tab_rail(ui, frozen));
                ui.separator();
                ui.vertical(|ui| {
                    egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                        let gate = PanelGate::evaluate(self.tab, &self.connect, frozen);
                        if let Some(r) = gate.reason() {
                            ui.colored_label(egui::Color32::YELLOW, r.message());
                            ui.separator();
                        }
                        match self.tab {
                            Tab::Camera => self.camera_tab(ui),
                            Tab::Calibrate => {
                                if self.active_calib_ui(ui) {
                                    self.load_calibration();
                                }
                                if let Some(req) = self.geometry_section_ui(ui) {
                                    self.set_geom_mode(req);
                                }
                                let start_pending_geom_off =
                                    ui.add_enabled_ui(gate.is_open(), |ui| self.calibrate_tab(ui)).inner;
                                if start_pending_geom_off {
                                    self.set_geom_mode(fs_calib::GeomMode::Off);
                                    if self.geom.is_none() {
                                        if let Some(s) = &self.session {
                                            s.pipeline.calib_state.lock().unwrap().enabled = true;
                                        }
                                        if !matches!(self.geom_msg, Some(Err(_))) {
                                            self.geom_msg = Some(Ok(
                                                "标定会话要求原始几何 —— 几何处理已自动关闭".to_string()
                                            ));
                                        }
                                    }
                                }
                            }
                            Tab::Record => ui.add_enabled_ui(gate.is_open(), |ui| self.record_tab(ui)).inner,
                        }
                    });
                });
            });
        });

        egui::CentralPanel::default().show(ui, |ui| {
            let Some(s) = &self.session else {
                ui.centered_and_justified(|ui| ui.label("未连接相机"));
                return;
            };
            ui.columns(2, |cols| {
                cols[0].label("FLIR");
                if let Some(t) = &self.flir_tex {
                    let resp = cols[0].add(egui::Image::new(t).max_width(cols[0].available_width()));
                    let c = s.pipeline.calib_state.lock().unwrap();
                    let age0 = c.last_detect0_at.map(|at| at.elapsed());
                    ui_calib::draw_pane_overlay(
                        &cols[0],
                        resp.rect,
                        t.size_vec2(),
                        c.last_detect0.as_deref(),
                        age0,
                        c.last_detect0_accepted,
                        c.cam0.coverage,
                        self.show_coverage,
                    );
                }
                cols[1].label("EVK4 (recon)");
                if let Some(t) = &self.recon_tex {
                    let resp = cols[1].add(egui::Image::new(t).max_width(cols[1].available_width()));
                    let c = s.pipeline.calib_state.lock().unwrap();
                    let age1 = c.last_detect1_at.map(|at| at.elapsed());
                    ui_calib::draw_pane_overlay(
                        &cols[1],
                        resp.rect,
                        t.size_vec2(),
                        c.last_detect1.as_deref(),
                        age1,
                        c.last_detect1_accepted,
                        c.cam1.coverage,
                        self.show_coverage,
                    );
                }
            });
        });

        ctx.request_repaint_after(std::time::Duration::from_millis(16));
    }


    fn on_exit(&mut self) {
        self.session = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calib_worker::CalibUiState;
    use crate::record_worker::RecordUiState;


    #[test]
    fn activity_recording_reflects_the_workers_published_state() {
        let calib = Mutex::new(CalibUiState::default());
        let record = Mutex::new(RecordUiState { recording: true, ..RecordUiState::default() });
        let a = activity_from(&calib, &record);
        assert!(a.recording, "录制中的活动位必须来自 worker 发布的状态,不能恒为 false");
    }

    #[test]
    fn activity_recording_is_false_when_the_worker_says_so_too() {
        let calib = Mutex::new(CalibUiState::default());
        let record = Mutex::new(RecordUiState::default());
        let a = activity_from(&calib, &record);
        assert!(!a.recording);
    }
}
