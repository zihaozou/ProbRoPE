
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crossbeam_channel::{Sender, TrySendError};

use crate::settings::{
    clamp_f64, clamp_i32, clamp_u32, fps_apply_enabled, sanitize_af_band, should_copy_snapshot,
    AntiflickerMode, AutoMode, EvkSettingsSnapshot, FlirSettingsSnapshot, FpsApplyState,
    TrailFilterType, EVK_BIAS_NAMES,
};
use crate::sources::evk_thread::EvkCmd;
use crate::sources::live::FlirCmd;


const BUSY_FLASH: Duration = Duration::from_millis(1500);


const TEMP_WARN_C: f64 = 60.0;
const TEMP_HOT_C: f64 = 70.0;

pub struct FlirSettingsUi {
    pub cmd_tx: Sender<FlirCmd>,
    pub snapshot: Arc<Mutex<FlirSettingsSnapshot>>,
}

pub struct EvkSettingsUi {
    pub cmd_tx: Sender<EvkCmd>,
    pub snapshot: Arc<Mutex<EvkSettingsSnapshot>>,
}


pub struct SettingsHandles {
    pub flir: Option<FlirSettingsUi>,
    pub evk: Option<EvkSettingsUi>,
}

pub struct SettingsPanelState {
    flir: Option<FlirPanel>,
    evk: Option<EvkPanel>,
}

impl SettingsPanelState {
    pub fn new(handles: SettingsHandles) -> Self {
        SettingsPanelState {
            flir: handles.flir.map(FlirPanel::new),
            evk: handles.evk.map(EvkPanel::new),
        }
    }


    pub fn flir_json(&self) -> serde_json::Value {
        match &self.flir {
            Some(p) => snapshot_json(&*p.handles.snapshot.lock().unwrap()),
            None => serde_json::Value::Null,
        }
    }


    pub fn evk_json(&self) -> serde_json::Value {
        match &self.evk {
            Some(p) => snapshot_json(&*p.handles.snapshot.lock().unwrap()),
            None => serde_json::Value::Null,
        }
    }
}

fn snapshot_json<T: serde::Serialize>(snap: &T) -> serde_json::Value {
    serde_json::to_value(snap).unwrap_or_else(|e| serde_json::json!({ "serialize_error": e.to_string() }))
}

pub fn settings_panel(ui: &mut egui::Ui, state: &mut SettingsPanelState, recording: bool) {
    match &mut state.flir {
        Some(p) => flir_section(ui, p, recording),
        None => {
            ui.strong("FLIR");
            ui.weak("not connected (replay mode)");
        }
    }
    ui.add_space(8.0);
    match &mut state.evk {
        Some(p) => evk_section(ui, p),
        None => {
            ui.strong("EVK4");
            ui.weak("not connected");
        }
    }
}


struct FlirPanel {
    handles: FlirSettingsUi,

    seeded_revision: u64,

    last_device_fps: f64,
    exposure_us: f64,
    gain_db: f64,
    fps_edit: f64,
    busy_at: Option<Instant>,
    disconnected: bool,

    exposure_throttle: crate::throttle::SendThrottle,
    gain_throttle: crate::throttle::SendThrottle,
}

impl FlirPanel {
    fn new(handles: FlirSettingsUi) -> Self {
        FlirPanel {
            handles,
            seeded_revision: 0,
            last_device_fps: 0.0,
            exposure_us: 0.0,
            gain_db: 0.0,
            fps_edit: 0.0,
            busy_at: None,
            disconnected: false,
            exposure_throttle: Default::default(),
            gain_throttle: Default::default(),
        }
    }

    fn send(&mut self, cmd: FlirCmd) {
        match self.handles.cmd_tx.try_send(cmd) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) => self.busy_at = Some(Instant::now()),
            Err(TrySendError::Disconnected(_)) => self.disconnected = true,
        }
    }
}

fn flir_section(ui: &mut egui::Ui, p: &mut FlirPanel, recording: bool) {
    ui.strong("FLIR Blackfly S");
    let snap = p.handles.snapshot.lock().unwrap().clone();
    if !snap.populated {
        ui.weak("waiting for device…");
        return;
    }
    if recording {
        ui.weak("录制中:fps Apply 与 IspEnable 已锁定(两者都会停止采集);曝光 / 增益仍可实时调整。");
    }

    if should_copy_snapshot(snap.populated, snap.revision, p.seeded_revision) {
        p.exposure_us = snap.exposure_us;
        p.gain_db = snap.gain_db;
        p.seeded_revision = snap.revision;
    }
    if snap.fps != p.last_device_fps {
        p.fps_edit = snap.fps;
        p.last_device_fps = snap.fps;
    }

    let now_ms = (ui.input(|i| i.time) * 1000.0) as u64;

    let mut exposure_auto = snap.exposure_auto;
    combo_auto(ui, "flir_exposure_auto", "exposure auto", &mut exposure_auto);
    if exposure_auto != snap.exposure_auto {
        p.send(FlirCmd::SetExposureAuto(exposure_auto));
    }
    let exposure_editable = snap.exposure_auto == AutoMode::Off;
    if !exposure_editable {
        p.exposure_us = snap.exposure_us;
    }
    let exp_range = snap.exposure_range_us;
    let resp = ui.add_enabled(
        exposure_editable,
        egui::Slider::new(&mut p.exposure_us, exp_range.0..=exp_range.1)
            .logarithmic(true)
            .text("exposure µs"),
    );
    if p.exposure_throttle.update(resp.changed(), resp.drag_stopped(), now_ms) {
        p.send(FlirCmd::SetExposureUs(clamp_f64(p.exposure_us, exp_range)));
    }

    let mut gain_auto = snap.gain_auto;
    combo_auto(ui, "flir_gain_auto", "gain auto", &mut gain_auto);
    if gain_auto != snap.gain_auto {
        p.send(FlirCmd::SetGainAuto(gain_auto));
    }
    let gain_editable = snap.gain_auto == AutoMode::Off;
    if !gain_editable {
        p.gain_db = snap.gain_db;
    }
    let gain_range = snap.gain_range_db;
    let resp = ui.add_enabled(
        gain_editable,
        egui::Slider::new(&mut p.gain_db, gain_range.0..=gain_range.1).text("gain dB"),
    );
    if p.gain_throttle.update(resp.changed(), resp.drag_stopped(), now_ms) {
        p.send(FlirCmd::SetGainDb(clamp_f64(p.gain_db, gain_range)));
    }

    let mut isp = snap.isp_enable;
    if ui
        .add_enabled(!recording, egui::Checkbox::new(&mut isp, "IspEnable (on-camera ISP)"))
        .on_hover_text(
            "Toggling briefly restarts the stream. OFF (default) = raw Bayer out, host-side \
             demosaic -- matches the pipeline's RgbFrame design. ON enables the camera's \
             color/sharpening/saturation processing blocks.",
        )
        .on_disabled_hover_text("录制中不可用:切换 ISP 会重启采集,take 会断在这里。请先停止录制。")
        .changed()
    {
        p.send(FlirCmd::SetIspEnable(isp));
    }

    ui.horizontal(|ui| {
        ui.label("fps");
        ui.add(
            egui::DragValue::new(&mut p.fps_edit)
                .range(snap.fps_range.0..=snap.fps_range.1)
                .speed(0.1)
                .max_decimals(3),
        );
        let enabled =
            fps_apply_enabled(snap.populated, &snap.fps_apply, p.fps_edit, snap.fps_range) && !recording;
        if ui
            .add_enabled(enabled, egui::Button::new("Apply"))
            .on_hover_text("Stops the FLIR, re-runs the software-trigger sync handshake at the new fps, and reseeds the sync engine (~2s; both previews pause).")
            .on_disabled_hover_text(if recording {
                "录制中不可用:改 fps 会停止采集并重新握手,take 会断在这里。请先停止录制。"
            } else {
                "fps 不在设备支持的范围内,或上一次 Apply 还没完成。"
            })
            .clicked()
        {
            p.handles.snapshot.lock().unwrap().fps_apply = FpsApplyState::InFlight;
            if let Err(e) = p.handles.cmd_tx.try_send(FlirCmd::ApplyFps(p.fps_edit)) {
                let reason = match e {
                    TrySendError::Full(_) => "camera thread busy -- try again".to_string(),
                    TrySendError::Disconnected(_) => "FLIR thread not running".to_string(),
                };
                p.handles.snapshot.lock().unwrap().fps_apply = FpsApplyState::Failed(reason);
            }
        }
    });
    match &snap.fps_apply {
        FpsApplyState::Idle => {}
        FpsApplyState::InFlight => {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label("re-handshaking… (~2s, previews pause)");
            });
        }
        FpsApplyState::Failed(e) => {
            ui.colored_label(egui::Color32::RED, format!("fps apply failed: {e}"));
        }
    }

    if let Some(t) = snap.temperature_c {
        let color = if t >= TEMP_HOT_C {
            egui::Color32::RED
        } else if t >= TEMP_WARN_C {
            egui::Color32::YELLOW
        } else {
            ui.visuals().weak_text_color()
        };
        ui.colored_label(color, format!("sensor temp: {t:.1} °C"));
    }
    ui.weak(format!(
        "device: {:.3} fps · exposure {:.0} µs · gain {:.2} dB",
        snap.fps, snap.exposure_us, snap.gain_db
    ));

    status_lines(ui, snap.last_error.as_deref(), p.busy_at, p.disconnected, "FLIR");
}


struct EvkPanel {
    handles: EvkSettingsUi,
    seeded_revision: u64,
    expert_ranges: bool,
    biases: [i32; 5],
    erc_count: u32,
    af_low_hz: u32,
    af_high_hz: u32,
    trail_threshold_us: u32,
    busy_at: Option<Instant>,
    disconnected: bool,
}

impl EvkPanel {
    fn new(handles: EvkSettingsUi) -> Self {
        EvkPanel {
            handles,
            seeded_revision: 0,
            expert_ranges: false,
            biases: [0; 5],
            erc_count: 0,
            af_low_hz: 0,
            af_high_hz: 0,
            trail_threshold_us: 0,
            busy_at: None,
            disconnected: false,
        }
    }

    fn send(&mut self, cmd: EvkCmd) {
        match self.handles.cmd_tx.try_send(cmd) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) => self.busy_at = Some(Instant::now()),
            Err(TrySendError::Disconnected(_)) => self.disconnected = true,
        }
    }
}

fn evk_section(ui: &mut egui::Ui, p: &mut EvkPanel) {
    ui.strong("EVK4 (IMX636)");
    let snap = p.handles.snapshot.lock().unwrap().clone();
    if !snap.populated {
        ui.weak("waiting for device…");
        return;
    }
    if !(snap.biases_available || snap.erc_available || snap.af_available || snap.trail_available) {
        ui.weak("device settings unavailable (file replay)");
        return;
    }

    if should_copy_snapshot(snap.populated, snap.revision, p.seeded_revision) {
        for i in 0..5 {
            p.biases[i] = snap.biases[i].value;
        }
        p.erc_count = snap.erc_event_count;
        p.af_low_hz = snap.af_band_hz.0;
        p.af_high_hz = snap.af_band_hz.1;
        p.trail_threshold_us = snap.trail_threshold_us;
        p.seeded_revision = snap.revision;
    }

    if snap.biases_available {
        ui.checkbox(&mut p.expert_ranges, "expert bias ranges")
            .on_hover_text("Sliders span the sensor's full ALLOWED range instead of the safe RECOMMENDED range. Out-of-recommended values can distort the sensor response.");
        for i in 0..5 {
            if !snap.biases[i].available {
                ui.weak(format!("{}: read failed — not adjustable", EVK_BIAS_NAMES[i]));
                continue;
            }
            let range = if p.expert_ranges { snap.biases[i].allowed } else { snap.biases[i].recommended };
            let changed = ui
                .add(egui::Slider::new(&mut p.biases[i], range.0..=range.1).text(EVK_BIAS_NAMES[i]))
                .changed();
            if changed {
                p.send(EvkCmd::SetBias { name: EVK_BIAS_NAMES[i], value: clamp_i32(p.biases[i], snap.biases[i].allowed) });
            }
        }
    } else {
        ui.weak("biases: not available");
    }

    ui.add_space(4.0);

    if snap.erc_available {
        let mut on = snap.erc_enabled;
        if ui
            .checkbox(&mut on, "ERC (event-rate control)")
            .on_hover_text("Sensor-side rate limiter -- the first lever against recon overload (see the 重建 panel's banner).")
            .changed()
        {
            p.send(EvkCmd::SetErcEnable(on));
        }
        let range = snap.erc_count_range;
        let changed = ui
            .add_enabled(
                snap.erc_enabled,
                egui::Slider::new(&mut p.erc_count, range.0..=range.1).logarithmic(true).text("events/period"),
            )
            .changed();
        if changed {
            p.send(EvkCmd::SetErcCdEventCount(clamp_u32(p.erc_count, range)));
        }
        if snap.erc_period_us > 0 {
            ui.weak(format!(
                "≈ {:.1} Mev/s (period {} µs)",
                p.erc_count as f64 / snap.erc_period_us as f64,
                snap.erc_period_us
            ));
        }
    } else {
        ui.weak("ERC: not available");
    }

    ui.add_space(4.0);

    if snap.af_available {
        let mut on = snap.af_enabled;
        if ui.checkbox(&mut on, "anti-flicker").changed() {
            p.send(EvkCmd::SetAntiflickerEnable(on));
        }
        let sup = snap.af_freq_range_hz;
        let band_resp = ui.horizontal(|ui| {
            ui.label("band");
            let lo = ui.add(egui::DragValue::new(&mut p.af_low_hz).range(sup.0..=sup.1).suffix(" Hz"));
            ui.label("–");
            let hi = ui.add(egui::DragValue::new(&mut p.af_high_hz).range(sup.0..=sup.1).suffix(" Hz"));
            lo.changed() || hi.changed()
        });
        if band_resp.inner {
            let (lo, hi) = sanitize_af_band(p.af_low_hz, p.af_high_hz, sup);
            p.send(EvkCmd::SetAntiflickerBand { low_hz: lo, high_hz: hi });
        }
        let mut mode = snap.af_mode;
        let prev_mode = mode;
        egui::ComboBox::from_id_salt("evk_af_mode")
            .selected_text(mode.label())
            .show_ui(ui, |ui| {
                for m in AntiflickerMode::ALL {
                    ui.selectable_value(&mut mode, m, m.label());
                }
            });
        if mode != prev_mode {
            p.send(EvkCmd::SetAntiflickerMode(mode));
        }
    } else {
        ui.weak("anti-flicker: not available");
    }

    ui.add_space(4.0);

    if snap.trail_available {
        let mut on = snap.trail_enabled;
        if ui
            .checkbox(&mut on, "event trail filter (STC)")
            .on_hover_text("Suppresses trailing-event bursts (防拖尾). Set type + threshold before enabling for a clean start; changing them while enabled micro-resets the filter.")
            .changed()
        {
            p.send(EvkCmd::SetTrailFilterEnable(on));
        }
        let mut ty = snap.trail_type;
        let prev_ty = ty;
        egui::ComboBox::from_id_salt("evk_trail_type")
            .selected_text(ty.label())
            .show_ui(ui, |ui| {
                for t in TrailFilterType::ALL {
                    if snap.trail_available_types == 0 || snap.trail_available_types & t.bit() != 0 {
                        ui.selectable_value(&mut ty, t, t.label());
                    }
                }
            });
        if ty != prev_ty {
            p.send(EvkCmd::SetTrailFilterType(ty));
        }
        let range = snap.trail_threshold_range_us;
        let changed = ui
            .add(egui::Slider::new(&mut p.trail_threshold_us, range.0..=range.1).logarithmic(true).text("threshold µs"))
            .changed();
        if changed {
            p.send(EvkCmd::SetTrailFilterThresholdUs(clamp_u32(p.trail_threshold_us, range)));
        }
    } else {
        ui.weak("trail filter: not available");
    }

    status_lines(ui, snap.last_error.as_deref(), p.busy_at, p.disconnected, "EVK4");
}


fn combo_auto(ui: &mut egui::Ui, id: &str, label: &str, value: &mut AutoMode) {
    ui.horizontal(|ui| {
        egui::ComboBox::from_id_salt(id)
            .selected_text(value.node_entry())
            .show_ui(ui, |ui| {
                for m in AutoMode::ALL {
                    ui.selectable_value(value, m, m.node_entry());
                }
            });
        ui.label(label);
    });
}


fn status_lines(ui: &mut egui::Ui, last_error: Option<&str>, busy_at: Option<Instant>, disconnected: bool, who: &str) {
    if let Some(err) = last_error {
        ui.colored_label(egui::Color32::RED, format!("apply error: {err}"));
    }
    if busy_at.is_some_and(|t| t.elapsed() < BUSY_FLASH) {
        ui.colored_label(egui::Color32::YELLOW, "camera thread busy — adjustment dropped, re-drag to retry");
    }
    if disconnected {
        ui.colored_label(egui::Color32::RED, format!("{who} thread not running — settings inert"));
    }
}
