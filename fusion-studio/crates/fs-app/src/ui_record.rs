
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use fs_record::unix_ms_now;
use fs_record::video::{probe_ffmpeg, FrameStorage};
use windows_sys::Win32::Foundation::SYSTEMTIME;
use windows_sys::Win32::System::SystemInformation::GetLocalTime;

use crate::calib_worker::CalibUiState;
use crate::record_worker::{RecordCmd, RecordUiState, StopReason, RECORD_QUEUE_BYTES};
use crate::settings::SharedTuning;
use crate::stream_rates::{RateBasis, RateWindow, StreamRates, EVENTS_BIN_BYTES_PER_EVENT, EVT3_BYTES_PER_EVENT};
use crate::take_plan::{self, ManifestInputs, TakeInputs, TakePlan};
use crate::ui_settings::SettingsPanelState;
use crate::wiring::RecordChannel;

const START_TIMEOUT: Duration = Duration::from_secs(4);

const FREE_SPACE_POLL: Duration = Duration::from_secs(1);

const QUEUE_WARN_FRAC: f64 = 0.5;

pub struct RecordHandles<'a> {
    pub record: &'a RecordChannel,
    pub record_state: &'a Arc<Mutex<RecordUiState>>,
    pub rates: &'a StreamRates,
    pub tuning: &'a SharedTuning,

    pub settings: &'a SettingsPanelState,

    pub calib: &'a Arc<Mutex<CalibUiState>>,

    pub calibration_json: Option<&'a serde_json::Value>,

    pub calib_note: &'a str,
    pub geometry_op: serde_json::Value,
    pub calibration_source: Option<String>,

    pub event_size: (u32, u32),
    pub geometry_desc: &'a str,
    pub geometry: fs_calib::GeomMode,
    pub has_active_calib: bool,
}


struct FfmpegProber {
    slot: Arc<Mutex<Option<(String, Result<(), String>)>>>,
    busy: Arc<AtomicBool>,
    last_kick: Option<Instant>,
    last_key: String,
}

impl Default for FfmpegProber {
    fn default() -> Self {
        FfmpegProber { slot: Arc::default(), busy: Arc::default(), last_kick: None, last_key: String::new() }
    }
}

impl FfmpegProber {
    fn maintain(&mut self, path: &Path, now: Instant) {
        let key = path.display().to_string();
        let due = self.last_key != key || self.last_kick.is_none_or(|t| now.duration_since(t) >= FREE_SPACE_POLL);
        if !due || self.busy.load(Ordering::Relaxed) {
            return;
        }
        self.last_key = key.clone();
        self.last_kick = Some(now);
        self.busy.store(true, Ordering::Relaxed);
        let slot = self.slot.clone();
        let busy = self.busy.clone();
        let path = path.to_path_buf();
        std::thread::spawn(move || {
            let r = probe_ffmpeg(&path);
            *slot.lock().unwrap_or_else(|p| p.into_inner()) = Some((key, r));
            busy.store(false, Ordering::Relaxed);
        });
    }


    fn result_for(&self, path: &Path) -> Option<Result<(), String>> {
        let key = path.display().to_string();
        let guard = self.slot.lock().unwrap_or_else(|p| p.into_inner());
        guard.as_ref().and_then(|(k, r)| (k == &key).then(|| r.clone()))
    }
}


struct Pending {
    since: Instant,

    prior_error: Option<String>,
}

pub struct RecordPanelState {
    pub dir: String,
    pub name: String,

    pub storage: FrameStorage,
    pub ffmpeg: PathBuf,

    cq: u8,

    prober: FfmpegProber,

    rates: RateWindow,
    free_bytes: Option<u64>,
    free_at: Option<Instant>,
    free_key: String,
    pending: Option<Pending>,
    pub(crate) panel_error: Option<String>,

    sync_lost_resolved: bool,
}

impl Default for RecordPanelState {
    fn default() -> Self {
        RecordPanelState {
            dir: ".\\takes\\".to_string(),
            name: default_take_name(),
            storage: FrameStorage::Rgb24Bin,
            ffmpeg: PathBuf::from("ffmpeg"),
            cq: 19,
            prober: FfmpegProber::default(),
            rates: RateWindow::default(),
            free_bytes: None,
            free_at: None,
            free_key: String::new(),
            pending: None,
            panel_error: None,
            sync_lost_resolved: false,
        }
    }
}

impl RecordPanelState {

    pub(crate) fn start_pending(&self) -> bool {
        self.pending.is_some()
    }
}

pub fn record_panel(ui: &mut egui::Ui, p: &mut RecordPanelState, h: RecordHandles<'_>) {
    let st = h.record_state.lock().unwrap().clone();
    let now = Instant::now();

    let (ev_bytes, ev_count) = h.rates.event_totals();
    p.rates.observe(now, ev_bytes, ev_count);

    if p.free_key != p.dir || p.free_at.is_none_or(|t| now.duration_since(t) >= FREE_SPACE_POLL) {
        p.free_bytes = take_plan::free_space(&take_plan::absolute_dir(&p.dir));
        p.free_key = p.dir.clone();
        p.free_at = Some(now);
    }
    let ffmpeg_probe = if matches!(p.storage, FrameStorage::Rgb24Bin) {
        None
    } else {
        if !st.recording {
            p.prober.maintain(&p.ffmpeg, now);
        }
        p.prober.result_for(&p.ffmpeg)
    };

    let fps = h.tuning.fps();
    let plan = TakePlan::compute(
        TakeInputs {
            dir_text: &p.dir,
            name: &p.name,
            fps,
            frame: h.rates.frame_spec(),
            event: p.rates.rate(),
            storage: p.storage,
            ffmpeg_probe: ffmpeg_probe.as_ref(),
            geometry: h.geometry,
            has_active_calib: h.has_active_calib,
        },
        p.free_bytes,
    );

    resolve_pending(p, &st, now);

    ui.heading("录制 Record");

    let editable = !st.recording && p.pending.is_none();
    ui.add_enabled_ui(editable, |ui| {
        ui.horizontal(|ui| {
            ui.label("目录");
            ui.add(egui::TextEdit::singleline(&mut p.dir).desired_width(f32::INFINITY));
        });
        ui.horizontal(|ui| {
            ui.label("名称");
            ui.add(egui::TextEdit::singleline(&mut p.name).desired_width(f32::INFINITY));
        });
        ui.horizontal(|ui| {
            ui.label("帧存储");
            if ui.radio(matches!(p.storage, FrameStorage::Rgb24Bin), "RGB24 裸").clicked() {
                p.storage = FrameStorage::Rgb24Bin;
            }
            if ui
                .radio(matches!(p.storage, FrameStorage::HevcLossless), "HEVC 无损")
                .on_hover_text(
                    "hevc_nvenc lossless(yuv444)。RGB↔YUV 8bit 往返有 ±1 量化,非逐位无损。\n\
                     注意:yuv444 是 HEVC Rext profile,Windows 自带播放器/多数硬解码器不支持,\
                     直接播放会花屏 —— 文件没坏,用 mpv、新版 VLC 或 ffmpeg/OpenCV(软件解码)打开。",
                )
                .clicked()
            {
                p.storage = FrameStorage::HevcLossless;
            }
            if ui.radio(matches!(p.storage, FrameStorage::HevcLossy { .. }), "HEVC 有损").clicked() {
                p.storage = FrameStorage::HevcLossy { cq: p.cq };
            }
        });
        if matches!(p.storage, FrameStorage::HevcLossy { .. }) {
            ui.horizontal(|ui| {
                ui.label("CQ");
                if ui.add(egui::DragValue::new(&mut p.cq).range(1..=51)).changed() {
                    p.storage = FrameStorage::HevcLossy { cq: p.cq };
                }
                ui.weak("1–51,越小越接近无损(默认 19)");
            });
        }
        if !matches!(p.storage, FrameStorage::Rgb24Bin) {
            ui.horizontal(|ui| {
                ui.label("ffmpeg");
                let mut text = p.ffmpeg.display().to_string();
                if ui.add(egui::TextEdit::singleline(&mut text).desired_width(f32::INFINITY)).changed() {
                    p.ffmpeg = PathBuf::from(text);
                }
            });
        }
    });
    if !st.recording && p.pending.is_none() {
        ui.weak(format!("落盘到 {}", plan.take_dir(&p.name).display()));
    }

    ui.add_space(4.0);
    controls(ui, p, &h, &st, &plan, fps, now);

    ui.add_space(6.0);
    if st.recording {
        recording_lines(ui, &st, &plan);
    } else if st.finalizing {
        ui.horizontal(|ui| {
            ui.spinner();
            ui.label("收尾中…(正在等编码器写完,最长 45 秒)");
        });
    } else {
        preflight_lines(ui, &plan, h.calib_note, &p.storage, h.geometry_desc, h.event_size);
        if p.pending.is_none() {
            last_take_summary(ui, &st);
            sync_lost_prompt(ui, p, &st);
        }
    }

    if p.pending.is_none() {
        if let Some(e) = &st.last_error {
            ui.colored_label(egui::Color32::RED, format!("录制错误:{e}"));
        }
    }
    if let Some(e) = &p.panel_error {
        ui.colored_label(egui::Color32::RED, e);
    }
}


fn controls(
    ui: &mut egui::Ui,
    p: &mut RecordPanelState,
    h: &RecordHandles<'_>,
    st: &RecordUiState,
    plan: &TakePlan,
    fps: f64,
    now: Instant,
) {
    if st.recording {
        if ui
            .button("停止")
            .on_hover_text("停止并收尾:把队列里剩下的帧写完、停止事件录制、把停止时刻的统计补进 manifest。")
            .clicked()
        {
            p.panel_error =
                h.record.send(RecordCmd::Stop { reason: StopReason::User }).err().map(|e| format!("停止命令未送达:{e}"));
        }
        return;
    }
    if p.pending.is_some() {
        ui.horizontal(|ui| {
            ui.spinner();
            ui.label("启动中…(正在等录制线程确认 take 已经开始)");
        });
        return;
    }
    if let Some(reason) = &plan.refusal {
        ui.colored_label(egui::Color32::YELLOW, reason.clone());
    }
    if ui
        .add_enabled(plan.refusal.is_none(), egui::Button::new("开始录制"))
        .on_hover_text(
            "开一个新 take:帧按所选存储档落盘(frames.bin 或 frames.mkv),事件写进 events.bin,每帧的同步真值写进 sync.jsonl。",
        )
        .clicked()
    {
        start(p, h, plan, fps, st, now);
    }
}

fn start(
    p: &mut RecordPanelState,
    h: &RecordHandles<'_>,
    plan: &TakePlan,
    fps: f64,
    st: &RecordUiState,
    now: Instant,
) {
    let Some(frame) = plan.frame else { return };
    let board = {
        let c = h.calib.lock().unwrap();
        take_plan::board_json(c.session_exists, &c.board)
    };
    let manifest = take_plan::manifest(
        ManifestInputs {
            take_name: &p.name,
            fps,
            exposure_us: h.tuning.exposure_us(),
            frame,
            flir: h.settings.flir_json(),
            evk: h.settings.evk_json(),
            board,
        },
        unix_ms_now(),
    );
    let cmd = RecordCmd::Start {
        dir: plan.take_dir(&p.name),
        manifest,
        opts: take_plan::writer_opts(plan.frame_bps, plan.free_bytes),
        calibration_json: h.calibration_json.cloned(),
        storage: p.storage,
        ffmpeg: p.ffmpeg.clone(),
        event_size: h.event_size,
        geometry_op: h.geometry_op.clone(),
        calibration_source: h.calibration_source.clone(),
    };
    match h.record.send(cmd) {
        Ok(()) => on_start_sent(p, now, st.last_error.clone()),
        Err(e) => p.panel_error = Some(format!("开始录制命令未送达:{e}")),
    }
}


fn on_start_sent(p: &mut RecordPanelState, now: Instant, prior_error: Option<String>) {
    p.panel_error = None;
    p.pending = Some(Pending { since: now, prior_error });
    p.name = default_take_name();
    p.sync_lost_resolved = false;
}


fn resolve_pending(p: &mut RecordPanelState, st: &RecordUiState, now: Instant) {
    let Some(pend) = &p.pending else { return };
    if st.recording || st.last_error != pend.prior_error {
        p.pending = None;
        return;
    }
    if now.duration_since(pend.since) >= START_TIMEOUT {
        p.pending = None;
        if st.last_error.is_none() {
            p.panel_error =
                Some(format!("等了 {} 秒仍未开始录制,而且录制线程没有给出原因 —— 请重试;若反复如此,重启程序。", START_TIMEOUT.as_secs()));
        }
    }
}


fn storage_label(s: &FrameStorage) -> String {
    match s {
        FrameStorage::Rgb24Bin => "RGB24 裸".to_string(),
        FrameStorage::HevcLossless => "HEVC 无损(±1 量化,非逐位无损)".to_string(),
        FrameStorage::HevcLossy { cq } => format!("HEVC 有损 CQ{cq}"),
    }
}

fn preflight_lines(
    ui: &mut egui::Ui,
    plan: &TakePlan,
    calib_note: &str,
    storage: &FrameStorage,
    geometry_desc: &str,
    event_size: (u32, u32),
) {
    ui.strong("预检");
    let frame_part = match plan.frame {
        Some(f) => format!("{} {}×{}({geometry_desc})", storage_label(storage), f.w, f.h),
        None => format!("{}(帧几何未知)", storage_label(storage)),
    };
    let total = match plan.total_bps {
        Some(bps) => format!("约 {}", mb_s(bps)),
        None => "码率测量中".to_string(),
    };
    ui.label(format!("将录制:{frame_part} + events.bin {}×{},合计{total}", event_size.0, event_size.1));
    match (plan.frame, plan.frame_bps) {
        (Some(f), Some(bps)) => {
            ui.label(format!(
                "帧 {}×{} {} @ {:.1} fps = {}({})",
                f.w, f.h, f.format_name(), plan.fps, mb_s(bps), plan.frame_basis
            ));
        }
        _ => {
            ui.label("帧 —— 还没有同步帧,尺寸与格式未知");
        }
    }
    match plan.event {
        Some(e) => {
            let basis = match e.basis {
                RateBasis::RawEvt3Bytes => format!(
                    "近 5 秒实测,EVT3 原始字节 ÷{EVT3_BYTES_PER_EVENT} 估事件数 × {EVENTS_BIN_BYTES_PER_EVENT:.0} 字节/事件"
                ),
                RateBasis::DecodedEvents => {
                    format!("近 5 秒实测,解码事件数 × {EVENTS_BIN_BYTES_PER_EVENT:.0} 字节/事件(events.bin)")
                }
            };
            ui.label(format!("事件 {}({basis})", mb_s(e.bytes_per_s)));
        }
        None => {
            ui.label("事件 —— 正在测量(需要几秒)");
        }
    }
    let free = match plan.free_bytes {
        Some(b) => format!("剩余 {}", gb(b)),
        None => "剩余空间未知".to_string(),
    };
    match (plan.total_bps, plan.minutes) {
        (Some(bps), Some(min)) => {
            ui.label(format!("合计 {} · {free} → 约可录 {}", mb_s(bps), minutes(min)));
        }
        _ => {
            ui.label(format!("合计 —— 事件码率还没测出来,暂不估算可录时长 · {free}"));
        }
    }
    ui.label(calib_note);
}

fn recording_lines(ui: &mut egui::Ui, st: &RecordUiState, plan: &TakePlan) {
    let elapsed = st.started_at.map(|t| t.elapsed()).unwrap_or_default();
    ui.horizontal(|ui| {
        let (rect, _) = ui.allocate_exact_size(egui::vec2(10.0, 10.0), egui::Sense::hover());
        ui.painter().circle_filled(rect.center(), 5.0, egui::Color32::RED);
        ui.strong(format!("正在录制 {}", hms(elapsed)));
    });
    if let Some(d) = &st.take_dir {
        ui.weak(format!("落盘到 {}", d.display()));
    }
    let secs = elapsed.as_secs_f64();
    let avg = if secs > 0.0 { st.bytes_written as f64 / secs } else { 0.0 };
    ui.label(format!("帧 {} · 已写 {} · 平均 {}", st.frames_written, gb(st.bytes_written), mb_s(avg)));

    let drop_text = format!("丢帧 {}", st.dropped_frames);
    if st.dropped_frames > 0 {
        ui.colored_label(egui::Color32::RED, drop_text);
    } else {
        ui.label(drop_text);
    }

    let ev_text = format!("事件 {} · 丢弃 {}", st.events_written, st.events_dropped);
    if st.events_dropped > 0 {
        ui.colored_label(egui::Color32::RED, ev_text);
    } else {
        ui.label(ev_text);
    }
    let oob_text = format!("越界 {}(几何 LUT 丢弃)", st.events_dropped_oob);
    if st.events_dropped_oob > 0 {
        ui.colored_label(egui::Color32::RED, oob_text);
    } else {
        ui.label(oob_text);
    }

    let frac = st.queued_bytes as f64 / RECORD_QUEUE_BYTES as f64;
    let queue_text =
        format!("队列 {} / {}({:.0}%)", gb(st.queued_bytes), gb(RECORD_QUEUE_BYTES), frac * 100.0);
    if frac >= QUEUE_WARN_FRAC {
        ui.colored_label(egui::Color32::YELLOW, format!("{queue_text} —— 写盘跟不上,再涨就会开始丢帧"));
    } else {
        ui.label(queue_text);
    }

    match (plan.free_bytes, plan.minutes) {
        (Some(b), Some(min)) => ui.label(format!("剩余 {} —— 按当前码率约还能录 {}", gb(b), minutes(min))),
        (Some(b), None) => ui.label(format!("剩余 {}", gb(b))),
        (None, _) => ui.label("剩余空间未知"),
    };
}

fn last_take_summary(ui: &mut egui::Ui, st: &RecordUiState) {
    let Some(dir) = &st.take_dir else { return };
    ui.add_space(4.0);
    ui.weak(format!("上一个 take:{} 帧 · {}", st.frames_written, gb(st.bytes_written)));
    ui.weak(dir.display().to_string());
    if st.dropped_frames > 0 {
        ui.colored_label(egui::Color32::RED, format!("丢帧 {}", st.dropped_frames));
    }
    if st.sync_lost {
        ui.colored_label(egui::Color32::RED, "该 take 因失步中断,失步之后的 t_evk_us 不可信");
    }
}

fn sync_lost_prompt(ui: &mut egui::Ui, p: &mut RecordPanelState, st: &RecordUiState) {
    let Some(dir) = pending_sync_lost_question(p, st) else { return };
    ui.add_space(4.0);
    ui.horizontal(|ui| {
        ui.label("这段素材要删除还是保留?");
        if ui.button("删除该 take").clicked() {
            if let Err(e) = resolve_sync_lost(true, dir) {
                p.panel_error = Some(format!("删除 {} 失败:{e}", dir.display()));
            }
            p.sync_lost_resolved = true;
        }
        if ui.button("保留").clicked() {
            p.sync_lost_resolved = true;
        }
    });
}


fn pending_sync_lost_question<'a>(p: &RecordPanelState, st: &'a RecordUiState) -> Option<&'a Path> {
    if !st.sync_lost || p.sync_lost_resolved {
        return None;
    }
    st.take_dir.as_deref()
}


fn resolve_sync_lost(delete: bool, dir: &Path) -> std::io::Result<()> {
    if delete {
        std::fs::remove_dir_all(dir)
    } else {
        Ok(())
    }
}

fn default_take_name() -> String {
    let mut t = SYSTEMTIME {
        wYear: 0,
        wMonth: 0,
        wDayOfWeek: 0,
        wDay: 0,
        wHour: 0,
        wMinute: 0,
        wSecond: 0,
        wMilliseconds: 0,
    };
    unsafe { GetLocalTime(&mut t) };
    take_name(t.wYear, t.wMonth, t.wDay, t.wHour, t.wMinute, t.wSecond)
}

fn take_name(y: u16, mo: u16, d: u16, h: u16, mi: u16, s: u16) -> String {
    format!("take_{y:04}-{mo:02}-{d:02}_{h:02}-{mi:02}-{s:02}")
}

fn mb_s(bytes_per_s: f64) -> String {
    format!("{:.1} MB/s", bytes_per_s / 1e6)
}

fn gb(bytes: u64) -> String {
    format!("{:.2} GB", bytes as f64 / 1e9)
}

fn minutes(min: f64) -> String {
    if min >= 120.0 {
        format!("{:.1} 小时", min / 60.0)
    } else {
        format!("{min:.0} 分钟")
    }
}

fn hms(d: Duration) -> String {
    let s = d.as_secs();
    format!("{:02}:{:02}:{:02}", s / 3600, (s / 60) % 60, s % 60)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state() -> RecordUiState {
        RecordUiState::default()
    }

    #[test]
    fn the_default_take_name_is_usable_as_a_directory_name() {
        let n = take_name(2026, 8, 19, 15, 30, 0);
        assert_eq!(n, "take_2026-08-19_15-30-00");
        assert!(!n.contains(':'));
        assert!(TakePlan::compute(
            TakeInputs {
                dir_text: "takes",
                name: &default_take_name(),
                fps: 30.0,
                frame: Some(crate::stream_rates::FrameSpec {
                    w: 4,
                    h: 2,
                    format: fs_core::PixelFormat::Bayer8,
                    bytes: 8,
                }),
                event: None,
                storage: FrameStorage::Rgb24Bin,
                ffmpeg_probe: None,
                geometry: fs_calib::GeomMode::Off,
                has_active_calib: false,
            },
            Some(1_000_000_000_000),
        )
        .refusal
        .is_none());
    }

    #[test]
    fn pending_clears_the_moment_the_worker_is_actually_recording() {
        let mut p = RecordPanelState::default();
        let now = Instant::now();
        p.pending = Some(Pending { since: now, prior_error: None });
        let st = RecordUiState { recording: true, ..state() };
        resolve_pending(&mut p, &st, now);
        assert!(p.pending.is_none());
        assert!(p.panel_error.is_none());
    }

    #[test]
    fn pending_clears_as_soon_as_the_worker_reports_a_new_failure() {
        let mut p = RecordPanelState::default();
        let now = Instant::now();
        p.pending = Some(Pending { since: now, prior_error: None });
        let st = RecordUiState { last_error: Some("事件录制未能启动".into()), ..state() };
        resolve_pending(&mut p, &st, now);
        assert!(p.pending.is_none());
        assert!(p.panel_error.is_none(), "worker 已经说明了原因,面板不该再叠一句自己的");
    }

    #[test]
    fn an_identical_repeat_failure_still_ends_at_the_timeout() {
        let mut p = RecordPanelState::default();
        let t0 = Instant::now();
        let err = Some("事件录制未能启动:相机线程已经退出".to_string());
        p.pending = Some(Pending { since: t0, prior_error: err.clone() });
        let st = RecordUiState { last_error: err, ..state() };
        resolve_pending(&mut p, &st, t0 + START_TIMEOUT - Duration::from_millis(1));
        assert!(p.pending.is_some(), "还没到时间就放弃会把一次正常的等待报成失败");
        resolve_pending(&mut p, &st, t0 + START_TIMEOUT);
        assert!(p.pending.is_none());
        assert!(p.panel_error.is_none(), "worker 已经给了原因,面板不该再编一句");
    }

    #[test]
    fn a_silent_worker_times_out_with_an_explicit_message() {
        let mut p = RecordPanelState::default();
        let t0 = Instant::now();
        p.pending = Some(Pending { since: t0, prior_error: None });
        resolve_pending(&mut p, &state(), t0 + START_TIMEOUT);
        assert!(p.pending.is_none());
        assert!(p.panel_error.unwrap().contains("重试"));
    }


    #[test]
    fn a_sent_start_hands_the_name_field_to_the_next_take() {
        let mut p = RecordPanelState::default();
        p.name = "my_custom_take".to_string();
        on_start_sent(&mut p, Instant::now(), None);
        assert_ne!(p.name, "my_custom_take", "上一个 take 的名字不能留给下一个");
        assert!(p.name.starts_with("take_"), "{}", p.name);
        assert!(p.pending.is_some());
        assert!(p.panel_error.is_none());
    }

    #[test]
    fn the_question_is_asked_once_per_take_and_again_after_the_next_start() {
        let mut p = RecordPanelState::default();
        let st = RecordUiState {
            sync_lost: true,
            take_dir: Some(std::path::PathBuf::from("C:\\takes\\take-001")),
            ..state()
        };
        let dir = pending_sync_lost_question(&p, &st).expect("失步后第一帧就该问");
        assert!(dir.ends_with("take-001"));
        p.sync_lost_resolved = true;
        assert!(pending_sync_lost_question(&p, &st).is_none(), "答过一次就不许再问");
        on_start_sent(&mut p, Instant::now(), None);
        assert!(
            pending_sync_lost_question(&p, &st).is_some(),
            "下一次 Start 之后,同一个目录的失步也必须重新问"
        );
    }


    #[test]
    fn no_question_without_a_sync_loss_or_without_a_take_dir() {
        let p = RecordPanelState::default();
        assert!(pending_sync_lost_question(&p, &state()).is_none());
        let st = RecordUiState { sync_lost: true, take_dir: None, ..state() };
        assert!(pending_sync_lost_question(&p, &st).is_none());
    }


    #[test]
    fn delete_removes_the_take_directory() {
        let dir = tempfile::tempdir().unwrap();
        let take = dir.path().join("take-001");
        std::fs::create_dir_all(&take).unwrap();
        std::fs::write(take.join("manifest.json"), b"{}").unwrap();
        std::fs::write(take.join("frames.bin"), b"data").unwrap();
        resolve_sync_lost(true, &take).unwrap();
        assert!(!take.exists(), "删除必须把整个 take 目录清掉");
    }

    #[test]
    fn keep_leaves_the_take_directory_intact_and_valid() {
        let dir = tempfile::tempdir().unwrap();
        let take = dir.path().join("take-001");
        std::fs::create_dir_all(&take).unwrap();
        std::fs::write(take.join("manifest.json"), b"{\"frames_written\":5}").unwrap();
        resolve_sync_lost(false, &take).unwrap();
        assert!(take.exists());
        let content = std::fs::read_to_string(take.join("manifest.json")).unwrap();
        assert_eq!(content, "{\"frames_written\":5}", "保留必须原样不动,manifest 依然完整可读");
    }

    #[test]
    fn durations_and_sizes_read_the_way_a_human_expects() {
        assert_eq!(hms(Duration::from_secs(0)), "00:00:00");
        assert_eq!(hms(Duration::from_secs(3725)), "01:02:05");
        assert_eq!(mb_s(39_321_600.0), "39.3 MB/s");
        assert_eq!(gb(3_300_000_000), "3.30 GB");
        assert_eq!(minutes(45.4), "45 分钟");
        assert_eq!(minutes(180.0), "3.0 小时");
    }
}
