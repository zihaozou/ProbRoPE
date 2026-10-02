
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crossbeam_channel::{bounded, Receiver, RecvTimeoutError, Sender, TryRecvError};
use fs_core::clock::ClockParams;
use fs_core::{EventBatch, SyncedFrame};
use fs_record::events::EventsBinWriter;
use fs_record::video::{ffmpeg_args, FrameStorage, VideoSink};
use fs_record::{FinalizeInfo, Manifest, TakeWriter, WriterOpts};

use crate::transform::EventRecordTap;

pub const RECORD_QUEUE_BYTES: u64 = 256 << 20;


const FRAME_BURST: usize = 8;

const EVENT_BURST: usize = 32;


const EVENT_CHANNEL_CAP: usize = 512;

const IDLE_POLL: Duration = Duration::from_millis(20);


const PUBLISH_INTERVAL: Duration = Duration::from_millis(500);

fn frame_bytes(f: &SyncedFrame) -> u64 {
    f.frame.data.len() as u64
}

struct QueuedFrame {
    frame: Arc<SyncedFrame>,
    bytes: u64,
    queued: Arc<AtomicU64>,
}

impl QueuedFrame {
    fn new(frame: Arc<SyncedFrame>, bytes: u64, queued: Arc<AtomicU64>) -> Self {
        queued.fetch_add(bytes, Ordering::AcqRel);
        QueuedFrame { frame, bytes, queued }
    }
}

impl Drop for QueuedFrame {
    fn drop(&mut self) {
        self.queued.fetch_sub(self.bytes, Ordering::AcqRel);
    }
}


pub struct RecordTx {
    tx: Sender<QueuedFrame>,

    rx: Receiver<QueuedFrame>,
    queued: Arc<AtomicU64>,
    dropped: Arc<AtomicU64>,
    cap_bytes: u64,
}


pub struct RecordRx {
    rx: Receiver<QueuedFrame>,
    queued: Arc<AtomicU64>,
    dropped: Arc<AtomicU64>,
}

pub fn record_channel(cap_bytes: u64) -> (RecordTx, RecordRx) {
    let (tx, rx) = crossbeam_channel::unbounded();
    let queued = Arc::new(AtomicU64::new(0));
    let dropped = Arc::new(AtomicU64::new(0));
    (
        RecordTx {
            tx,
            rx: rx.clone(),
            queued: Arc::clone(&queued),
            dropped: Arc::clone(&dropped),
            cap_bytes,
        },
        RecordRx { rx, queued, dropped },
    )
}

impl RecordTx {

    pub fn send(&self, frame: Arc<SyncedFrame>) {
        let bytes = frame_bytes(&frame);
        while self.queued.load(Ordering::Acquire) + bytes > self.cap_bytes {
            match self.rx.try_recv() {
                Ok(_oldest) => {
                    self.dropped.fetch_add(1, Ordering::Relaxed);
                }
                Err(_) => break,
            }
        }
        if self.tx.send(QueuedFrame::new(frame, bytes, Arc::clone(&self.queued))).is_err() {
            self.dropped.fetch_add(1, Ordering::Relaxed);
        }
    }
}

impl RecordRx {

    pub(crate) fn take(&self) -> Result<Arc<SyncedFrame>, TryRecvError> {
        self.rx.try_recv().map(|q| Arc::clone(&q.frame))
    }

    pub fn dropped(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }


    pub fn queued_bytes(&self) -> u64 {
        self.queued.load(Ordering::Acquire)
    }

    pub fn len(&self) -> usize {
        self.rx.len()
    }
}

pub enum StopReason {

    User,

    SyncLost,
    WriteError(String),
    Shutdown,
}

impl StopReason {
    fn tag(&self) -> &'static str {
        match self {
            StopReason::User => "user",
            StopReason::SyncLost => "sync_lost",
            StopReason::WriteError(_) => "write_error",
            StopReason::Shutdown => "shutdown",
        }
    }
}

pub enum RecordCmd {
    Start {
        dir: PathBuf,
        manifest: Manifest,
        opts: WriterOpts,

        calibration_json: Option<serde_json::Value>,
        storage: FrameStorage,

        ffmpeg: PathBuf,

        event_size: (u32, u32),

        geometry_op: serde_json::Value,
        calibration_source: Option<String>,
    },
    Stop { reason: StopReason },
}

#[derive(Clone, Debug, Default)]
pub struct RecordUiState {
    pub recording: bool,
    pub take_dir: Option<PathBuf>,
    pub frames_written: u64,
    pub dropped_frames: u64,

    pub bytes_written: u64,

    pub events_written: u64,

    pub events_dropped: u64,
    pub events_dropped_oob: u64,
    pub queued_bytes: u64,
    pub started_at: Option<Instant>,

    pub finalizing: bool,

    pub last_error: Option<String>,

    pub sync_lost: bool,
}

#[derive(Clone, Default)]
pub struct RecordTaps {

    pub tap: EventRecordTap,
    pub oob_dropped: Arc<AtomicU64>,
    pub frames_dropped_pre_transform: Arc<AtomicU64>,
}

type TakeOpener = Box<dyn FnMut(&Path, Manifest, WriterOpts, bool) -> std::io::Result<TakeWriter> + Send>;


struct ActiveTake {
    writer: TakeWriter,

    video: Option<VideoSink>,
    events: EventsBinWriter,
    events_rx: Receiver<EventBatch>,

    dropped_at_start: u64,
    pre_transform_at_start: u64,
    tap_dropped_at_start: u64,
    oob_at_start: u64,
}

impl ActiveTake {
    fn push_frame(&mut self, f: &SyncedFrame) -> Result<(), String> {
        match self.video.as_mut() {
            None => self.writer.write_frame(f).map_err(|e| write_failed(&e)),
            Some(sink) => {
                self.writer.write_sync_line(f).map_err(|e| format!("写 sync.jsonl 失败:{e}"))?;
                sink.push_frame(&f.frame.data).map_err(|e| format!("写 frames.mkv 失败:{e}"))
            }
        }
    }

    fn push_events(&mut self, batch: &EventBatch) -> Result<(), String> {
        for e in &batch.events {
            self.events
                .push(e.t_us, e.x, e.y, e.p as u8)
                .map_err(|err| format!("写 events.bin 失败:{err}"))?;
        }
        Ok(())
    }


    fn frame_bytes_written(&self) -> u64 {
        match &self.video {
            None => self.writer.bytes_written(),
            Some(sink) => sink.bytes_in(),
        }
    }
}


struct Worker {
    frames: RecordRx,
    state: Arc<Mutex<RecordUiState>>,
    open: TakeOpener,
    clock: Arc<Mutex<Option<ClockParams>>>,

    taps: RecordTaps,
    take: Option<ActiveTake>,

    sources_gone: bool,
    cur: RecordUiState,
    last_publish: Instant,
}

pub fn run(
    cmds: Receiver<RecordCmd>,
    frames: RecordRx,
    state: Arc<Mutex<RecordUiState>>,
    clock: Arc<Mutex<Option<ClockParams>>>,
    taps: RecordTaps,
) {
    Worker::new(frames, state, Box::new(TakeWriter::create), clock, taps).run(cmds);
}

impl Worker {
    fn new(
        frames: RecordRx,
        state: Arc<Mutex<RecordUiState>>,
        open: TakeOpener,
        clock: Arc<Mutex<Option<ClockParams>>>,
        taps: RecordTaps,
    ) -> Self {
        Worker {
            frames,
            state,
            open,
            clock,
            taps,
            take: None,
            sources_gone: false,
            cur: RecordUiState::default(),
            last_publish: Instant::now(),
        }
    }

    fn run(mut self, cmds: Receiver<RecordCmd>) {
        let reason = self.pump(&cmds);
        self.finish_take(reason);
        self.publish();
    }

    fn pump(&mut self, cmds: &Receiver<RecordCmd>) -> StopReason {
        loop {
            loop {
                match cmds.try_recv() {
                    Ok(cmd) => self.handle(cmd),
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => return StopReason::Shutdown,
                }
            }

            let frames_drained = self.drain_frames(FRAME_BURST);
            let events_drained = self.drain_events(EVENT_BURST);
            if self.sources_gone && self.take.is_some() {
                self.cur.last_error = Some(FRAMES_GONE.into());
                self.finish_take(StopReason::Shutdown);
            }
            if self.cur.recording {
                self.publish_if_due();
            }

            if frames_drained < FRAME_BURST && events_drained < EVENT_BURST {
                match cmds.recv_timeout(IDLE_POLL) {
                    Ok(cmd) => self.handle(cmd),
                    Err(RecvTimeoutError::Timeout) => {}
                    Err(RecvTimeoutError::Disconnected) => return StopReason::Shutdown,
                }
            }
        }
    }

    fn handle(&mut self, cmd: RecordCmd) {
        match cmd {
            RecordCmd::Start {
                dir,
                manifest,
                opts,
                calibration_json,
                storage,
                ffmpeg,
                event_size,
                geometry_op,
                calibration_source,
            } => self.start(dir, manifest, opts, calibration_json, storage, ffmpeg, event_size, geometry_op, calibration_source),
            RecordCmd::Stop { reason } => self.finish_take(reason),
        }
    }

    fn start(
        &mut self,
        dir: PathBuf,
        manifest: Manifest,
        opts: WriterOpts,
        calibration_json: Option<serde_json::Value>,
        storage: FrameStorage,
        ffmpeg: PathBuf,
        event_size: (u32, u32),
        geometry_op: serde_json::Value,
        calibration_source: Option<String>,
    ) {
        if self.take.is_some() {
            self.finish_take(StopReason::User);
        }
        if self.sources_gone {
            self.cur = RecordUiState { last_error: Some(FRAMES_GONE.into()), ..RecordUiState::default() };
            self.publish();
            return;
        }
        for _ in 0..self.frames.len() {
            if self.frames.take().is_err() {
                break;
            }
        }

        let dir_created_by_us = !dir.exists();

        match self.open_take(&dir, manifest, opts, calibration_json, &storage, &ffmpeg, event_size, geometry_op, calibration_source) {
            Ok(t) => {
                self.take = Some(t);
                self.cur = RecordUiState {
                    recording: true,
                    take_dir: Some(dir),
                    started_at: Some(Instant::now()),
                    ..RecordUiState::default()
                };
            }
            Err(msg) => {
                let msg = if dir_created_by_us && dir.exists() {
                    match std::fs::remove_dir_all(&dir) {
                        Ok(()) => msg,
                        Err(e) => format!("{msg};清理半开 take 目录失败({e}),请手动删除 {}", dir.display()),
                    }
                } else {
                    msg
                };
                eprintln!("record: {msg}");
                self.cur = RecordUiState { last_error: Some(msg), ..RecordUiState::default() };
            }
        }
        self.publish();
    }

    fn open_take(
        &mut self,
        dir: &Path,
        mut manifest: Manifest,
        opts: WriterOpts,
        calibration_json: Option<serde_json::Value>,
        storage: &FrameStorage,
        ffmpeg: &Path,
        event_size: (u32, u32),
        geometry_op: serde_json::Value,
        calibration_source: Option<String>,
    ) -> Result<ActiveTake, String> {
        let frame_size = manifest.frame_size;
        let fps = manifest.fps;
        let video_out = dir.join("frames.mkv");
        manifest.frame_storage = storage.manifest_name().to_string();
        manifest.encoder_args = match storage {
            FrameStorage::Rgb24Bin => None,
            s => Some(ffmpeg_args(s, frame_size, fps, &video_out).join(" ")),
        };
        manifest.geometry_op = geometry_op;
        manifest.calibration_source = calibration_source;

        let with_frames = matches!(storage, FrameStorage::Rgb24Bin);
        let writer = (self.open)(dir, manifest, opts, with_frames)
            .map_err(|e| format!("无法创建 take({}):{e}", dir.display()))?;

        if let Some(doc) = calibration_json {
            let path = dir.join("calibration.json");
            std::fs::write(&path, serde_json::to_string_pretty(&doc).expect("json serialize"))
                .map_err(|e| format!("无法写入标定文档({}):{e}", path.display()))?;
        }

        let events = EventsBinWriter::create(&dir.join("events.bin"), event_size)
            .map_err(|e| format!("无法创建 events.bin:{e}"))?;

        let (ev_tx, ev_rx) = bounded(EVENT_CHANNEL_CAP);
        self.taps.tap.arm(ev_tx);

        let video = match storage {
            FrameStorage::Rgb24Bin => None,
            s => match VideoSink::create(ffmpeg, s, frame_size, fps, &video_out) {
                Ok(v) => Some(v),
                Err(e) => {
                    self.taps.tap.disarm();
                    return Err(format!("无法启动视频编码器:{e}"));
                }
            },
        };

        Ok(ActiveTake {
            writer,
            video,
            events,
            events_rx: ev_rx,
            dropped_at_start: self.frames.dropped(),
            pre_transform_at_start: self.taps.frames_dropped_pre_transform.load(Ordering::Relaxed),
            tap_dropped_at_start: self.taps.tap.dropped.load(Ordering::Relaxed),
            oob_at_start: self.taps.oob_dropped.load(Ordering::Relaxed),
        })
    }


    fn drain_frames(&mut self, burst: usize) -> usize {
        let mut n = 0;
        while n < burst {
            let f = match self.frames.take() {
                Ok(f) => f,
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    self.sources_gone = true;
                    break;
                }
            };
            n += 1;
            let res = self.take.as_mut().map(|t| t.push_frame(&f));
            if let Some(Err(e)) = res {
                self.finish_take(StopReason::WriteError(e));
            }
        }
        n
    }


    fn drain_events(&mut self, burst: usize) -> usize {
        let mut n = 0;
        while n < burst {
            let Some(t) = self.take.as_mut() else { break };
            let batch = match t.events_rx.try_recv() {
                Ok(b) => b,
                Err(_) => break,
            };
            n += 1;
            if let Err(e) = t.push_events(&batch) {
                self.finish_take(StopReason::WriteError(e));
            }
        }
        n
    }


    fn finish_take(&mut self, reason: StopReason) {
        let Some(mut t) = self.take.take() else { return };
        self.taps.tap.disarm();

        let tag = reason.tag();
        let sync_lost = matches!(reason, StopReason::SyncLost);
        let mut write_error = matches!(reason, StopReason::WriteError(_));
        let mut err = match reason {
            StopReason::WriteError(e) => Some(e),
            _ => None,
        };

        if !write_error {
            for _ in 0..self.frames.len() {
                let Ok(f) = self.frames.take() else { break };
                if let Err(e) = t.push_frame(&f) {
                    err = Some(e);
                    write_error = true;
                    break;
                }
            }
        }
        if !write_error {
            while let Ok(batch) = t.events_rx.try_recv() {
                if let Err(e) = t.push_events(&batch) {
                    err = Some(e);
                    write_error = true;
                    break;
                }
            }
        }
        let mut discarded_at_stop = 0u64;
        while let Ok(b) = t.events_rx.try_recv() {
            discarded_at_stop += b.events.len() as u64;
        }

        let late = t.events.late_dropped();
        let events_written = t.events.events_written();
        let tap_dropped =
            self.taps.tap.dropped.load(Ordering::Relaxed).saturating_sub(t.tap_dropped_at_start);
        let events_dropped = tap_dropped + late + discarded_at_stop;
        let events_dropped_oob =
            self.taps.oob_dropped.load(Ordering::Relaxed).saturating_sub(t.oob_at_start);
        let dropped_frames = self.frames.dropped().saturating_sub(t.dropped_at_start)
            + self
                .taps
                .frames_dropped_pre_transform
                .load(Ordering::Relaxed)
                .saturating_sub(t.pre_transform_at_start);

        self.cur.frames_written = t.writer.frames_written;
        self.cur.bytes_written = t.frame_bytes_written();
        self.cur.dropped_frames = dropped_frames;
        self.cur.events_written = events_written;
        self.cur.events_dropped = events_dropped;
        self.cur.events_dropped_oob = events_dropped_oob;
        self.cur.queued_bytes = 0;
        self.cur.recording = false;
        self.cur.sync_lost = sync_lost;

        if late > 0 {
            let msg = format!("{late} 条迟到事件(t < t0)被丢弃");
            err = Some(match err {
                Some(first) => format!("{first};{msg}"),
                None => msg,
            });
        }

        let ActiveTake { writer, mut video, events, events_rx, .. } = t;
        drop(events_rx);

        if let Err(e) = events.finalize() {
            let msg = format!("events.bin 收尾失败:{e}");
            err = Some(match err {
                Some(first) => format!("{first};{msg}"),
                None => msg,
            });
        }

        if let Some(sink) = video.take() {
            if write_error {
                drop(sink);
            } else {
                self.cur.finalizing = true;
                self.publish();
                let res = sink.finalize();
                self.cur.finalizing = false;
                if let Err(e) = res {
                    let msg = format!("视频收尾失败:{e}");
                    err = Some(match err {
                        Some(first) => format!("{first};{msg}"),
                        None => msg,
                    });
                }
            }
        }

        let clock_model = *self.clock.lock().unwrap_or_else(|p| p.into_inner());

        let stats = serde_json::json!({ "stop_reason": tag });
        let info = FinalizeInfo {
            stopped_unix_ms: fs_record::unix_ms_now(),
            final_stats: stats,
            events_written,
            events_dropped,
            events_dropped_oob,
            clock_model,
            dropped_frames,
            sync_lost,
        };
        if let Err(e) = writer.finalize(info) {
            let msg = format!("收尾失败:{e}");
            err = Some(match err {
                Some(first) => format!("{first};{msg}"),
                None => msg,
            });
        }
        if let Some(e) = err {
            eprintln!("record: {e}");
            self.cur.last_error = Some(e);
        }
        self.publish();
    }

    fn publish_if_due(&mut self) {
        if self.last_publish.elapsed() >= PUBLISH_INTERVAL {
            self.publish();
        }
    }

    fn publish(&mut self) {
        if let Some(t) = &self.take {
            self.cur.frames_written = t.writer.frames_written;
            self.cur.bytes_written = t.frame_bytes_written();
            self.cur.dropped_frames = self.frames.dropped().saturating_sub(t.dropped_at_start)
                + self
                    .taps
                    .frames_dropped_pre_transform
                    .load(Ordering::Relaxed)
                    .saturating_sub(t.pre_transform_at_start);
            self.cur.queued_bytes = self.frames.queued_bytes();
            self.cur.events_written = t.events.events_written();
            self.cur.events_dropped = self
                .taps
                .tap
                .dropped
                .load(Ordering::Relaxed)
                .saturating_sub(t.tap_dropped_at_start)
                + t.events.late_dropped();
            self.cur.events_dropped_oob =
                self.taps.oob_dropped.load(Ordering::Relaxed).saturating_sub(t.oob_at_start);
        }
        let snapshot = self.cur.clone();
        *self.state.lock().unwrap_or_else(|p| p.into_inner()) = snapshot;
        self.last_publish = Instant::now();
    }
}


const FRAMES_GONE: &str = "帧通道已断开:采集已停止";

fn write_failed(e: &std::io::Error) -> String {
    format!("写 frames.bin 失败:{e}")
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::io::Write;
    use std::sync::atomic::AtomicUsize;

    use crossbeam_channel::bounded;
    use fs_core::{Event, PixelFormat, RgbFrame, SyncSource};
    use fs_record::events::read_events_bin;
    use fs_record::FrameSink;

    fn synced(seq: u64, bytes: usize) -> Arc<SyncedFrame> {
        Arc::new(SyncedFrame {
            frame: RgbFrame {
                seq,
                t_cam_us: seq as i64 * 33_333,
                w: 4,
                h: 2,
                data: vec![seq as u8; bytes],
                format: PixelFormat::Rgb8,
            },
            t_evk_us: seq as i64 * 33_333 + 3000,
            source: SyncSource::Matched,
        })
    }

    fn manifest() -> Manifest {
        Manifest {
            version: fs_record::MANIFEST_VERSION,
            take_name: "t".into(),
            created_unix_ms: 0,
            started_unix_ms: 0,
            software_version: fs_record::SOFTWARE_VERSION.into(),
            fps: 30.0,
            exposure_us: 5000,
            frame_size: (4, 2),
            frame_format: "Rgb8".into(),
            frame_storage: "rgb24_bin".into(),
            encoder_args: None,
            geometry_op: serde_json::json!({"mode": "off"}),
            calibration_source: None,
            flir: serde_json::json!({}),
            evk: serde_json::json!({}),
            board: serde_json::json!(null),
        }
    }

    fn start_cmd(dir: &Path) -> RecordCmd {
        RecordCmd::Start {
            dir: dir.to_path_buf(),
            manifest: manifest(),
            opts: WriterOpts::default(),
            calibration_json: None,
            storage: FrameStorage::Rgb24Bin,
            ffmpeg: PathBuf::from("ffmpeg"),
            event_size: (1280, 720),
            geometry_op: serde_json::json!({"mode": "off"}),
            calibration_source: None,
        }
    }


    fn batch(evs: &[(i64, u16, u16, i8)]) -> EventBatch {
        EventBatch { events: evs.iter().map(|&(t_us, x, y, p)| Event { t_us, x, y, p }).collect() }
    }


    fn no_clock() -> Arc<Mutex<Option<ClockParams>>> {
        Arc::new(Mutex::new(None))
    }

    fn wait_until(state: &Arc<Mutex<RecordUiState>>, what: &str, f: impl Fn(&RecordUiState) -> bool) {
        let deadline = Instant::now() + Duration::from_secs(5);
        while Instant::now() < deadline {
            if f(&state.lock().unwrap()) {
                return;
            }
            std::thread::sleep(Duration::from_millis(2));
        }
        panic!("等了 5 秒也没等到:{what}");
    }

    fn read_manifest(dir: &Path) -> serde_json::Value {
        serde_json::from_str(&std::fs::read_to_string(dir.join("manifest.json")).unwrap()).unwrap()
    }

    #[test]
    fn writes_frames_and_finalizes() {
        let dir = tempfile::tempdir().unwrap();
        let take_dir = dir.path().join("take-001");
        let (tx, rx) = record_channel(RECORD_QUEUE_BYTES);
        let state = Arc::new(Mutex::new(RecordUiState::default()));
        let (cmd_tx, cmd_rx) = bounded(8);
        let h = {
            let state = state.clone();
            std::thread::spawn(move || run(cmd_rx, rx, state, no_clock(), RecordTaps::default()))
        };

        cmd_tx.send(start_cmd(&take_dir)).unwrap();
        wait_until(&state, "录制开始", |s| s.recording);

        for i in 0..5 {
            tx.send(synced(i, 8));
        }
        cmd_tx.send(RecordCmd::Stop { reason: StopReason::User }).unwrap();
        drop(cmd_tx);
        h.join().unwrap();

        let s = state.lock().unwrap().clone();
        assert!(!s.recording);
        assert_eq!(s.frames_written, 5);
        assert_eq!(s.bytes_written, 40);
        assert_eq!(s.dropped_frames, 0);
        assert_eq!(s.last_error, None, "正常路径不该有错误");
        assert!(!s.sync_lost);
        assert!(!s.finalizing, "收尾结束后 finalizing 必须已清");
        assert_eq!(s.take_dir.as_deref(), Some(take_dir.as_path()));

        assert_eq!(std::fs::metadata(take_dir.join("frames.bin")).unwrap().len(), 40);
        assert_eq!(std::fs::metadata(take_dir.join("frames.idx")).unwrap().len(), 5 * 32);
        let sync = std::fs::read_to_string(take_dir.join("sync.jsonl")).unwrap();
        assert_eq!(sync.lines().count(), 5);
        assert_eq!(
            std::fs::metadata(take_dir.join("events.bin")).unwrap().len(),
            24,
            "零事件 take 的 events.bin 是纯头(spec §4.3),不是缺文件"
        );
        let m = read_manifest(&take_dir);
        assert_eq!(m["frames_written"], 5);
        assert_eq!(m["final_stats"]["stop_reason"], "user");
        assert!(!take_dir.join("events.raw").exists(), "SDK raw logging 已退役,不该再产出 events.raw");
        assert!(m.get("events_time_shift_us").is_none(), "v3 里不该再有 events_time_shift_us");
        assert_eq!(m["events_written"], 0);
        assert_eq!(m["events_dropped"], 0);
        assert_eq!(m["events_dropped_oob"], 0);
    }


    #[test]
    fn unified_take_rgb24_with_events() {
        let dir = tempfile::tempdir().unwrap();
        let take_dir = dir.path().join("take-001");
        let (tx, rx) = record_channel(RECORD_QUEUE_BYTES);
        let state = Arc::new(Mutex::new(RecordUiState::default()));
        let taps = RecordTaps::default();
        let (cmd_tx, cmd_rx) = bounded(8);
        let h = {
            let state = state.clone();
            let taps = taps.clone();
            std::thread::spawn(move || run(cmd_rx, rx, state, no_clock(), taps))
        };

        cmd_tx.send(start_cmd(&take_dir)).unwrap();
        wait_until(&state, "录制开始", |s| s.recording);
        assert!(taps.tap.is_armed(), "开录之后 tap 必须已挂上,否则事件路根本没接");

        for i in 0..3 {
            tx.send(synced(i, 8));
        }
        let fed: Vec<(i64, u16, u16, i8)> =
            vec![(1_000, 5, 7, 1), (1_001, 1279, 0, 0), (2_000, 0, 719, 1), (2_500, 42, 3, 0)];
        taps.tap.offer(&batch(&fed[..2]));
        taps.tap.offer(&batch(&fed[2..]));

        cmd_tx.send(RecordCmd::Stop { reason: StopReason::User }).unwrap();
        drop(cmd_tx);
        h.join().unwrap();

        assert!(!taps.tap.is_armed(), "收尾必须摘下 tap,否则回调会一直对着死通道计丢弃");

        let s = state.lock().unwrap().clone();
        assert_eq!(s.frames_written, 3);
        assert_eq!(s.events_written, 4);
        assert_eq!(s.events_dropped, 0);
        assert_eq!(s.last_error, None, "正常路径不该有错误");

        assert!(take_dir.join("frames.bin").exists());
        assert!(take_dir.join("frames.idx").exists());
        assert_eq!(std::fs::read_to_string(take_dir.join("sync.jsonl")).unwrap().lines().count(), 3);

        let f = read_events_bin(&take_dir.join("events.bin")).unwrap();
        assert_eq!((f.width, f.height), (1280, 720), "头部画幅 = Start 命令给的事件几何");
        assert_eq!(f.t0_us, 1_000, "t0 = 首个写入事件的绝对 t_evk_us");
        let want: Vec<(i64, u16, u16, u8)> = fed.iter().map(|&(t, x, y, p)| (t, x, y, p as u8)).collect();
        assert_eq!(f.events, want, "events.bin 解回必须与喂入逐条一致");

        let m = read_manifest(&take_dir);
        assert_eq!(m["version"], 3);
        assert_eq!(m["events_written"], 4);
        assert_eq!(m["events_dropped"], 0);
        assert_eq!(m["events_dropped_oob"], 0);
        assert_eq!(m["frame_storage"], "rgb24_bin");
        assert!(m["encoder_args"].is_null(), "裸 RGB 档没有编码器");
    }

    #[test]
    fn per_take_event_drop_counters_are_snapshot_diffs() {
        let dir = tempfile::tempdir().unwrap();
        let take_dir = dir.path().join("take-001");
        let (tx, rx) = record_channel(RECORD_QUEUE_BYTES);
        let state = Arc::new(Mutex::new(RecordUiState::default()));
        let taps = RecordTaps::default();
        taps.oob_dropped.store(1_000, Ordering::Relaxed);
        taps.tap.dropped.store(500, Ordering::Relaxed);
        taps.frames_dropped_pre_transform.store(77, Ordering::Relaxed);
        let (cmd_tx, cmd_rx) = bounded(8);
        let h = {
            let state = state.clone();
            let taps = taps.clone();
            std::thread::spawn(move || run(cmd_rx, rx, state, no_clock(), taps))
        };

        cmd_tx.send(start_cmd(&take_dir)).unwrap();
        wait_until(&state, "录制开始", |s| s.recording);

        tx.send(synced(0, 8));
        taps.tap.offer(&batch(&[(1_000, 1, 1, 1), (1_200, 2, 2, 0), (900, 3, 3, 1)]));
        taps.oob_dropped.fetch_add(3, Ordering::Relaxed);
        taps.tap.dropped.fetch_add(9, Ordering::Relaxed);
        taps.frames_dropped_pre_transform.fetch_add(2, Ordering::Relaxed);

        wait_until(&state, "录制中发布 events_dropped_oob 快照差", |s| {
            s.recording && s.events_dropped_oob == 3
        });

        cmd_tx.send(RecordCmd::Stop { reason: StopReason::User }).unwrap();
        drop(cmd_tx);
        h.join().unwrap();

        let m = read_manifest(&take_dir);
        assert_eq!(m["events_written"], 2);
        assert_eq!(m["events_dropped_oob"], 3, "必须是快照差,不是累计 1003");
        assert_eq!(m["events_dropped"], 10, "tap 差值 9 + 迟到 1;不是累计 509");
        assert_eq!(m["dropped_frames"], 2, "变换级进料口的丢帧要折进丢帧口径(快照差,不是 79)");
        assert_eq!(m["frames_written"], 1);

        let s = state.lock().unwrap().clone();
        assert_eq!(s.events_dropped, 10);
        assert_eq!(s.events_dropped_oob, 3, "收尾后发布的越界数与 manifest 同口径(快照差)");
        let err = s.last_error.expect("迟到丢弃非零必须在 last_error 点名");
        assert!(err.contains("迟到"), "{err}");
    }

    #[test]
    fn video_storage_with_missing_ffmpeg_fails_start_loudly() {
        let dir = tempfile::tempdir().unwrap();
        let bad = dir.path().join("take-video");
        let good = dir.path().join("take-ok");
        let missing = dir.path().join("no-such-ffmpeg.exe");
        let (tx, rx) = record_channel(RECORD_QUEUE_BYTES);
        let state = Arc::new(Mutex::new(RecordUiState::default()));
        let taps = RecordTaps::default();
        let (cmd_tx, cmd_rx) = bounded(8);
        let h = {
            let state = state.clone();
            let taps = taps.clone();
            std::thread::spawn(move || run(cmd_rx, rx, state, no_clock(), taps))
        };

        cmd_tx
            .send(RecordCmd::Start {
                dir: bad.clone(),
                manifest: manifest(),
                opts: WriterOpts::default(),
                calibration_json: None,
                storage: FrameStorage::HevcLossy { cq: 19 },
                ffmpeg: missing.clone(),
                event_size: (1280, 720),
                geometry_op: serde_json::json!({"mode": "off"}),
                calibration_source: None,
            })
            .unwrap();
        wait_until(&state, "启动失败被发布出来", |s| s.last_error.is_some());

        let s = state.lock().unwrap().clone();
        assert!(!s.recording, "编码器起不来还进录制态,就是最糟的那种静默");
        let err = s.last_error.clone().unwrap();
        assert!(err.contains(&missing.display().to_string()), "错误必须点名 ffmpeg 路径:{err}");
        assert!(!bad.exists(), "失败的 Start 不许在盘上留半开 take 目录");
        assert!(!taps.tap.is_armed(), "失败的 Start 必须卸掉已挂的 tap —— 留着会污染下一次 take 的快照差");

        cmd_tx.send(start_cmd(&good)).unwrap();
        wait_until(&state, "第二个 take 开始", |s| s.recording);
        assert_eq!(state.lock().unwrap().last_error, None, "新 take 必须清掉上一次的错误");
        for i in 0..2 {
            tx.send(synced(i, 8));
        }
        cmd_tx.send(RecordCmd::Stop { reason: StopReason::User }).unwrap();
        drop(cmd_tx);
        h.join().unwrap();
        assert_eq!(std::fs::metadata(good.join("frames.bin")).unwrap().len(), 16);
        assert_eq!(state.lock().unwrap().last_error, None);
    }


    #[test]
    fn video_take_writes_sync_rows_and_no_frame_files() {
        let dir = tempfile::tempdir().unwrap();
        let take_dir = dir.path().join("take-video");
        let fake = dir.path().join("fake-ffmpeg.cmd");
        std::fs::write(
            &fake,
            "@powershell -NoProfile -Command \"[Console]::OpenStandardInput().CopyTo([System.IO.Stream]::Null)\"\r\n",
        )
        .unwrap();

        let (tx, rx) = record_channel(RECORD_QUEUE_BYTES);
        let state = Arc::new(Mutex::new(RecordUiState::default()));
        let (cmd_tx, cmd_rx) = bounded(8);
        let h = {
            let state = state.clone();
            std::thread::spawn(move || run(cmd_rx, rx, state, no_clock(), RecordTaps::default()))
        };

        cmd_tx
            .send(RecordCmd::Start {
                dir: take_dir.clone(),
                manifest: manifest(),
                opts: WriterOpts::default(),
                calibration_json: None,
                storage: FrameStorage::HevcLossless,
                ffmpeg: fake,
                event_size: (1280, 720),
                geometry_op: serde_json::json!({"mode": "off"}),
                calibration_source: None,
            })
            .unwrap();
        wait_until(&state, "录制开始", |s| s.recording);
        for i in 0..2 {
            tx.send(synced(i, 24));
        }
        cmd_tx.send(RecordCmd::Stop { reason: StopReason::User }).unwrap();
        drop(cmd_tx);
        h.join().unwrap();

        let s = state.lock().unwrap().clone();
        assert_eq!(s.last_error, None, "健康编码器的正常收尾不该有错误");
        assert_eq!(s.frames_written, 2);
        assert_eq!(s.bytes_written, 48, "视频档的字节口径 = 喂进编码器管道的裸字节");
        assert!(!s.finalizing, "收尾结束后 finalizing 必须已清");

        assert!(!take_dir.join("frames.bin").exists(), "视频档不许有 frames.bin(空文件会被当零帧裸 take)");
        assert!(!take_dir.join("frames.idx").exists(), "视频档不许有 frames.idx —— 帧索引就是 sync.jsonl 行号");
        assert_eq!(
            std::fs::read_to_string(take_dir.join("sync.jsonl")).unwrap().lines().count(),
            2,
            "视频第 k 帧 == sync.jsonl 第 k 行"
        );
        assert!(take_dir.join("events.bin").exists());
        let m = read_manifest(&take_dir);
        assert_eq!(m["frame_storage"], "hevc_lossless");
        let args = m["encoder_args"].as_str().expect("视频档必须记下实际编码参数");
        assert!(args.contains("hevc_nvenc"), "{args}");
        assert!(args.contains("frames.mkv"), "{args}");
        assert_eq!(m["frames_written"], 2);
    }

    #[test]
    fn late_events_do_not_skip_the_encoder_exit_check() {
        let dir = tempfile::tempdir().unwrap();
        let take_dir = dir.path().join("take-video");
        let fake = dir.path().join("fake-ffmpeg.cmd");
        std::fs::write(
            &fake,
            "@powershell -NoProfile -Command \"[Console]::OpenStandardInput().CopyTo([System.IO.Stream]::Null)\"\r\n@exit /b 3\r\n",
        )
        .unwrap();

        let (tx, rx) = record_channel(RECORD_QUEUE_BYTES);
        let state = Arc::new(Mutex::new(RecordUiState::default()));
        let taps = RecordTaps::default();
        let (cmd_tx, cmd_rx) = bounded(8);
        let h = {
            let state = state.clone();
            let taps = taps.clone();
            std::thread::spawn(move || run(cmd_rx, rx, state, no_clock(), taps))
        };

        cmd_tx
            .send(RecordCmd::Start {
                dir: take_dir.clone(),
                manifest: manifest(),
                opts: WriterOpts::default(),
                calibration_json: None,
                storage: FrameStorage::HevcLossless,
                ffmpeg: fake,
                event_size: (1280, 720),
                geometry_op: serde_json::json!({"mode": "off"}),
                calibration_source: None,
            })
            .unwrap();
        wait_until(&state, "录制开始", |s| s.recording);
        tx.send(synced(0, 24));
        taps.tap.offer(&batch(&[(1_000, 1, 1, 1), (900, 2, 2, 0)]));
        cmd_tx.send(RecordCmd::Stop { reason: StopReason::User }).unwrap();
        drop(cmd_tx);
        h.join().unwrap();

        let s = state.lock().unwrap().clone();
        let err = s.last_error.expect("迟到提示与编码器退出码都必须浮出");
        let late = err.find("迟到").expect("迟到提示必须在(spec §4.3 的正常丢弃要可见)");
        let exit = err.find("视频收尾失败").expect("finalize 必须真的跑了 —— 非零退出码只有它能抓住");
        assert!(err.contains("退出码非零"), "{err}");
        assert!(late < exit, "根因在前:迟到提示先、视频收尾失败后:{err}");
        assert_eq!(s.events_dropped, 1, "迟到那一条计入丢弃");
        assert_eq!(s.events_written, 1);

        let m = read_manifest(&take_dir);
        assert_eq!(m["final_stats"]["stop_reason"], "user");
        assert_eq!(m["frames_written"], 1);
        assert_eq!(m["events_dropped"], 1);
        assert!(!take_dir.join("frames.bin").exists());
        assert_eq!(std::fs::read_to_string(take_dir.join("sync.jsonl")).unwrap().lines().count(), 1);
    }


    #[test]
    fn manifest_records_geometry_and_calibration_source() {
        let dir = tempfile::tempdir().unwrap();
        let take_a = dir.path().join("take-a");
        let take_b = dir.path().join("take-b");
        let (_tx, rx) = record_channel(RECORD_QUEUE_BYTES);
        let state = Arc::new(Mutex::new(RecordUiState::default()));
        let (cmd_tx, cmd_rx) = bounded(8);
        let h = {
            let state = state.clone();
            std::thread::spawn(move || run(cmd_rx, rx, state, no_clock(), RecordTaps::default()))
        };

        cmd_tx.send(start_cmd(&take_a)).unwrap();
        wait_until(&state, "take A 开始", |s| s.recording);
        cmd_tx.send(RecordCmd::Stop { reason: StopReason::User }).unwrap();
        wait_until(&state, "take A 收尾", |s| !s.recording);

        let geom = serde_json::json!({"mode": "undistort"});
        cmd_tx
            .send(RecordCmd::Start {
                dir: take_b.clone(),
                manifest: manifest(),
                opts: WriterOpts::default(),
                calibration_json: None,
                storage: FrameStorage::Rgb24Bin,
                ffmpeg: PathBuf::from("ffmpeg"),
                event_size: (1280, 720),
                geometry_op: geom.clone(),
                calibration_source: Some("loaded:C:\\calib\\stereo.json".into()),
            })
            .unwrap();
        wait_until(&state, "take B 开始", |s| s.recording);
        cmd_tx.send(RecordCmd::Stop { reason: StopReason::User }).unwrap();
        drop(cmd_tx);
        h.join().unwrap();

        let a = read_manifest(&take_a);
        assert_eq!(a["geometry_op"], serde_json::json!({"mode": "off"}));
        assert!(a["calibration_source"].is_null());

        let b = read_manifest(&take_b);
        assert_eq!(b["geometry_op"], geom, "geometry_op 必须逐字落盘");
        assert_eq!(b["calibration_source"], "loaded:C:\\calib\\stereo.json");
    }


    #[test]
    fn failed_start_leaves_no_half_open_take_and_recovers() {
        let dir = tempfile::tempdir().unwrap();
        let bad = dir.path().join("take-bad");
        let good = dir.path().join("take-ok");
        let opened = Arc::new(AtomicUsize::new(0));
        let opener: TakeOpener = {
            let opened = opened.clone();
            Box::new(move |d: &Path, m: Manifest, o: WriterOpts, wf: bool| {
                if opened.fetch_add(1, Ordering::SeqCst) == 0 {
                    std::fs::create_dir_all(d)?;
                    std::fs::write(d.join("manifest.json"), b"{}")?;
                    Err(std::io::Error::other("模拟:manifest 之后 frames.idx 打不开"))
                } else {
                    TakeWriter::create(d, m, o, wf)
                }
            })
        };

        let (tx, rx) = record_channel(RECORD_QUEUE_BYTES);
        let state = Arc::new(Mutex::new(RecordUiState::default()));
        let (cmd_tx, cmd_rx) = bounded(8);
        let h = {
            let state = state.clone();
            std::thread::spawn(move || {
                Worker::new(rx, state, opener, no_clock(), RecordTaps::default()).run(cmd_rx)
            })
        };

        cmd_tx.send(start_cmd(&bad)).unwrap();
        wait_until(&state, "启动失败被发布出来", |s| s.last_error.is_some());
        let s = state.lock().unwrap().clone();
        assert!(!s.recording);
        let err = s.last_error.unwrap();
        assert!(err.contains("无法创建 take"), "{err}");
        assert!(!bad.exists(), "半开的 take 目录必须被清掉 —— 留着会把用户下一次同名 Start 也挡死");

        cmd_tx.send(start_cmd(&good)).unwrap();
        wait_until(&state, "第二个 take 开始", |s| s.recording);
        assert_eq!(state.lock().unwrap().last_error, None, "新 take 必须清掉上一次的错误");
        tx.send(synced(0, 8));
        cmd_tx.send(RecordCmd::Stop { reason: StopReason::User }).unwrap();
        drop(cmd_tx);
        h.join().expect("worker 不许 panic");
        assert_eq!(std::fs::metadata(good.join("frames.bin")).unwrap().len(), 8);
    }


    #[test]
    fn finalize_failure_appends_after_the_root_cause() {

        struct DiskFullAndTruncateFails {
            budget: usize,
        }
        impl Write for DiskFullAndTruncateFails {
            fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
                if self.budget == 0 {
                    return Err(std::io::Error::other("There is not enough space on the disk. (os error 112)"));
                }
                let n = buf.len().min(self.budget);
                self.budget -= n;
                Ok(n)
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        impl FrameSink for DiskFullAndTruncateFails {
            fn set_len(&self, _: u64) -> std::io::Result<()> {
                Err(std::io::Error::other("模拟:收尾截断也失败"))
            }
        }

        let dir = tempfile::tempdir().unwrap();
        let take_dir = dir.path().join("take-full");
        std::fs::create_dir_all(&take_dir).unwrap();
        let opener: TakeOpener = Box::new(move |d: &Path, m: Manifest, o: WriterOpts, _wf: bool| {
            let o = WriterOpts { buf_bytes: 0, ..o };
            TakeWriter::with_frame_sink(d, m, o, Box::new(DiskFullAndTruncateFails { budget: 16 }))
        });

        let (tx, rx) = record_channel(RECORD_QUEUE_BYTES);
        let state = Arc::new(Mutex::new(RecordUiState::default()));
        let (cmd_tx, cmd_rx) = bounded(8);
        let h = {
            let state = state.clone();
            std::thread::spawn(move || {
                Worker::new(rx, state, opener, no_clock(), RecordTaps::default()).run(cmd_rx)
            })
        };

        cmd_tx.send(start_cmd(&take_dir)).unwrap();
        wait_until(&state, "录制开始", |s| s.recording);
        for i in 0..3 {
            tx.send(synced(i, 8));
        }
        wait_until(&state, "写盘失败被发布出来", |s| s.last_error.is_some());
        drop(cmd_tx);
        h.join().unwrap();

        let err = state.lock().unwrap().last_error.clone().unwrap();
        let root = err.find("not enough space").expect("根因(写失败)必须在错误里");
        let fin = err.find("收尾失败").expect("finalize 的失败也必须在错误里");
        assert!(root < fin, "根因必须在前,收尾失败追加在后:{err}");
        assert!(err.contains(';'), "两段之间用 `;` 连接:{err}");
    }

    #[test]
    fn start_writes_calibration_json_when_present() {
        let dir = tempfile::tempdir().unwrap();
        let take_dir = dir.path().join("take-001");
        let (_tx, rx) = record_channel(RECORD_QUEUE_BYTES);
        let state = Arc::new(Mutex::new(RecordUiState::default()));
        let (cmd_tx, cmd_rx) = bounded(8);
        let h = {
            let state = state.clone();
            std::thread::spawn(move || run(cmd_rx, rx, state, no_clock(), RecordTaps::default()))
        };

        let doc = serde_json::json!({"schema": "stereo_calibration.v1", "probe": 1});
        cmd_tx
            .send(RecordCmd::Start {
                dir: take_dir.clone(),
                manifest: manifest(),
                opts: WriterOpts::default(),
                calibration_json: Some(doc.clone()),
                storage: FrameStorage::Rgb24Bin,
                ffmpeg: PathBuf::from("ffmpeg"),
                event_size: (1280, 720),
                geometry_op: serde_json::json!({"mode": "off"}),
                calibration_source: Some("live_session".into()),
            })
            .unwrap();
        wait_until(&state, "录制开始", |s| s.recording);
        cmd_tx.send(RecordCmd::Stop { reason: StopReason::User }).unwrap();
        drop(cmd_tx);
        h.join().unwrap();

        let got: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(take_dir.join("calibration.json")).unwrap()).unwrap();
        assert_eq!(got, doc, "calibration.json 必须与传入的文档值级一致");
        assert_eq!(state.lock().unwrap().last_error, None, "正常路径不该有错误");
    }

    #[test]
    fn start_without_calibration_writes_no_file() {
        let dir = tempfile::tempdir().unwrap();
        let take_dir = dir.path().join("take-001");
        let (_tx, rx) = record_channel(RECORD_QUEUE_BYTES);
        let state = Arc::new(Mutex::new(RecordUiState::default()));
        let (cmd_tx, cmd_rx) = bounded(8);
        let h = {
            let state = state.clone();
            std::thread::spawn(move || run(cmd_rx, rx, state, no_clock(), RecordTaps::default()))
        };

        cmd_tx.send(start_cmd(&take_dir)).unwrap();
        wait_until(&state, "录制开始", |s| s.recording);
        cmd_tx.send(RecordCmd::Stop { reason: StopReason::User }).unwrap();
        drop(cmd_tx);
        h.join().unwrap();

        assert!(take_dir.join("manifest.json").exists(), "take 本身要正常落盘");
        assert!(!take_dir.join("calibration.json").exists(), "没有标定就不该有文件,占位文件比没有更糟");
    }


    #[test]
    fn sync_lost_stop_finalizes_with_sync_lost_true_and_raises_the_prompt_flag() {
        let dir = tempfile::tempdir().unwrap();
        let take_dir = dir.path().join("take-001");
        let (tx, rx) = record_channel(RECORD_QUEUE_BYTES);
        let state = Arc::new(Mutex::new(RecordUiState::default()));
        let taps = RecordTaps::default();
        let (cmd_tx, cmd_rx) = bounded(8);
        let h = {
            let state = state.clone();
            let taps = taps.clone();
            std::thread::spawn(move || run(cmd_rx, rx, state, no_clock(), taps))
        };

        cmd_tx.send(start_cmd(&take_dir)).unwrap();
        wait_until(&state, "录制开始", |s| s.recording);
        for i in 0..3 {
            tx.send(synced(i, 8));
        }
        taps.tap.offer(&batch(&[(5_000, 1, 2, 1)]));
        cmd_tx.send(RecordCmd::Stop { reason: StopReason::SyncLost }).unwrap();
        drop(cmd_tx);
        h.join().unwrap();

        let s = state.lock().unwrap().clone();
        assert!(!s.recording, "失步之后不能还停在录制态");
        assert!(s.sync_lost, "sync_lost 必须为 true —— 录制面板的删除/保留询问就挂在这个字段上");
        assert_eq!(s.frames_written, 3, "失步不是写盘失败,已经采到的帧要写完,不能因为中断就扔掉");
        assert_eq!(
            s.take_dir.as_deref(),
            Some(take_dir.as_path()),
            "take_dir 必须留着 —— 询问和后续的删除都要靠它找到这个目录"
        );

        let f = read_events_bin(&take_dir.join("events.bin")).unwrap();
        assert_eq!(f.events, vec![(5_000, 1, 2, 1)], "失步收尾同样要把已排队的事件写完");
        let m = read_manifest(&take_dir);
        assert_eq!(m["sync_lost"], true, "manifest 必须如实记录这次中断的原因,后处理要靠它判断尾部数据是否可信");
        assert_eq!(m["final_stats"]["stop_reason"], "sync_lost");
        assert_eq!(m["events_written"], 1);
    }


    #[test]
    fn full_queue_drops_and_counts_instead_of_blocking() {
        let (tx, rx) = record_channel(10 * 8);
        let (done_tx, done_rx) = bounded::<()>(1);
        std::thread::spawn(move || {
            for i in 0..15 {
                tx.send(synced(i, 8));
            }
            let _ = done_tx.send(());
        });
        done_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("send 阻塞了上游:录制通道绝不允许背压");

        assert_eq!(rx.dropped(), 5);
        assert_eq!(rx.queued_bytes(), 80, "水位必须停在上限,不能越过");
        let seqs: Vec<u64> = std::iter::from_fn(|| rx.take().ok().map(|f| f.frame.seq)).collect();
        assert_eq!(seqs, (5..15).collect::<Vec<u64>>(), "丢的必须是最旧的");
        assert_eq!(rx.queued_bytes(), 0, "取走之后字节数必须还回来");
    }


    #[test]
    fn queue_is_bounded_by_bytes_not_frame_count() {
        const CAP: u64 = 4 << 20;

        let (big_tx, big_rx) = record_channel(CAP);
        for i in 0..8 {
            big_tx.send(synced(i, 1 << 20));
        }
        assert_eq!(big_rx.len(), 4);
        assert_eq!(big_rx.dropped(), 4);

        let (small_tx, small_rx) = record_channel(CAP);
        for i in 0..8 {
            small_tx.send(synced(i, 1 << 10));
        }
        assert_eq!(small_rx.dropped(), 0);
        assert_eq!(small_rx.len(), 8);

        for i in 8..4096 {
            small_tx.send(synced(i, 1 << 10));
        }
        assert_eq!(small_rx.dropped(), 0);
        assert_eq!(small_rx.queued_bytes(), CAP);
        small_tx.send(synced(4096, 1 << 10));
        assert_eq!(small_rx.dropped(), 1);
        assert_eq!(small_rx.queued_bytes(), CAP, "溢出之后水位仍然不许越过上限");
    }


    #[test]
    fn write_error_stops_the_take_and_surfaces_the_reason() {

        struct DiskFull {
            budget: usize,
        }
        impl Write for DiskFull {
            fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
                if self.budget == 0 {
                    return Err(std::io::Error::other("There is not enough space on the disk. (os error 112)"));
                }
                let n = buf.len().min(self.budget);
                self.budget -= n;
                Ok(n)
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        impl FrameSink for DiskFull {
            fn set_len(&self, _: u64) -> std::io::Result<()> {
                Ok(())
            }
        }

        let dir = tempfile::tempdir().unwrap();
        let bad = dir.path().join("take-full");
        let good = dir.path().join("take-ok");
        std::fs::create_dir_all(&bad).unwrap();
        let opened = Arc::new(AtomicUsize::new(0));
        let opener: TakeOpener = {
            let opened = opened.clone();
            Box::new(move |d: &Path, m: Manifest, o: WriterOpts, wf: bool| {
                if opened.fetch_add(1, Ordering::SeqCst) == 0 {
                    let o = WriterOpts { buf_bytes: 0, ..o };
                    TakeWriter::with_frame_sink(d, m, o, Box::new(DiskFull { budget: 16 }))
                } else {
                    TakeWriter::create(d, m, o, wf)
                }
            })
        };

        let (tx, rx) = record_channel(RECORD_QUEUE_BYTES);
        let state = Arc::new(Mutex::new(RecordUiState::default()));
        let (cmd_tx, cmd_rx) = bounded(8);
        let h = {
            let state = state.clone();
            std::thread::spawn(move || {
                Worker::new(rx, state, opener, no_clock(), RecordTaps::default()).run(cmd_rx)
            })
        };

        cmd_tx.send(start_cmd(&bad)).unwrap();
        wait_until(&state, "录制开始", |s| s.recording);
        for i in 0..5 {
            tx.send(synced(i, 8));
        }

        wait_until(&state, "写盘失败被发布出来", |s| s.last_error.is_some());
        let s = state.lock().unwrap().clone();
        assert!(!s.recording, "写不动了还留在录制态,就是最糟的那种静默");
        let err = s.last_error.clone().unwrap();
        assert!(err.contains("not enough space"), "错误原因要能读:{err}");
        assert_eq!(s.frames_written, 2, "只有写成功的两帧算数");
        assert!(!s.sync_lost, "写盘失败不是失步,不该去问用户删不删");

        assert_eq!(std::fs::metadata(bad.join("frames.idx")).unwrap().len(), 2 * 32);
        assert_eq!(std::fs::read_to_string(bad.join("sync.jsonl")).unwrap().lines().count(), 2);
        let m = read_manifest(&bad);
        assert_eq!(m["frames_written"], 2);
        assert_eq!(m["final_stats"]["stop_reason"], "write_error");

        cmd_tx.send(start_cmd(&good)).unwrap();
        wait_until(&state, "第二个 take 开始", |s| s.recording);
        assert_eq!(state.lock().unwrap().last_error, None, "新 take 必须清掉上一次的错误");
        for i in 0..3 {
            tx.send(synced(i, 8));
        }
        cmd_tx.send(RecordCmd::Stop { reason: StopReason::User }).unwrap();
        drop(cmd_tx);
        h.join().expect("worker 不许 panic");

        assert_eq!(std::fs::metadata(good.join("frames.bin")).unwrap().len(), 24);
        let s = state.lock().unwrap().clone();
        assert_eq!(s.frames_written, 3);
        assert_eq!(s.last_error, None);
    }

    #[test]
    fn losing_the_frame_source_stops_the_take_and_says_why() {
        let dir = tempfile::tempdir().unwrap();
        let take_dir = dir.path().join("take-001");
        let (tx, rx) = record_channel(RECORD_QUEUE_BYTES);
        let state = Arc::new(Mutex::new(RecordUiState::default()));
        let (cmd_tx, cmd_rx) = bounded(8);
        let h = {
            let state = state.clone();
            std::thread::spawn(move || run(cmd_rx, rx, state, no_clock(), RecordTaps::default()))
        };

        cmd_tx.send(start_cmd(&take_dir)).unwrap();
        wait_until(&state, "录制开始", |s| s.recording);
        for i in 0..2 {
            tx.send(synced(i, 8));
        }
        drop(tx);

        wait_until(&state, "录制因采集停止而收尾", |s| !s.recording);
        let s = state.lock().unwrap().clone();
        assert!(s.last_error.unwrap().contains("采集已停止"));
        assert_eq!(s.frames_written, 2, "断开之前排队的帧仍然要写完");
        let m = read_manifest(&take_dir);
        assert_eq!(m["frames_written"], 2);

        let second = dir.path().join("take-002");
        cmd_tx.send(start_cmd(&second)).unwrap();
        drop(cmd_tx);
        h.join().unwrap();

        assert!(!second.exists(), "被挡下的录制不该在盘上留下半个 take 目录");
        let s = state.lock().unwrap().clone();
        assert!(!s.recording);
        assert_eq!(s.last_error.as_deref(), Some("帧通道已断开:采集已停止"));
    }

    #[test]
    fn dropping_the_command_channel_still_finalizes_the_take() {
        let dir = tempfile::tempdir().unwrap();
        let take_dir = dir.path().join("take-001");
        let (tx, rx) = record_channel(RECORD_QUEUE_BYTES);
        let state = Arc::new(Mutex::new(RecordUiState::default()));
        let (cmd_tx, cmd_rx) = bounded(8);
        let h = {
            let state = state.clone();
            std::thread::spawn(move || run(cmd_rx, rx, state, no_clock(), RecordTaps::default()))
        };

        let opts = WriterOpts { buf_bytes: 64 << 10, preallocate_bytes: 1 << 20 };
        cmd_tx
            .send(RecordCmd::Start {
                dir: take_dir.clone(),
                manifest: manifest(),
                opts,
                calibration_json: None,
                storage: FrameStorage::Rgb24Bin,
                ffmpeg: PathBuf::from("ffmpeg"),
                event_size: (1280, 720),
                geometry_op: serde_json::json!({"mode": "off"}),
                calibration_source: None,
            })
            .unwrap();
        wait_until(&state, "录制开始", |s| s.recording);
        for i in 0..4 {
            tx.send(synced(i, 8));
        }
        drop(cmd_tx);
        h.join().unwrap();

        assert_eq!(
            std::fs::metadata(take_dir.join("frames.bin")).unwrap().len(),
            32,
            "收尾没跑的话,预分配的尾巴还在"
        );
        let m = read_manifest(&take_dir);
        assert_eq!(m["frames_written"], 4);
        assert_eq!(m["final_stats"]["stop_reason"], "shutdown");
        assert!(!state.lock().unwrap().recording);
    }


    #[test]
    fn finish_take_reads_the_published_clock_model() {
        let dir = tempfile::tempdir().unwrap();
        let take_dir = dir.path().join("take-001");
        let (tx, rx) = record_channel(RECORD_QUEUE_BYTES);
        let state = Arc::new(Mutex::new(RecordUiState::default()));
        let (cmd_tx, cmd_rx) = bounded(8);
        let clock = Arc::new(Mutex::new(None));
        let h = {
            let state = state.clone();
            let clock = clock.clone();
            std::thread::spawn(move || run(cmd_rx, rx, state, clock, RecordTaps::default()))
        };

        cmd_tx.send(start_cmd(&take_dir)).unwrap();
        wait_until(&state, "录制开始", |s| s.recording);
        *clock.lock().unwrap() = Some(ClockParams { x0: 1_000, drift: 0.000_02, offset_us: 3_000.0 });
        for i in 0..2 {
            tx.send(synced(i, 8));
        }
        cmd_tx.send(RecordCmd::Stop { reason: StopReason::User }).unwrap();
        drop(cmd_tx);
        h.join().unwrap();

        let m = read_manifest(&take_dir);
        assert_eq!(m["clock_model"]["x0"], 1_000);
        assert_eq!(m["clock_model"]["drift"], 0.000_02);
        assert_eq!(m["clock_model"]["offset_us"], 3_000.0);
    }

    #[test]
    fn finish_take_writes_null_clock_model_when_never_seeded() {
        let dir = tempfile::tempdir().unwrap();
        let take_dir = dir.path().join("take-001");
        let (tx, rx) = record_channel(RECORD_QUEUE_BYTES);
        let state = Arc::new(Mutex::new(RecordUiState::default()));
        let (cmd_tx, cmd_rx) = bounded(8);
        let h = {
            let state = state.clone();
            std::thread::spawn(move || run(cmd_rx, rx, state, no_clock(), RecordTaps::default()))
        };

        cmd_tx.send(start_cmd(&take_dir)).unwrap();
        wait_until(&state, "录制开始", |s| s.recording);
        for i in 0..2 {
            tx.send(synced(i, 8));
        }
        cmd_tx.send(RecordCmd::Stop { reason: StopReason::User }).unwrap();
        drop(cmd_tx);
        h.join().unwrap();

        let m = read_manifest(&take_dir);
        assert!(m["clock_model"].is_null(), "从未 seed 过就不该编一个看着正常的假映射");
    }
}
