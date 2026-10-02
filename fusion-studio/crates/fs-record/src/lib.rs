pub mod events;
pub mod video;

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use fs_core::clock::ClockParams;
use fs_core::SyncedFrame;
use serde::Serialize;

pub const SOFTWARE_VERSION: &str = env!("FS_RECORD_SOFTWARE_VERSION");

pub const MANIFEST_VERSION: u32 = 3;

pub fn unix_ms_now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

#[derive(Serialize, Clone)]
pub struct Manifest {
    pub version: u32,
    pub take_name: String,
    pub created_unix_ms: i64,
    pub started_unix_ms: i64,
    pub software_version: String,
    pub fps: f64,
    pub exposure_us: i64,

    pub frame_size: (u32, u32),

    pub frame_format: String,

    pub frame_storage: String,

    pub encoder_args: Option<String>,
    pub geometry_op: serde_json::Value,
    pub calibration_source: Option<String>,
    pub flir: serde_json::Value,

    pub evk: serde_json::Value,

    pub board: serde_json::Value,
}

#[derive(Default)]
pub struct FinalizeInfo {
    pub stopped_unix_ms: i64,
    pub final_stats: serde_json::Value,

    pub events_written: u64,
    pub events_dropped: u64,

    pub events_dropped_oob: u64,
    pub clock_model: Option<ClockParams>,
    pub dropped_frames: u64,

    pub sync_lost: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct WriterOpts {

    pub buf_bytes: usize,
    pub preallocate_bytes: u64,
}

impl Default for WriterOpts {
    fn default() -> Self {
        WriterOpts { buf_bytes: 4 << 20, preallocate_bytes: 0 }
    }
}


pub trait FrameSink: Write + Send {

    fn set_len(&self, len: u64) -> std::io::Result<()>;
}

impl FrameSink for File {
    fn set_len(&self, len: u64) -> std::io::Result<()> { File::set_len(self, len) }
}


struct FrameFiles {
    bin: BufWriter<Box<dyn FrameSink>>,
    idx: BufWriter<File>,
    offset: u64,
}

pub struct TakeWriter {
    dir: PathBuf,
    manifest: Manifest,

    frames: Option<FrameFiles>,
    sync: BufWriter<File>,
    pub frames_written: u64,
}

impl TakeWriter {

    pub fn create(dir: &Path, manifest: Manifest, opts: WriterOpts, with_frames: bool) -> std::io::Result<Self> {
        std::fs::create_dir_all(dir)?;
        if with_frames {
            let frames = File::create(dir.join("frames.bin"))?;
            Self::with_frame_sink(dir, manifest, opts, Box::new(frames))
        } else {
            Self::build(dir, manifest, None)
        }
    }


    pub fn with_frame_sink(
        dir: &Path,
        manifest: Manifest,
        opts: WriterOpts,
        frames: Box<dyn FrameSink>,
    ) -> std::io::Result<Self> {
        if opts.preallocate_bytes > 0 { frames.set_len(opts.preallocate_bytes)?; }
        let files = FrameFiles {
            bin: BufWriter::with_capacity(opts.buf_bytes, frames),
            idx: BufWriter::new(File::create(dir.join("frames.idx"))?),
            offset: 0,
        };
        Self::build(dir, manifest, Some(files))
    }

    fn build(dir: &Path, manifest: Manifest, frames: Option<FrameFiles>) -> std::io::Result<Self> {
        std::fs::write(dir.join("manifest.json"), serde_json::to_string_pretty(&manifest)?)?;
        Ok(Self {
            dir: dir.to_path_buf(),
            manifest,
            frames,
            sync: BufWriter::new(File::create(dir.join("sync.jsonl"))?),
            frames_written: 0,
        })
    }

    pub fn bytes_written(&self) -> u64 { self.frames.as_ref().map_or(0, |f| f.offset) }

    pub fn write_frame(&mut self, s: &SyncedFrame) -> std::io::Result<()> {
        let f = &s.frame;
        let Some(fr) = self.frames.as_mut() else {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "视频档的 take 没有 frames.bin —— 帧字节该走 VideoSink,这里只写 write_sync_line",
            ));
        };
        fr.bin.write_all(&f.data)?;
        fr.idx.write_all(&f.seq.to_le_bytes())?;
        fr.idx.write_all(&f.t_cam_us.to_le_bytes())?;
        fr.idx.write_all(&fr.offset.to_le_bytes())?;
        fr.idx.write_all(&(f.data.len() as u64).to_le_bytes())?;
        write_sync_row(&mut self.sync, s)?;
        fr.offset += f.data.len() as u64;
        self.frames_written += 1;
        Ok(())
    }

    pub fn write_sync_line(&mut self, s: &SyncedFrame) -> std::io::Result<()> {
        write_sync_row(&mut self.sync, s)?;
        self.frames_written += 1;
        Ok(())
    }


    pub fn finalize(mut self, info: FinalizeInfo) -> std::io::Result<()> {
        if let Some(fr) = self.frames.as_mut() {
            fr.bin.flush()?;
            fr.bin.get_ref().set_len(fr.offset)?;
            fr.idx.flush()?;
        }
        self.sync.flush()?;
        let mut m = serde_json::to_value(&self.manifest)?;
        m["stopped_unix_ms"] = info.stopped_unix_ms.into();
        m["final_stats"] = info.final_stats;
        m["frames_written"] = self.frames_written.into();
        m["events_written"] = info.events_written.into();
        m["events_dropped"] = info.events_dropped.into();
        m["events_dropped_oob"] = info.events_dropped_oob.into();
        m["clock_model"] = serde_json::to_value(info.clock_model)?;
        m["dropped_frames"] = info.dropped_frames.into();
        m["sync_lost"] = info.sync_lost.into();
        std::fs::write(self.dir.join("manifest.json"), serde_json::to_string_pretty(&m)?)?;
        Ok(())
    }
}

fn write_sync_row(sync: &mut BufWriter<File>, s: &SyncedFrame) -> std::io::Result<()> {
    let f = &s.frame;
    let source = match s.source { fs_core::SyncSource::Matched => "matched",
                                  fs_core::SyncSource::Interpolated => "interpolated" };
    writeln!(sync, r#"{{"seq":{},"t_flir_us":{},"t_evk_us":{},"source":"{}"}}"#,
             f.seq, f.t_cam_us, s.t_evk_us, source)
}

#[cfg(test)]
mod tests {
    use super::*;
    use fs_core::{PixelFormat, RgbFrame, SyncSource, SyncedFrame};

    fn synced(seq: u64, t: i64) -> SyncedFrame {
        SyncedFrame {
            frame: RgbFrame { seq, t_cam_us: t, w: 4, h: 2, data: vec![seq as u8; 8], format: PixelFormat::Bayer8 },
            t_evk_us: t + 3000,
            source: SyncSource::Matched,
        }
    }

    fn manifest() -> Manifest {
        Manifest {
            version: MANIFEST_VERSION, take_name: "t".into(), created_unix_ms: 0, started_unix_ms: 0,
            software_version: SOFTWARE_VERSION.into(),
            fps: 30.0, exposure_us: 5000,
            frame_size: (4, 2), frame_format: "Bayer8".into(),
            frame_storage: "rgb24_bin".into(), encoder_args: None,
            geometry_op: serde_json::json!({"mode": "off"}), calibration_source: None,
            flir: serde_json::json!({}), evk: serde_json::json!({}), board: serde_json::json!(null),
        }
    }

    #[test]
    fn roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let mut w = TakeWriter::create(dir.path(), manifest(), WriterOpts::default(), true).unwrap();
        for i in 0..3 { w.write_frame(&synced(i, i as i64 * 33_333)).unwrap(); }
        assert_eq!(w.bytes_written(), 24);
        w.finalize(FinalizeInfo {
            stopped_unix_ms: 1000,
            final_stats: serde_json::json!({"matched": 3}),
            events_written: 12_345,
            events_dropped: 7,
            events_dropped_oob: 3,
            clock_model: None,
            dropped_frames: 0,
            sync_lost: false,
        }).unwrap();

        let idx = std::fs::read(dir.path().join("frames.idx")).unwrap();
        assert_eq!(idx.len(), 3 * 32);
        assert_eq!(u64::from_le_bytes(idx[32..40].try_into().unwrap()), 1);
        assert_eq!(u64::from_le_bytes(idx[48..56].try_into().unwrap()), 8);
        assert_eq!(u64::from_le_bytes(idx[56..64].try_into().unwrap()), 8);

        let bin = std::fs::read(dir.path().join("frames.bin")).unwrap();
        assert_eq!(bin.len(), 24);
        assert_eq!(&bin[8..16], &[1u8; 8]);

        let sync = std::fs::read_to_string(dir.path().join("sync.jsonl")).unwrap();
        assert_eq!(sync.lines().count(), 3);
        let line0: serde_json::Value = serde_json::from_str(sync.lines().next().unwrap()).unwrap();
        assert_eq!(line0["t_evk_us"], 3000);
        assert_eq!(line0["source"], "matched");

        let m: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.path().join("manifest.json")).unwrap()).unwrap();
        assert_eq!(m["version"], 3);
        assert_eq!(m["final_stats"]["matched"], 3);
        assert_eq!(m["events_written"], 12_345);
        assert_eq!(m["events_dropped"], 7);
        assert_eq!(m["events_dropped_oob"], 3);
        assert!(m.get("events_time_shift_us").is_none(),
                "时间基偏移概念已随 SDK logging 消亡,字段必须整个消失而不是留个 null");
    }


    #[test]
    fn finalized_manifest_has_every_v3_field() {
        let dir = tempfile::tempdir().unwrap();
        let m = Manifest {
            version: MANIFEST_VERSION,
            take_name: "take_2026-08-19_15-30-00".into(),
            created_unix_ms: 1_755_600_000_000,
            started_unix_ms: 1_755_600_000_050,
            software_version: "abc1234".into(),
            fps: 84.5,
            exposure_us: 5000,
            frame_size: (1280, 1024),
            frame_format: "Rgb8".into(),
            frame_storage: "hevc_lossy".into(),
            encoder_args: Some("-hide_banner -loglevel error".into()),
            geometry_op: serde_json::json!({"mode": "undistort"}),
            calibration_source: Some("loaded:C:\\calib\\stereo.json".into()),
            flir: serde_json::json!({"exposure_us": 5000.0, "gain_db": 6.0}),
            evk: serde_json::json!({"biases_available": true}),
            board: serde_json::json!({"inner_cols": 9, "inner_rows": 6}),
        };
        let mut w = TakeWriter::create(dir.path(), m, WriterOpts::default(), true).unwrap();
        for i in 0..2 { w.write_frame(&synced(i, i as i64 * 33_333)).unwrap(); }
        w.finalize(FinalizeInfo {
            stopped_unix_ms: 1_755_600_060_000,
            final_stats: serde_json::json!({
                "matched": 2, "interpolated": 0, "spurious": 0,
                "drift_ppm": 12.0, "offset_us": 3021.0,
            }),
            events_written: 4_800_000,
            events_dropped: 42,
            events_dropped_oob: 17,
            clock_model: Some(ClockParams { x0: 1_755_600_000_050, drift: 0.000012, offset_us: 3021.0 }),
            dropped_frames: 4,
            sync_lost: true,
        }).unwrap();

        let text = std::fs::read_to_string(dir.path().join("manifest.json")).unwrap();
        let m: serde_json::Value = serde_json::from_str(&text).unwrap();

        assert_eq!(m["version"], 3);
        assert_eq!(m["take_name"], "take_2026-08-19_15-30-00");
        assert_eq!(m["created_unix_ms"], 1_755_600_000_000i64);
        assert_eq!(m["started_unix_ms"], 1_755_600_000_050i64);
        assert_eq!(m["software_version"], "abc1234");
        assert_eq!(m["fps"], 84.5);
        assert_eq!(m["exposure_us"], 5000);
        assert_eq!(m["frame_size"], serde_json::json!([1280, 1024]));
        assert_eq!(m["frame_format"], "Rgb8");
        assert_eq!(m["frame_storage"], "hevc_lossy");
        assert_eq!(m["encoder_args"], "-hide_banner -loglevel error");
        assert_eq!(m["geometry_op"], serde_json::json!({"mode": "undistort"}));
        assert_eq!(m["calibration_source"], "loaded:C:\\calib\\stereo.json");
        assert_eq!(m["flir"], serde_json::json!({"exposure_us": 5000.0, "gain_db": 6.0}));
        assert_eq!(m["evk"], serde_json::json!({"biases_available": true}));
        assert_eq!(m["board"], serde_json::json!({"inner_cols": 9, "inner_rows": 6}));
        assert_eq!(m["events_written"], 4_800_000);
        assert_eq!(m["events_dropped"], 42);
        assert_eq!(m["events_dropped_oob"], 17);
        assert!(m.get("events_time_shift_us").is_none(),
                "v3 里这个字段必须不存在(spec §4.4 删除项)");
        assert_eq!(m["clock_model"]["x0"], 1_755_600_000_050i64);
        assert_eq!(m["clock_model"]["drift"], 0.000012);
        assert_eq!(m["clock_model"]["offset_us"], 3021.0);
        assert_eq!(m["final_stats"]["matched"], 2);
        assert_eq!(m["final_stats"]["interpolated"], 0);
        assert_eq!(m["final_stats"]["spurious"], 0);
        assert_eq!(m["final_stats"]["drift_ppm"], 12.0);
        assert_eq!(m["final_stats"]["offset_us"], 3021.0);
        assert_eq!(m["frames_written"], 2);
        assert_eq!(m["dropped_frames"], 4);
        assert_eq!(m["sync_lost"], true);
        assert_eq!(m["stopped_unix_ms"], 1_755_600_060_000i64);
    }

    #[test]
    fn missing_clock_model_is_null_not_a_fabricated_value() {
        let dir = tempfile::tempdir().unwrap();
        let mut w = TakeWriter::create(dir.path(), manifest(), WriterOpts::default(), true).unwrap();
        w.write_frame(&synced(0, 0)).unwrap();
        w.finalize(FinalizeInfo {
            clock_model: None,
            final_stats: serde_json::json!({}),
            ..Default::default()
        }).unwrap();

        let m: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.path().join("manifest.json")).unwrap()).unwrap();
        assert!(m["clock_model"].is_null());
    }


    #[test]
    fn finalize_truncates_the_preallocated_tail() {
        let dir = tempfile::tempdir().unwrap();
        let opts = WriterOpts { buf_bytes: 64 << 10, preallocate_bytes: 1 << 20 };
        let mut w = TakeWriter::create(dir.path(), manifest(), opts, true).unwrap();
        assert_eq!(std::fs::metadata(dir.path().join("frames.bin")).unwrap().len(), 1 << 20,
                   "预分配必须真的落到文件长度上,否则这条测试什么也没验证");
        for i in 0..3 { w.write_frame(&synced(i, i as i64 * 33_333)).unwrap(); }
        w.finalize(FinalizeInfo { final_stats: serde_json::json!({}), ..Default::default() }).unwrap();

        assert_eq!(std::fs::metadata(dir.path().join("frames.bin")).unwrap().len(), 24);
    }

    #[test]
    fn a_frameless_take_writes_sync_rows_but_no_frame_files() {
        let dir = tempfile::tempdir().unwrap();
        let mut w = TakeWriter::create(dir.path(), manifest(), WriterOpts::default(), false).unwrap();
        for i in 0..3 {
            w.write_sync_line(&synced(i, i as i64 * 33_333)).unwrap();
        }
        assert_eq!(w.frames_written, 3);
        assert_eq!(w.bytes_written(), 0, "视频档的帧字节归 VideoSink 数,这里必须是 0");
        let err = w.write_frame(&synced(9, 0)).unwrap_err();
        assert!(err.to_string().contains("VideoSink"), "{err}");
        assert_eq!(w.frames_written, 3, "被拒绝的 write_frame 不许计数");
        w.finalize(FinalizeInfo { final_stats: serde_json::json!({}), ..Default::default() }).unwrap();

        assert!(!dir.path().join("frames.bin").exists(), "视频档不该有 frames.bin");
        assert!(!dir.path().join("frames.idx").exists(), "视频档不该有 frames.idx");
        let sync = std::fs::read_to_string(dir.path().join("sync.jsonl")).unwrap();
        assert_eq!(sync.lines().count(), 3, "sync 行是视频档唯一的帧索引,必须一行不少");
        let m: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.path().join("manifest.json")).unwrap()).unwrap();
        assert_eq!(m["frames_written"], 3);
    }

    #[test]
    fn a_failing_sink_surfaces_its_error() {
        struct Full;
        impl Write for Full {
            fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
                Err(std::io::Error::new(std::io::ErrorKind::Other, "There is not enough space on the disk."))
            }
            fn flush(&mut self) -> std::io::Result<()> { Ok(()) }
        }
        impl FrameSink for Full {
            fn set_len(&self, _: u64) -> std::io::Result<()> { Ok(()) }
        }

        let dir = tempfile::tempdir().unwrap();
        let opts = WriterOpts { buf_bytes: 0, preallocate_bytes: 0 };
        let mut w = TakeWriter::with_frame_sink(dir.path(), manifest(), opts, Box::new(Full)).unwrap();
        let err = w.write_frame(&synced(0, 0)).unwrap_err();
        assert!(err.to_string().contains("not enough space"), "{err}");
    }
}
