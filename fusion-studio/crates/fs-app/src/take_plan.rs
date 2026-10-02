
use std::os::windows::ffi::OsStrExt;
use std::path::{Path, PathBuf};

use fs_record::video::FrameStorage;
use fs_record::{Manifest, WriterOpts, MANIFEST_VERSION, SOFTWARE_VERSION};

use windows_sys::Win32::Storage::FileSystem::GetDiskFreeSpaceExW;

use crate::stream_rates::{EventRate, FrameSpec};

pub const MIN_FREE_BYTES: u64 = 2_000_000_000;

const PREALLOCATE_SECS: f64 = 30.0;

const ILLEGAL_NAME_CHARS: &[char] = &['\\', '/', ':', '*', '?', '"', '<', '>', '|'];

const HEVC_LOSSLESS_VS_RAW: f64 = 0.4;
const HEVC_LOSSY_VS_RAW: f64 = 0.08;


pub struct TakePlan {

    pub dir: PathBuf,
    pub frame: Option<FrameSpec>,
    pub fps: f64,
    pub frame_bps: Option<f64>,

    pub frame_basis: String,
    pub event: Option<EventRate>,
    pub total_bps: Option<f64>,
    pub free_bytes: Option<u64>,
    pub minutes: Option<f64>,

    pub refusal: Option<String>,
}

pub struct TakeInputs<'a> {
    pub dir_text: &'a str,
    pub name: &'a str,
    pub fps: f64,
    pub frame: Option<FrameSpec>,
    pub event: Option<EventRate>,

    pub storage: FrameStorage,
    pub ffmpeg_probe: Option<&'a Result<(), String>>,
    pub geometry: fs_calib::GeomMode,
    pub has_active_calib: bool,
}

impl TakePlan {

    pub fn compute(inp: TakeInputs<'_>, free_bytes: Option<u64>) -> TakePlan {
        let dir = absolute_dir(inp.dir_text);
        let raw_bps = inp.frame.map(|f| f.bytes as f64 * inp.fps.max(0.0));
        let frame_bps = raw_bps.map(|raw| match inp.storage {
            FrameStorage::Rgb24Bin => raw,
            FrameStorage::HevcLossless => raw * HEVC_LOSSLESS_VS_RAW,
            FrameStorage::HevcLossy { .. } => raw * HEVC_LOSSY_VS_RAW,
        });
        let frame_basis = frame_basis(&inp.storage, inp.ffmpeg_probe);
        let total_bps = match (frame_bps, inp.event) {
            (Some(f), Some(e)) => Some(f + e.bytes_per_s),
            _ => None,
        };
        let minutes = match (free_bytes, total_bps) {
            (Some(free), Some(bps)) if bps > 0.0 => {
                Some(free.saturating_sub(MIN_FREE_BYTES) as f64 / bps / 60.0)
            }
            _ => None,
        };
        let refusal = refusal(&inp, &dir, free_bytes);
        TakePlan { dir, frame: inp.frame, fps: inp.fps, frame_bps, frame_basis, event: inp.event, total_bps, free_bytes, minutes, refusal }
    }


    pub fn take_dir(&self, name: &str) -> PathBuf {
        self.dir.join(name.trim())
    }
}

fn frame_basis(storage: &FrameStorage, probe: Option<&Result<(), String>>) -> String {
    let (tier, factor, band) = match storage {
        FrameStorage::Rgb24Bin => return "精确:逐帧字节 × fps".into(),
        FrameStorage::HevcLossless => ("无损", HEVC_LOSSLESS_VS_RAW, "40–60 MB/s"),
        FrameStorage::HevcLossy { .. } => ("有损", HEVC_LOSSY_VS_RAW, "5–15 MB/s"),
    };
    let mut s = format!("估计:HEVC {tier} ≈{factor}× 裸 RGB(经验带宽 {band} 档,真实码率由内容决定)");
    if matches!(probe, Some(Ok(()))) {
        s.push_str(";已探测到 hevc_nvenc 编码器(硬件可用性由首帧验证)");
    }
    s
}

fn refusal(inp: &TakeInputs<'_>, dir: &Path, free_bytes: Option<u64>) -> Option<String> {
    if inp.dir_text.trim().is_empty() {
        return Some("请先填写落盘目录(例如 .\\takes\\)".into());
    }
    let name = inp.name.trim();
    if name.is_empty() {
        return Some("请先填写 take 名称".into());
    }
    if let Some(c) = name.chars().find(|c| ILLEGAL_NAME_CHARS.contains(c)) {
        return Some(format!("take 名称不能包含 {c} —— 它会直接用作目录名,请改成字母、数字、`-` 或 `_`"));
    }
    if inp.frame.is_none() {
        return Some("还没有收到同步帧 —— 请到「相机」tab 确认同步已建立后再开始".into());
    }
    if !matches!(inp.geometry, fs_calib::GeomMode::Off) && !inp.has_active_calib {
        return Some("几何处理开着却没有当前标定 —— 这是门控失守的 bug,请报告;先到「标定」tab 关闭几何处理".into());
    }
    if let fs_calib::GeomMode::SharedK { target } = inp.geometry {
        if let Err(e) = fs_calib::geom::validate_shared_target(target) {
            return Some(format!("SharedK 目标尺寸非法({e})—— 请到「标定」tab 修正后再开始"));
        }
    }
    if !matches!(inp.storage, FrameStorage::Rgb24Bin) {
        match inp.ffmpeg_probe {
            None => {
                return Some(
                    "正在探测 ffmpeg / hevc_nvenc 可用性 —— 请稍候再开始;若长时间停留,请检查 ffmpeg 路径(可能指向失效的网络位置)或重启程序".into(),
                )
            }
            Some(Err(e)) => return Some(format!("视频档不可用:{e}")),
            Some(Ok(())) => {}
        }
    }
    let Some(free) = free_bytes else {
        return Some("读不到该目录所在磁盘的剩余空间 —— 请确认路径有效(盘符存在、不是断开的网络位置)".into());
    };
    if free < MIN_FREE_BYTES {
        return Some(format!(
            "剩余空间不足 {} GB(当前 {:.1} GB)—— 请清理磁盘,或把目录换到别的盘再开始",
            MIN_FREE_BYTES / 1_000_000_000,
            free as f64 / 1e9
        ));
    }
    let take = dir.join(name);
    if take.exists() {
        return Some(format!("{} 已经存在 —— 请换一个 take 名称再开始", take.display()));
    }
    None
}

pub fn writer_opts(frame_bps: Option<f64>, free_bytes: Option<u64>) -> WriterOpts {
    let want = match frame_bps {
        Some(bps) if bps > 0.0 => (bps * PREALLOCATE_SECS) as u64,
        _ => 0,
    };
    let cap = free_bytes.map_or(0, |f| f.saturating_sub(MIN_FREE_BYTES) / 2);
    WriterOpts { preallocate_bytes: want.min(cap), ..WriterOpts::default() }
}

pub struct ManifestInputs<'a> {
    pub take_name: &'a str,
    pub fps: f64,
    pub exposure_us: i64,
    pub frame: FrameSpec,

    pub flir: serde_json::Value,
    pub evk: serde_json::Value,
    pub board: serde_json::Value,
}

pub fn manifest(inp: ManifestInputs<'_>, now_unix_ms: i64) -> Manifest {
    Manifest {
        version: MANIFEST_VERSION,
        take_name: inp.take_name.trim().to_string(),
        created_unix_ms: now_unix_ms,
        started_unix_ms: now_unix_ms,
        software_version: SOFTWARE_VERSION.to_string(),
        fps: inp.fps,
        exposure_us: inp.exposure_us,
        frame_size: (inp.frame.w, inp.frame.h),
        frame_format: inp.frame.format_name(),
        frame_storage: FrameStorage::Rgb24Bin.manifest_name().to_string(),
        encoder_args: None,
        geometry_op: serde_json::json!({"mode": "off"}),
        calibration_source: None,
        flir: inp.flir,
        evk: inp.evk,
        board: inp.board,
    }
}


pub fn board_json(session_exists: bool, board: &fs_calib::board::BoardConfig) -> serde_json::Value {
    if !session_exists {
        return serde_json::Value::Null;
    }
    serde_json::to_value(board).unwrap_or_else(|e| serde_json::json!({ "serialize_error": e.to_string() }))
}


pub fn free_space(dir: &Path) -> Option<u64> {
    let existing = nearest_existing(dir)?;
    let mut wide: Vec<u16> = existing.as_os_str().encode_wide().collect();
    wide.push(0);
    let mut avail: u64 = 0;
    let ok = unsafe {
        let (mut total, mut free) = (0u64, 0u64);
        GetDiskFreeSpaceExW(wide.as_ptr(), &mut avail, &mut total, &mut free)
    };
    (ok != 0).then_some(avail)
}


pub fn nearest_existing(dir: &Path) -> Option<PathBuf> {
    dir.ancestors().find(|a| a.is_dir()).map(|a| a.to_path_buf())
}


pub fn absolute_dir(text: &str) -> PathBuf {
    absolute(Path::new(text.trim()))
}

fn absolute(p: &Path) -> PathBuf {
    if p.is_absolute() {
        return p.to_path_buf();
    }
    match std::env::current_dir() {
        Ok(cwd) => cwd.join(p),
        Err(_) => p.to_path_buf(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stream_rates::RateBasis;
    use fs_core::PixelFormat;

    const FLIR_FRAME: FrameSpec =
        FrameSpec { w: 1280, h: 1024, format: PixelFormat::Bayer8, bytes: 1_310_720 };

    fn inputs<'a>(dir: &'a str, name: &'a str) -> TakeInputs<'a> {
        TakeInputs {
            dir_text: dir,
            name,
            fps: 30.0,
            frame: Some(FLIR_FRAME),
            event: Some(EventRate { bytes_per_s: 23_300_000.0, basis: RateBasis::RawEvt3Bytes }),
            storage: FrameStorage::Rgb24Bin,
            ffmpeg_probe: None,
            geometry: fs_calib::GeomMode::Off,
            has_active_calib: false,
        }
    }

    #[test]
    fn geometry_fallback_refusals_guard_the_gate() {
        let mut on_without_calib = inputs(".\\takes\\", "t");
        on_without_calib.geometry = fs_calib::GeomMode::Undistort;
        let r = TakePlan::compute(on_without_calib, Some(1_000 << 30)).refusal.expect("非 Off 无标定必须拒绝");
        assert!(r.contains("bug"), "这是门控失守,要说成 bug 而不是普通操作错误:{r}");

        let mut odd_target = inputs(".\\takes\\", "t");
        odd_target.geometry = fs_calib::GeomMode::SharedK { target: (1281, 720) };
        odd_target.has_active_calib = true;
        let r = TakePlan::compute(odd_target, Some(1_000 << 30)).refusal.expect("奇数目标尺寸必须拒绝");
        assert!(r.contains("奇数"), "必须转述 validate_shared_target 的原话:{r}");

        let mut huge_target = inputs(".\\takes\\", "t");
        huge_target.geometry = fs_calib::GeomMode::SharedK { target: (4096, 720) };
        huge_target.has_active_calib = true;
        let r = TakePlan::compute(huge_target, Some(1_000 << 30)).refusal.expect("超 4094 的目标必须拒绝");
        assert!(r.contains("4094"), "{r}");

        let mut legal = inputs(".\\takes\\", "t");
        legal.geometry = fs_calib::GeomMode::SharedK { target: (1280, 720) };
        legal.has_active_calib = true;
        assert!(TakePlan::compute(legal, Some(1_000 << 30)).refusal.is_none(), "合法的几何组合不该被兜底拦住");
    }


    #[test]
    fn total_bitrate_is_frames_plus_events() {
        let p = TakePlan::compute(inputs(".\\takes\\", "t"), Some(1_000 << 30));
        assert!((p.frame_bps.unwrap() - 39_321_600.0).abs() < 1.0);
        assert!((p.total_bps.unwrap() - 62_621_600.0).abs() < 1.0);
    }

    #[test]
    fn frame_rate_is_exact_for_raw_and_labeled_estimates_for_hevc() {
        let raw = TakePlan::compute(inputs(".\\takes\\", "t"), Some(1_000 << 30));
        assert!((raw.frame_bps.unwrap() - 39_321_600.0).abs() < 1.0);
        assert!(raw.frame_basis.contains("精确"), "{}", raw.frame_basis);

        let ok = Ok(());
        let mut lossless = inputs(".\\takes\\", "t");
        lossless.storage = FrameStorage::HevcLossless;
        lossless.ffmpeg_probe = Some(&ok);
        let p = TakePlan::compute(lossless, Some(1_000 << 30));
        assert!((p.frame_bps.unwrap() - 39_321_600.0 * 0.4).abs() < 1.0, "{:?}", p.frame_bps);
        assert!(p.frame_basis.contains("估计"), "{}", p.frame_basis);
        assert!(p.frame_basis.contains("0.4"), "估计的依据要在标签里:{}", p.frame_basis);
        assert!(p.frame_basis.contains("首帧"), "探测的边界必须说清:{}", p.frame_basis);

        let mut lossy = inputs(".\\takes\\", "t");
        lossy.storage = FrameStorage::HevcLossy { cq: 19 };
        lossy.ffmpeg_probe = Some(&ok);
        let p = TakePlan::compute(lossy, Some(1_000 << 30));
        assert!((p.frame_bps.unwrap() - 39_321_600.0 * 0.08).abs() < 1.0, "{:?}", p.frame_bps);
        assert!(p.frame_basis.contains("估计"), "{}", p.frame_basis);
    }


    #[test]
    fn video_storage_requires_a_passing_ffmpeg_probe() {
        let failed: Result<(), String> = Err("ffmpeg(C:\\x\\ffmpeg.exe)没有 hevc_nvenc 编码器".into());
        let mut inp = inputs(".\\takes\\", "t");
        inp.storage = FrameStorage::HevcLossy { cq: 19 };
        inp.ffmpeg_probe = Some(&failed);
        let r = TakePlan::compute(inp, Some(1_000 << 30)).refusal.expect("探测失败必须拒绝");
        assert!(r.contains("hevc_nvenc"), "必须转述探测的原话,用户才知道修什么:{r}");

        let mut pending = inputs(".\\takes\\", "t");
        pending.storage = FrameStorage::HevcLossless;
        let r = TakePlan::compute(pending, Some(1_000 << 30)).refusal.expect("没探测完必须拒绝");
        assert!(r.contains("探测"), "{r}");

        let ok = Ok(());
        let mut passing = inputs(".\\takes\\", "t");
        passing.storage = FrameStorage::HevcLossless;
        passing.ffmpeg_probe = Some(&ok);
        assert!(TakePlan::compute(passing, Some(1_000 << 30)).refusal.is_none());

        assert!(TakePlan::compute(inputs(".\\takes\\", "t"), Some(1_000 << 30)).refusal.is_none(),
                "裸 RGB 档不经 ffmpeg,不该被探测挡住");
    }

    #[test]
    fn an_unknown_event_rate_makes_the_estimate_unknown_not_optimistic() {
        let mut inp = inputs(".\\takes\\", "t");
        inp.event = None;
        let p = TakePlan::compute(inp, Some(1_000 << 30));
        assert!(p.frame_bps.is_some(), "帧那一半是算得准的,不该跟着未知");
        assert_eq!(p.total_bps, None);
        assert_eq!(p.minutes, None);
    }

    #[test]
    fn the_estimate_reserves_the_refusal_threshold() {
        let mut inp = inputs(".\\takes\\", "t");
        inp.fps = 0.0;
        inp.event = Some(EventRate { bytes_per_s: 10_000_000.0, basis: RateBasis::RawEvt3Bytes });
        let p = TakePlan::compute(inp, Some(12_000_000_000));
        assert!((p.minutes.unwrap() - 1000.0 / 60.0).abs() < 0.01, "{:?}", p.minutes);
    }

    #[test]
    fn refuses_below_two_gigabytes_and_says_what_to_do() {
        let p = TakePlan::compute(inputs(".\\takes\\", "t"), Some((1.2 * 1e9) as u64));
        let r = p.refusal.expect("剩余空间不足必须拒绝");
        assert!(r.contains("2 GB"), "{r}");
        assert!(r.contains("1.2 GB"), "必须说出当前值,否则用户不知道差多少:{r}");
        assert!(r.contains("清理") || r.contains("换到"), "必须给出处置办法:{r}");
        assert!(TakePlan::compute(inputs(".\\takes\\", "t"), Some(MIN_FREE_BYTES)).refusal.is_none());
        assert!(TakePlan::compute(inputs(".\\takes\\", "t"), Some(MIN_FREE_BYTES - 1)).refusal.is_some());
    }

    #[test]
    fn refuses_when_free_space_is_unknown() {
        let p = TakePlan::compute(inputs("Z:\\nowhere", "t"), None);
        assert!(p.refusal.unwrap().contains("剩余空间"));
    }

    #[test]
    fn refuses_a_take_dir_that_already_exists() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir(tmp.path().join("take_x")).unwrap();
        let dir_text = tmp.path().to_str().unwrap().to_string();
        let p = TakePlan::compute(inputs(&dir_text, "take_x"), Some(1_000 << 30));
        let r = p.refusal.expect("已存在的目录必须拒绝,否则第二个 take 会覆盖第一个");
        assert!(r.contains("已经存在"), "{r}");
        assert!(r.contains("名称"), "必须告诉用户怎么办 —— 换个名称:{r}");
        assert!(TakePlan::compute(inputs(&dir_text, "take_y"), Some(1_000 << 30)).refusal.is_none());
    }

    #[test]
    fn a_dead_path_is_refused_by_free_space_before_any_per_frame_stat() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir(tmp.path().join("take_x")).unwrap();
        let dir_text = tmp.path().to_str().unwrap().to_string();
        let r = TakePlan::compute(inputs(&dir_text, "take_x"), None).refusal.unwrap();
        assert!(r.contains("剩余空间"), "剩余空间未知必须先拦,不该先去 stat 目录:{r}");
    }

    #[test]
    fn refuses_before_any_synced_frame_has_been_seen() {
        let mut inp = inputs(".\\takes\\", "t");
        inp.frame = None;
        let p = TakePlan::compute(inp, Some(1_000 << 30));
        assert!(p.refusal.unwrap().contains("同步帧"));
    }

    #[test]
    fn refuses_empty_or_illegal_names_and_empty_dirs() {
        assert!(TakePlan::compute(inputs("", "t"), Some(1_000 << 30)).refusal.unwrap().contains("目录"));
        assert!(TakePlan::compute(inputs(".\\takes\\", "   "), Some(1_000 << 30)).refusal.unwrap().contains("名称"));
        let r = TakePlan::compute(inputs(".\\takes\\", "a/b"), Some(1_000 << 30)).refusal.unwrap();
        assert!(r.contains('/'), "{r}");
    }


    #[test]
    fn preallocation_never_outgrows_the_disk() {
        let bps = 39_321_600.0;
        let big = writer_opts(Some(bps), Some(1_000 << 30));
        assert_eq!(big.preallocate_bytes, (bps * PREALLOCATE_SECS) as u64);
        let tight = writer_opts(Some(bps), Some(MIN_FREE_BYTES + 100_000_000));
        assert_eq!(tight.preallocate_bytes, 50_000_000);
        assert_eq!(writer_opts(Some(bps), None).preallocate_bytes, 0);
        assert_eq!(writer_opts(None, Some(1_000 << 30)).preallocate_bytes, 0);
        assert_eq!(big.buf_bytes, WriterOpts::default().buf_bytes);
    }

    #[test]
    fn free_space_works_for_a_directory_that_does_not_exist_yet() {
        let tmp = tempfile::tempdir().unwrap();
        let missing = tmp.path().join("takes").join("take_2026-08-19_15-30-00");
        assert!(!missing.exists());
        assert_eq!(nearest_existing(&missing).as_deref(), Some(tmp.path()));
        assert!(free_space(&missing).is_some_and(|b| b > 0), "临时目录所在盘总该报得出剩余空间");
    }

    #[test]
    fn free_space_is_none_for_a_volume_that_does_not_exist() {
        assert_eq!(nearest_existing(Path::new("\\\\?\\NoSuchVolume\\x")), None);
        assert_eq!(free_space(Path::new("\\\\?\\NoSuchVolume\\x")), None);
    }

    #[test]
    fn manifest_carries_every_create_time_field() {
        let m = manifest(
            ManifestInputs {
                take_name: "  take_2026-08-19_15-30-00  ",
                fps: 84.5,
                exposure_us: 5000,
                frame: FLIR_FRAME,
                flir: serde_json::json!({"gain_db": 6.0}),
                evk: serde_json::json!({"biases_available": true}),
                board: serde_json::json!({"inner_cols": 7}),
            },
            1_755_600_000_000,
        );
        let v = serde_json::to_value(&m).unwrap();
        assert_eq!(v["version"], MANIFEST_VERSION);
        assert_eq!(v["take_name"], "take_2026-08-19_15-30-00", "名字两端的空白必须去掉:它也是目录名");
        assert_eq!(v["created_unix_ms"], 1_755_600_000_000i64);
        assert_eq!(v["started_unix_ms"], 1_755_600_000_000i64);
        assert_eq!(v["software_version"], SOFTWARE_VERSION);
        assert_eq!(v["fps"], 84.5);
        assert_eq!(v["exposure_us"], 5000);
        assert_eq!(v["frame_size"], serde_json::json!([1280, 1024]));
        assert_eq!(v["frame_format"], "Bayer8");
        assert_eq!(v["frame_storage"], "rgb24_bin");
        assert!(v["encoder_args"].is_null(), "裸 RGB 档没有编码器,必须是 null 而不是空串");
        assert_eq!(v["geometry_op"], serde_json::json!({"mode": "off"}));
        assert!(v["calibration_source"].is_null());
        assert_eq!(v["flir"]["gain_db"], 6.0);
        assert_eq!(v["evk"]["biases_available"], true);
        assert_eq!(v["board"]["inner_cols"], 7);
    }

    #[test]
    fn the_board_is_only_recorded_when_a_calibration_session_exists() {
        let board = fs_calib::board::BoardConfig {
            kind: fs_calib::board::BoardKind::Checkerboard,
            inner_cols: 7,
            inner_rows: 6,
            rect_w_mm: 82.0,
            rect_h_mm: 98.0,
        };
        assert!(board_json(false, &board).is_null());
        assert_eq!(board_json(true, &board)["inner_cols"], 7);
    }

    #[test]
    fn a_take_without_a_calibration_session_records_a_null_board() {
        let m = manifest(
            ManifestInputs {
                take_name: "t",
                fps: 30.0,
                exposure_us: 0,
                frame: FLIR_FRAME,
                flir: serde_json::Value::Null,
                evk: serde_json::Value::Null,
                board: serde_json::Value::Null,
            },
            0,
        );
        assert!(serde_json::to_value(&m).unwrap()["board"].is_null());
    }


    #[test]
    fn take_dir_is_absolute_and_named_after_the_take() {
        let p = TakePlan::compute(inputs(".\\takes\\", "take_x"), Some(1_000 << 30));
        let d = p.take_dir("take_x");
        assert!(d.is_absolute(), "{}", d.display());
        assert!(d.ends_with("take_x"), "{}", d.display());
    }
}
