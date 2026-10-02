use std::io::{Read, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, Command, ExitStatus, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};


#[derive(Clone, Copy, PartialEq, Debug)]
pub enum FrameStorage {
    Rgb24Bin,
    HevcLossless,

    HevcLossy { cq: u8 },
}

impl FrameStorage {

    pub fn manifest_name(&self) -> &'static str {
        match self {
            FrameStorage::Rgb24Bin => "rgb24_bin",
            FrameStorage::HevcLossless => "hevc_lossless",
            FrameStorage::HevcLossy { .. } => "hevc_lossy",
        }
    }
}


const STDERR_TAIL_BYTES: usize = 2048;

const FINALIZE_DEADLINE: Duration = Duration::from_secs(45);

const FINALIZE_POLL: Duration = Duration::from_millis(50);

pub fn ffmpeg_args(storage: &FrameStorage, size: (u32, u32), fps: f64, out: &Path) -> Vec<String> {
    let encoder: Vec<String> = match storage {
        FrameStorage::Rgb24Bin => {
            panic!("Rgb24Bin 不经 ffmpeg(裸帧直写 frames.bin)—— 调用方帧路选错了")
        }
        FrameStorage::HevcLossless => ["-c:v", "hevc_nvenc", "-preset", "p7", "-tune", "lossless", "-pix_fmt", "yuv444p"]
            .map(String::from)
            .into(),
        FrameStorage::HevcLossy { cq } => {
            let q = format!("{cq}");
            ["-c:v", "hevc_nvenc", "-preset", "p5", "-rc", "constqp", "-qp", &q, "-pix_fmt", "yuv420p"]
                .map(String::from)
                .into()
        }
    };
    let mut args: Vec<String> = [
        "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
    ]
    .map(String::from)
    .into();
    args.extend(["-s".into(), format!("{}x{}", size.0, size.1)]);
    args.extend(["-r".into(), format!("{fps}")]);
    args.extend(["-i".into(), "-".into()]);
    args.extend(encoder);
    args.extend(["-y".into(), out.to_string_lossy().into_owned()]);
    args
}


pub fn probe_ffmpeg(ffmpeg: &Path) -> Result<(), String> {
    let out = Command::new(ffmpeg)
        .args(["-hide_banner", "-encoders"])
        .stdin(Stdio::null())
        .output()
        .map_err(|e| {
            format!("无法运行 ffmpeg({}): {e} —— 请安装 ffmpeg 或在设置里指定其路径", ffmpeg.display())
        })?;
    if !out.status.success() {
        return Err(format!(
            "ffmpeg({})连 -encoders 都跑不成功({});stderr 尾巴: {}",
            ffmpeg.display(),
            out.status,
            utf8_tail(&out.stderr)
        ));
    }
    if !String::from_utf8_lossy(&out.stdout).contains("hevc_nvenc") {
        return Err(format!(
            "ffmpeg({})没有 hevc_nvenc 编码器 —— HEVC 档需要 NVIDIA 显卡与带 nvenc 的 ffmpeg 构建",
            ffmpeg.display()
        ));
    }
    Ok(())
}

#[derive(Debug)]
pub struct VideoSink {
    child: Child,
    stdin: Option<ChildStdin>,
    stderr_tail: Arc<Mutex<Vec<u8>>>,
    stderr_thread: Option<std::thread::JoinHandle<()>>,
    frame_bytes: usize,
    bytes_in: u64,
}

impl VideoSink {
    pub fn create(
        ffmpeg: &Path,
        storage: &FrameStorage,
        size: (u32, u32),
        fps: f64,
        out: &Path,
    ) -> Result<Self, String> {
        if matches!(storage, FrameStorage::Rgb24Bin) {
            return Err("Rgb24Bin 不经 ffmpeg(裸帧直写 frames.bin)—— 调用方帧路选错了".into());
        }
        if out.extension().and_then(|e| e.to_str()) != Some("mkv") {
            return Err(format!("输出必须是 .mkv(崩溃容错容器,spec §4.2),给的是 {}", out.display()));
        }
        let args = ffmpeg_args(storage, size, fps, out);
        let child = Command::new(ffmpeg)
            .args(&args)
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("无法启动 ffmpeg({}): {e}", ffmpeg.display()))?;
        Ok(Self::wrap_child(child, size.0 as usize * size.1 as usize * 3))
    }

    fn wrap_child(mut child: Child, frame_bytes: usize) -> Self {
        let stdin = child.stdin.take().expect("Stdio::piped() 之后 stdin 必然存在");
        let stderr = child.stderr.take().expect("Stdio::piped() 之后 stderr 必然存在");
        let stderr_tail = Arc::new(Mutex::new(Vec::new()));
        let tail = stderr_tail.clone();
        let stderr_thread = std::thread::spawn(move || drain_stderr(stderr, &tail));
        Self {
            child,
            stdin: Some(stdin),
            stderr_tail,
            stderr_thread: Some(stderr_thread),
            frame_bytes,
            bytes_in: 0,
        }
    }


    pub fn push_frame(&mut self, rgb24: &[u8]) -> Result<(), String> {
        if rgb24.len() != self.frame_bytes {
            return Err(format!("帧长度 {} 字节 != 期望 {}(w*h*3),拒绝写入", rgb24.len(), self.frame_bytes));
        }
        let stdin = self.stdin.as_mut().expect("stdin 只被 finalize 拿走,而 finalize 消费 self");
        if let Err(e) = stdin.write_all(rgb24) {
            return Err(self.death_report(&format!("写 ffmpeg 管道失败: {e}")));
        }
        self.bytes_in += rgb24.len() as u64;
        Ok(())
    }

    pub fn bytes_in(&self) -> u64 {
        self.bytes_in
    }


    pub fn finalize(self) -> Result<(), String> {
        self.finalize_with_deadline(FINALIZE_DEADLINE)
    }


    fn finalize_with_deadline(mut self, deadline: Duration) -> Result<(), String> {
        drop(self.stdin.take());
        let started = Instant::now();
        let mut status: Option<ExitStatus> = None;
        while started.elapsed() < deadline {
            match self.child.try_wait() {
                Ok(Some(s)) => {
                    status = Some(s);
                    break;
                }
                Ok(None) => std::thread::sleep(FINALIZE_POLL),
                Err(e) => return Err(format!("等待 ffmpeg 退出失败: {e}")),
            }
        }
        if status.is_none() {
            if let Ok(Some(s)) = self.child.try_wait() {
                status = Some(s);
            }
        }
        let killed = status.is_none();
        let status = match status {
            Some(s) => s,
            None => {
                if let Err(e) = self.child.kill() {
                    if e.kind() != std::io::ErrorKind::InvalidInput {
                        return Err(format!(
                            "ffmpeg 收尾超过 {deadline:?} 未退出,kill 又失败({e})—— 子进程可能仍在运行,请在任务管理器里确认"
                        ));
                    }
                }
                self.child
                    .wait()
                    .map_err(|e| format!("等待被 kill 的 ffmpeg 退出失败: {e}"))?
            }
        };
        if let Some(h) = self.stderr_thread.take() {
            let _ = h.join();
        }
        let tail = utf8_tail(&self.stderr_tail.lock().unwrap_or_else(|p| p.into_inner()));
        if killed {
            return Err(format!(
                "ffmpeg 收尾超过 {deadline:?} 仍未退出,已强制结束(编码器缓冲内的最后几帧已丢失,take 其余部分有效);stderr 尾巴: {tail}"
            ));
        }
        if !status.success() {
            return Err(format!("ffmpeg 收尾退出码非零({status});stderr 尾巴: {tail}"));
        }
        Ok(())
    }

    fn death_report(&mut self, what: &str) -> String {
        let mut probe: std::io::Result<Option<ExitStatus>> = Ok(None);
        for _ in 0..25 {
            probe = self.child.try_wait();
            match &probe {
                Ok(Some(_)) | Err(_) => break,
                Ok(None) => std::thread::sleep(Duration::from_millis(20)),
            }
        }
        if matches!(probe, Ok(Some(_))) {
            if let Some(h) = self.stderr_thread.take() {
                let _ = h.join();
            }
        }
        let verdict = match &probe {
            Ok(Some(s)) => format!("ffmpeg 已退出({s})"),
            Ok(None) => "ffmpeg 仍在运行".to_string(),
            Err(e) => format!("ffmpeg 状态未知(try_wait 失败: {e})"),
        };
        format!(
            "{what};{verdict};stderr 尾巴: {}",
            utf8_tail(&self.stderr_tail.lock().unwrap_or_else(|p| p.into_inner()))
        )
    }
}

fn drain_stderr(mut src: impl Read, tail: &Mutex<Vec<u8>>) {
    let mut buf = [0u8; 512];
    loop {
        match src.read(&mut buf) {
            Ok(0) | Err(_) => return,
            Ok(n) => {
                let mut t = tail.lock().unwrap_or_else(|p| p.into_inner());
                t.extend_from_slice(&buf[..n]);
                let overflow = t.len().saturating_sub(STDERR_TAIL_BYTES);
                if overflow > 0 {
                    t.drain(..overflow);
                }
            }
        }
    }
}

fn utf8_tail(bytes: &[u8]) -> String {
    if bytes.is_empty() {
        return "(无输出)".into();
    }
    String::from_utf8_lossy(bytes).trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;


    #[test]
    fn manifest_names_are_the_v3_contract() {
        assert_eq!(FrameStorage::Rgb24Bin.manifest_name(), "rgb24_bin");
        assert_eq!(FrameStorage::HevcLossless.manifest_name(), "hevc_lossless");
        assert_eq!(FrameStorage::HevcLossy { cq: 19 }.manifest_name(), "hevc_lossy");
    }

    #[test]
    fn lossless_args_are_pinned_flag_by_flag() {
        let args = ffmpeg_args(&FrameStorage::HevcLossless, (1280, 720), 84.5, Path::new("t/frames.mkv"));
        assert_eq!(args, vec![
            "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", "1280x720", "-r", "84.5", "-i", "-",
            "-c:v", "hevc_nvenc", "-preset", "p7", "-tune", "lossless", "-pix_fmt", "yuv444p",
            "-y", "t/frames.mkv",
        ]);
    }

    #[test]
    fn lossy_args_are_pinned_flag_by_flag() {
        let args = ffmpeg_args(&FrameStorage::HevcLossy { cq: 19 }, (160, 64), 30.0, Path::new("frames.mkv"));
        assert_eq!(args, vec![
            "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", "160x64", "-r", "30", "-i", "-",
            "-c:v", "hevc_nvenc", "-preset", "p5", "-rc", "constqp", "-qp", "19", "-pix_fmt", "yuv420p",
            "-y", "frames.mkv",
        ]);
    }


    #[test]
    #[should_panic(expected = "Rgb24Bin 不经 ffmpeg")]
    fn rgb24_storage_has_no_ffmpeg_args() {
        ffmpeg_args(&FrameStorage::Rgb24Bin, (8, 8), 30.0, Path::new("frames.mkv"));
    }


    #[test]
    fn probe_names_the_missing_binary() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("no-such-ffmpeg.exe");
        let err = probe_ffmpeg(&missing).unwrap_err();
        assert!(err.contains(&missing.display().to_string()), "{err}");
        assert!(err.contains("请安装 ffmpeg"), "{err}");
    }

    #[test]
    fn create_with_a_missing_binary_fails_immediately() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("no-such-ffmpeg.exe");
        let err = VideoSink::create(&missing, &FrameStorage::HevcLossless, (160, 64), 30.0,
                                    &dir.path().join("frames.mkv")).unwrap_err();
        assert!(err.contains("无法启动 ffmpeg"), "{err}");
        assert!(err.contains(&missing.display().to_string()), "{err}");
    }

    #[test]
    fn create_rejects_the_raw_storage_tier() {
        let dir = tempfile::tempdir().unwrap();
        let err = VideoSink::create(Path::new("ffmpeg"), &FrameStorage::Rgb24Bin, (160, 64), 30.0,
                                    &dir.path().join("frames.mkv")).unwrap_err();
        assert!(err.contains("frames.bin"), "{err}");
    }


    #[test]
    fn create_rejects_a_non_mkv_output() {
        let dir = tempfile::tempdir().unwrap();
        let err = VideoSink::create(Path::new("ffmpeg"), &FrameStorage::HevcLossless, (160, 64), 30.0,
                                    &dir.path().join("frames.mp4")).unwrap_err();
        assert!(err.contains(".mkv"), "{err}");
    }


    #[test]
    fn push_frame_rejects_a_wrong_length() {
        let dir = tempfile::tempdir().unwrap();
        let mut sink = VideoSink::create(Path::new("powershell.exe"), &FrameStorage::HevcLossless,
                                         (8, 8), 30.0, &dir.path().join("frames.mkv")).unwrap();
        let err = sink.push_frame(&[0u8; 10]).unwrap_err();
        assert!(err.contains("10"), "{err}");
        assert!(err.contains("192"), "错误必须给出期望长度 8*8*3=192:{err}");
    }

    #[test]
    fn a_dead_child_surfaces_its_exit_and_stderr() {
        let dir = tempfile::tempdir().unwrap();
        let mut sink = VideoSink::create(Path::new("powershell.exe"), &FrameStorage::HevcLossless,
                                         (8, 8), 30.0, &dir.path().join("frames.mkv")).unwrap();
        let frame = vec![0u8; 8 * 8 * 3];
        let mut got = None;
        for _ in 0..100_000 {
            if let Err(e) = sink.push_frame(&frame) {
                got = Some(e);
                break;
            }
        }
        let err = got.expect("子进程已死,写入必须在管道缓冲填满后失败");
        assert!(err.contains("已退出"), "必须报告退出状态:{err}");
        assert!(err.contains("-hide_banner"), "必须带上子进程 stderr 的原话:{err}");
    }


    #[test]
    fn finalize_kills_a_hung_child_at_the_deadline() {
        let child = Command::new("powershell.exe")
            .args(["-NoProfile", "-Command", "Start-Sleep 60"])
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .expect("测试替身必须能启动");
        let sink = VideoSink::wrap_child(child, 8 * 8 * 3);
        let t0 = std::time::Instant::now();
        let err = sink.finalize_with_deadline(Duration::from_millis(200)).unwrap_err();
        assert!(
            t0.elapsed() < Duration::from_secs(30),
            "看门狗没起作用:finalize 等了 {:?}",
            t0.elapsed()
        );
        assert!(err.contains("已强制结束"), "{err}");
        assert!(err.contains("200ms"), "必须说出超时上限:{err}");
        assert!(err.contains("stderr 尾巴"), "错误形状必须带 stderr 的下落:{err}");
    }

    #[test]
    fn finalize_reports_a_nonzero_exit_with_stderr() {
        let dir = tempfile::tempdir().unwrap();
        let sink = VideoSink::create(Path::new("powershell.exe"), &FrameStorage::HevcLossy { cq: 19 },
                                     (8, 8), 30.0, &dir.path().join("frames.mkv")).unwrap();
        let err = sink.finalize().unwrap_err();
        assert!(err.contains("退出码非零"), "{err}");
        assert!(err.contains("-hide_banner"), "必须带上子进程 stderr 的原话:{err}");
    }


    #[test]
    #[ignore = "需要 PATH 上有带 hevc_nvenc 的 ffmpeg 与 NVIDIA 显卡"]
    fn real_ffmpeg_smoke_eight_frames_to_mkv() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("frames.mkv");
        let mut sink = VideoSink::create(Path::new("ffmpeg"), &FrameStorage::HevcLossless,
                                         (160, 64), 30.0, &out).unwrap();
        let frame: Vec<u8> = (0..160 * 64 * 3).map(|i| (i % 251) as u8).collect();
        for _ in 0..8 {
            sink.push_frame(&frame).unwrap();
        }
        assert_eq!(sink.bytes_in(), 8 * 160 * 64 * 3);
        sink.finalize().unwrap();
        let len = std::fs::metadata(&out).unwrap().len();
        assert!(len > 0, "frames.mkv 不能是空文件");
    }

    #[test]
    #[ignore = "需要 PATH 上有带 hevc_nvenc 的 ffmpeg 与 NVIDIA 显卡"]
    fn real_ffmpeg_undersized_frames_die_loudly() {
        let dir = tempfile::tempdir().unwrap();
        let mut sink = VideoSink::create(Path::new("ffmpeg"), &FrameStorage::HevcLossless,
                                         (8, 8), 30.0, &dir.path().join("frames.mkv")).unwrap();
        let frame = vec![0u8; 8 * 8 * 3];
        let mut err = None;
        for _ in 0..8 {
            if let Err(e) = sink.push_frame(&frame) {
                err = Some(e);
                break;
            }
        }
        let err = match err {
            Some(e) => e,
            None => sink.finalize().unwrap_err(),
        };
        assert!(err.to_lowercase().contains("minimum"),
                "错误必须带上编码器对最小画幅的原话:{err}");
    }


    #[test]
    fn probe_error_survives_a_unicode_path() {
        let dir = tempfile::tempdir().unwrap();
        let missing: PathBuf = dir.path().join("视频工具").join("ffmpeg.exe");
        let err = probe_ffmpeg(&missing).unwrap_err();
        assert!(err.contains("视频工具"), "{err}");
    }
}
