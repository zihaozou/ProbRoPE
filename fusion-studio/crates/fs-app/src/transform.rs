
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use crossbeam_channel::{Receiver, RecvTimeoutError, Sender};
use fs_calib::CamMaps;
use fs_core::bus::send_latest;
use fs_core::{EventBatch, PixelFormat, RgbFrame, SyncedFrame};
use opencv::core::{Mat, Scalar, ToInputArray, BORDER_CONSTANT};
use opencv::imgproc;
use opencv::prelude::*;

use crate::pipeline::SHUTDOWN_POLL;
use crate::record_worker::RecordTx;


pub struct CamXform {

    pub src_size: (u32, u32),
    pub maps: CamMaps,
}

pub type CamMapsPair = (CamXform, CamXform);

pub struct FrameFanout {

    pub synced: (Sender<Arc<SyncedFrame>>, Receiver<Arc<SyncedFrame>>),

    pub calib: (Sender<Arc<SyncedFrame>>, Receiver<Arc<SyncedFrame>>),

    pub record: Option<RecordTx>,
}


pub fn spawn_frame_transform(
    rx: Receiver<Arc<SyncedFrame>>,
    txs: FrameFanout,
    maps: Arc<Mutex<Option<CamMapsPair>>>,
    shutdown: Arc<AtomicBool>,
) -> JoinHandle<()> {
    std::thread::spawn(move || run(rx, txs, maps, shutdown))
}

fn run(
    rx: Receiver<Arc<SyncedFrame>>,
    txs: FrameFanout,
    maps: Arc<Mutex<Option<CamMapsPair>>>,
    shutdown: Arc<AtomicBool>,
) {
    static CONVERT_LOGGED: AtomicBool = AtomicBool::new(false);
    static DIMS_LOGGED: AtomicBool = AtomicBool::new(false);
    static REMAP_LOGGED: AtomicBool = AtomicBool::new(false);

    loop {
        let sf = match rx.recv_timeout(SHUTDOWN_POLL) {
            Ok(sf) => sf,
            Err(RecvTimeoutError::Timeout) => {
                if shutdown.load(Ordering::Relaxed) {
                    return;
                }
                continue;
            }
            Err(RecvTimeoutError::Disconnected) => return,
        };

        let frame = match to_rgb8(&sf.frame) {
            Ok(f) => f,
            Err(e) => {
                log_once(&CONVERT_LOGGED, &format!("transform: 帧转换失败({e});丢帧"));
                continue;
            }
        };
        let pre = Arc::new(SyncedFrame { frame, t_evk_us: sf.t_evk_us, source: sf.source });

        send_latest(&txs.calib.0, &txs.calib.1, Arc::clone(&pre));

        let post = {
            let guard = maps.lock().unwrap_or_else(|p| p.into_inner());
            match guard.as_ref() {
                None => pre,
                Some((cam0, _)) => {
                    if (pre.frame.w, pre.frame.h) != cam0.src_size {
                        log_once(
                            &DIMS_LOGGED,
                            &format!(
                                "transform: 帧尺寸 {}x{} 与映射表源尺寸 {}x{} 不符 —— 门控失守?丢帧",
                                pre.frame.w, pre.frame.h, cam0.src_size.0, cam0.src_size.1
                            ),
                        );
                        continue;
                    }
                    match remap_rgb8(&pre.frame, &cam0.maps) {
                        Ok(f) => Arc::new(SyncedFrame { frame: f, t_evk_us: pre.t_evk_us, source: pre.source }),
                        Err(e) => {
                            log_once(&REMAP_LOGGED, &format!("transform: remap 失败({e});丢帧"));
                            continue;
                        }
                    }
                }
            }
        };

        if let Some(record) = &txs.record {
            record.send(Arc::clone(&post));
        }
        send_latest(&txs.synced.0, &txs.synced.1, post);
    }
}

fn log_once(logged: &AtomicBool, msg: &str) {
    if !logged.swap(true, Ordering::Relaxed) {
        eprintln!("{msg}");
    }
}


pub fn to_rgb8(f: &RgbFrame) -> Result<RgbFrame, String> {
    let data = match f.format {
        PixelFormat::Bayer8 => {
            let need = (f.w * f.h) as usize;
            if f.data.len() < need {
                return Err(format!("Bayer8 帧太小({} < {need})", f.data.len()));
            }
            let src = Mat::new_rows_cols_with_data(f.h as i32, f.w as i32, &f.data[..need])
                .map_err(|e| format!("Bayer8 建 Mat 失败:{e}"))?;
            cvt_to_rgb(&src, imgproc::COLOR_BayerBG2RGB, "demosaic")?
        }
        PixelFormat::Bgr8 => {
            let need = (f.w * f.h * 3) as usize;
            if f.data.len() < need {
                return Err(format!("Bgr8 帧太小({} < {need})", f.data.len()));
            }
            let flat = Mat::new_rows_cols_with_data(f.h as i32, (f.w * 3) as i32, &f.data[..need])
                .map_err(|e| format!("Bgr8 建 Mat 失败:{e}"))?;
            let bgr = flat.reshape(3, f.h as i32).map_err(|e| format!("Bgr8 reshape 失败:{e}"))?;
            cvt_to_rgb(&bgr, imgproc::COLOR_BGR2RGB, "BGR->RGB")?
        }
        PixelFormat::Gray8 => {
            let need = (f.w * f.h) as usize;
            if f.data.len() < need {
                return Err(format!("Gray8 帧太小({} < {need})", f.data.len()));
            }
            let src = Mat::new_rows_cols_with_data(f.h as i32, f.w as i32, &f.data[..need])
                .map_err(|e| format!("Gray8 建 Mat 失败:{e}"))?;
            cvt_to_rgb(&src, imgproc::COLOR_GRAY2RGB, "灰度扩展")?
        }
        PixelFormat::Rgb8 => f.data.clone(),
    };
    Ok(RgbFrame { seq: f.seq, t_cam_us: f.t_cam_us, w: f.w, h: f.h, data, format: PixelFormat::Rgb8 })
}

fn cvt_to_rgb(src: &impl ToInputArray, code: i32, what: &str) -> Result<Vec<u8>, String> {
    let mut dst = Mat::default();
    imgproc::cvt_color_def(src, &mut dst, code).map_err(|e| format!("{what}失败:{e}"))?;
    dst.data_bytes().map(|b| b.to_vec()).map_err(|e| format!("读取{what}输出失败:{e}"))
}


fn remap_rgb8(f: &RgbFrame, maps: &CamMaps) -> Result<RgbFrame, String> {
    let need = (f.w * f.h * 3) as usize;
    if f.data.len() < need {
        return Err(format!("remap 源帧太小({} < {need})", f.data.len()));
    }
    let flat = Mat::new_rows_cols_with_data(f.h as i32, (f.w * 3) as i32, &f.data[..need])
        .map_err(|e| format!("remap 源建 Mat 失败:{e}"))?;
    let src = flat.reshape(3, f.h as i32).map_err(|e| format!("remap 源 reshape 失败:{e}"))?;
    let mut dst = Mat::default();
    imgproc::remap(
        &src,
        &mut dst,
        &maps.frame_maps.0,
        &maps.frame_maps.1,
        imgproc::INTER_LINEAR,
        BORDER_CONSTANT,
        Scalar::all(0.0),
    )
    .map_err(|e| format!("remap 失败:{e}"))?;
    let data = dst.data_bytes().map(|b| b.to_vec()).map_err(|e| format!("读取 remap 输出失败:{e}"))?;
    let (ow, oh) = maps.out_size;
    Ok(RgbFrame { seq: f.seq, t_cam_us: f.t_cam_us, w: ow, h: oh, data, format: PixelFormat::Rgb8 })
}


pub struct EventLut {
    src_size: (u32, u32),
    out_size: (u32, u32),
    lut: Vec<u32>,
}

impl EventLut {
    pub fn from_cam_maps(maps: &CamMaps, src_size: (u32, u32)) -> Self {
        assert_eq!(
            src_size.0 as usize * src_size.1 as usize,
            maps.event_lut.len(),
            "EventLut 源尺寸 {}x{} 与映射表长度不符 —— 表与相机配错了对",
            src_size.0,
            src_size.1
        );
        EventLut { src_size, out_size: maps.out_size, lut: maps.event_lut.clone() }
    }
}

#[derive(Clone, Default)]
pub struct EventXform {

    lut: Arc<Mutex<Option<EventLut>>>,
    pub oob_dropped: Arc<AtomicU64>,
}

impl EventXform {

    pub fn set(&self, lut: Option<EventLut>) {
        *self.lut.lock().unwrap_or_else(|p| p.into_inner()) = lut;
    }


    pub fn is_active(&self) -> bool {
        self.lut.lock().unwrap_or_else(|p| p.into_inner()).is_some()
    }

    pub fn apply(&self, batch: &mut EventBatch) {
        let guard = self.lut.lock().unwrap_or_else(|p| p.into_inner());
        let Some(l) = guard.as_ref() else { return };
        let (sw, sh) = l.src_size;
        let ow = l.out_size.0;
        let mut dropped = 0u64;
        batch.events.retain_mut(|e| {
            let (x, y) = (u32::from(e.x), u32::from(e.y));
            if x >= sw || y >= sh {
                dropped += 1;
                return false;
            }
            match l.lut.get((y * sw + x) as usize) {
                Some(&t) if t != u32::MAX => {
                    e.x = (t % ow) as u16;
                    e.y = (t / ow) as u16;
                    true
                }
                _ => {
                    dropped += 1;
                    false
                }
            }
        });
        if dropped > 0 {
            self.oob_dropped.fetch_add(dropped, Ordering::Relaxed);
        }
    }
}


#[derive(Clone, Default)]
pub struct EventRecordTap {
    slot: Arc<Mutex<Option<Sender<EventBatch>>>>,
    pub dropped: Arc<AtomicU64>,
}

impl EventRecordTap {
    pub fn arm(&self, tx: Sender<EventBatch>) {
        *self.slot.lock().unwrap_or_else(|p| p.into_inner()) = Some(tx);
    }

    pub fn is_armed(&self) -> bool {
        self.slot.lock().unwrap_or_else(|p| p.into_inner()).is_some()
    }

    pub fn disarm(&self) {
        *self.slot.lock().unwrap_or_else(|p| p.into_inner()) = None;
    }

    pub fn offer(&self, batch: &EventBatch) {
        let guard = self.slot.lock().unwrap_or_else(|p| p.into_inner());
        let Some(tx) = guard.as_ref() else { return };
        if tx.try_send(batch.clone()).is_err() {
            self.dropped.fetch_add(batch.events.len() as u64, Ordering::Relaxed);
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    use crossbeam_channel::bounded;
    use fs_calib::{ActiveCalibration, CalibSource, GeomMode};
    use fs_core::SyncSource;
    use serde_json::json;

    use crate::record_worker::{record_channel, RecordRx, RECORD_QUEUE_BYTES};

    const RECV: Duration = Duration::from_secs(5);

    fn active_fixture(w: u32, h: u32, f: f64, cx: f64, cy: f64) -> ActiveCalibration {
        let cam = json!({
            "K": [[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]],
            "dist": [0.0, 0.0, 0.0, 0.0, 0.0],
            "image_size": [w, h],
            "reprojection_error_px": 0.2,
        });
        let doc = json!({
            "schema": "stereo_calibration.v1",
            "cameras": { "cam0": cam.clone(), "cam1": cam },
            "extrinsics": {
                "R_cam0_to_cam1": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                "T_cam0_to_cam1": [100.0, 0.0, 0.0],
                "stereo_reprojection_error_px": 0.3,
            },
        });
        fs_calib::active::load_stereo_calibration_v1(doc, CalibSource::LiveSession).expect("fixture 文档必须能加载")
    }

    fn pair_for(active: &ActiveCalibration, mode: GeomMode) -> CamMapsPair {
        let (m0, m1) = fs_calib::derive(active, mode).expect("derive 必须成功");
        (
            CamXform { src_size: active.cam0.image_size, maps: m0 },
            CamXform { src_size: active.cam1.image_size, maps: m1 },
        )
    }

    fn synced_frame(format: PixelFormat, w: u32, h: u32, data: Vec<u8>) -> Arc<SyncedFrame> {
        Arc::new(SyncedFrame {
            frame: RgbFrame { seq: 7, t_cam_us: 1_000, w, h, data, format },
            t_evk_us: 4_000,
            source: SyncSource::Matched,
        })
    }


    struct Rig {
        tx: Option<Sender<Arc<SyncedFrame>>>,
        synced_rx: Receiver<Arc<SyncedFrame>>,
        calib_rx: Receiver<Arc<SyncedFrame>>,
        record_rx: RecordRx,
        shutdown: Arc<AtomicBool>,
        handle: Option<JoinHandle<()>>,
    }

    fn rig(maps: Option<CamMapsPair>) -> Rig {
        let (tx, rx) = bounded::<Arc<SyncedFrame>>(16);
        let synced = bounded(16);
        let calib = bounded(16);
        let synced_rx = synced.1.clone();
        let calib_rx = calib.1.clone();
        let (record_tx, record_rx) = record_channel(RECORD_QUEUE_BYTES);
        let shutdown = Arc::new(AtomicBool::new(false));
        let handle = spawn_frame_transform(
            rx,
            FrameFanout { synced, calib, record: Some(record_tx) },
            Arc::new(Mutex::new(maps)),
            shutdown.clone(),
        );
        Rig { tx: Some(tx), synced_rx, calib_rx, record_rx, shutdown, handle: Some(handle) }
    }

    impl Rig {
        fn send(&self, frame: Arc<SyncedFrame>) {
            self.tx.as_ref().expect("rig 输入端仍在").send(frame).unwrap();
        }
    }

    impl Drop for Rig {
        fn drop(&mut self) {
            self.tx.take();
            if let Some(h) = self.handle.take() {
                let _ = h.join();
            }
        }
    }

    #[test]
    fn off_mode_debayers_to_rgb8() {
        let r = rig(None);

        r.send(synced_frame(PixelFormat::Bayer8, 8, 8, vec![200u8; 64]));
        let out = r.synced_rx.recv_timeout(RECV).expect("Bayer8 帧必须流出");
        assert_eq!(out.frame.format, PixelFormat::Rgb8);
        assert_eq!((out.frame.w, out.frame.h), (8, 8), "Off 模式不改尺寸");
        assert_eq!(out.frame.data.len(), 8 * 8 * 3);
        assert!(out.frame.data.iter().all(|&b| b == 200), "恒值 Bayer 场 demosaic 后仍是恒值");
        assert_eq!(
            (out.frame.seq, out.frame.t_cam_us, out.t_evk_us, out.source),
            (7, 1_000, 4_000, SyncSource::Matched)
        );

        let mut data = vec![0u8; 4 * 2 * 3];
        data[0] = 10;
        data[1] = 20;
        data[2] = 30;
        r.send(synced_frame(PixelFormat::Bgr8, 4, 2, data));
        let out = r.synced_rx.recv_timeout(RECV).expect("Bgr8 帧必须流出");
        assert_eq!(out.frame.format, PixelFormat::Rgb8);
        assert_eq!(out.frame.data.len(), 4 * 2 * 3);
        assert_eq!(&out.frame.data[..3], &[30, 20, 10], "BGR 必须换序成 RGB");
    }

    #[test]
    fn remap_applies_when_mode_on() {
        let (w, h) = (32u32, 24u32);
        let active = active_fixture(w, h, 100.0, w as f64 / 2.0 + 4.0, h as f64 / 2.0);
        let pair = pair_for(&active, GeomMode::Undistort);
        assert_eq!(pair.0.maps.out_size, (w, h), "Undistort 不改尺寸");
        let r = rig(Some(pair));

        let mut data = vec![0u8; (w * h) as usize];
        data[(7 * w + 10) as usize] = 255;
        r.send(synced_frame(PixelFormat::Gray8, w, h, data));

        let out = r.synced_rx.recv_timeout(RECV).expect("remap 后的帧必须流出");
        assert_eq!(out.frame.format, PixelFormat::Rgb8);
        assert_eq!((out.frame.w, out.frame.h), (w, h), "输出尺寸 = out_size");
        assert_eq!(out.frame.data.len(), (w * h * 3) as usize);
        let px = |x: u32, y: u32| {
            let i = ((y * w + x) * 3) as usize;
            [out.frame.data[i], out.frame.data[i + 1], out.frame.data[i + 2]]
        };
        assert!(px(6, 7).iter().all(|&c| c >= 250), "白点应从 (10,7) 平移到 (6,7),实得 {:?}", px(6, 7));
        assert_eq!(px(10, 7), [0, 0, 0], "原位置不应再有白点");
    }

    #[test]
    fn calib_tap_receives_pre_remap_frames() {
        let (w, h) = (32u32, 24u32);
        let target = (16u32, 16u32);
        let active = active_fixture(w, h, 100.0, w as f64 / 2.0, h as f64 / 2.0);
        let pair = pair_for(&active, GeomMode::SharedK { target });
        assert_eq!(pair.0.maps.out_size, target);
        let r = rig(Some(pair));

        r.send(synced_frame(PixelFormat::Gray8, w, h, vec![128u8; (w * h) as usize]));

        let synced = r.synced_rx.recv_timeout(RECV).expect("synced tap 必须收到帧");
        assert_eq!((synced.frame.w, synced.frame.h), target, "synced 收 remap 后的 out_size");
        assert_eq!(synced.frame.format, PixelFormat::Rgb8);

        let calib = r.calib_rx.try_recv().expect("calib tap 必须收到帧");
        assert_eq!((calib.frame.w, calib.frame.h), (w, h), "calib 必须收 remap 前的原始几何");
        assert_eq!(calib.frame.format, PixelFormat::Rgb8, "calib 收的是 debayer 后的帧");

        let record = r.record_rx.take().expect("record tap 必须收到帧");
        assert_eq!((record.frame.w, record.frame.h), target, "record 收 remap 后的 out_size");
        assert_eq!(record.frame.format, PixelFormat::Rgb8);
    }

    #[test]
    fn transform_thread_exits_on_close() {
        let mut r = rig(None);
        r.tx.take();
        let h = r.handle.take().expect("handle 仍在");
        let deadline = std::time::Instant::now() + Duration::from_secs(3);
        while !h.is_finished() && std::time::Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(20));
        }
        assert!(h.is_finished(), "输入断开后线程必须退出");
        h.join().unwrap();

        let mut r = rig(None);
        r.shutdown.store(true, Ordering::Relaxed);
        let h = r.handle.take().expect("handle 仍在");
        let deadline = std::time::Instant::now() + Duration::from_secs(3);
        while !h.is_finished() && std::time::Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(20));
        }
        assert!(h.is_finished(), "shutdown 置位后线程必须退出,哪怕发送端还活着");
        h.join().unwrap();
    }


    #[test]
    fn event_xform_maps_and_drops() {
        use fs_core::{Event, EventBatch};
        let batch_of = |coords: &[(u16, u16)]| EventBatch {
            events: coords.iter().map(|&(x, y)| Event { t_us: 1_234, x, y, p: 1 }).collect(),
        };

        let xform = EventXform::default();
        let mut b = batch_of(&[(10, 7), (2, 5)]);
        xform.apply(&mut b);
        assert_eq!(b.events.len(), 2);
        assert_eq!((b.events[0].x, b.events[0].y), (10, 7));
        assert_eq!(xform.oob_dropped.load(Ordering::Relaxed), 0, "恒等直通不该有越界计数");

        let (w, h) = (32u32, 24u32);
        let active = active_fixture(w, h, 100.0, w as f64 / 2.0 + 4.0, h as f64 / 2.0);
        let pair = pair_for(&active, GeomMode::Undistort);
        xform.set(Some(EventLut::from_cam_maps(&pair.1.maps, pair.1.src_size)));

        let mut b = batch_of(&[(10, 7), (2, 5), (31, 23)]);
        b.events[0].t_us = 42;
        b.events[0].p = 0;
        xform.apply(&mut b);
        assert_eq!(b.events.len(), 2, "x=2 平移后 x'=-2,应被 DROP 哨兵滤掉");
        assert_eq!((b.events[0].x, b.events[0].y), (6, 7));
        assert_eq!((b.events[0].t_us, b.events[0].p), (42, 0), "时间戳与极性绝不能被改写");
        assert_eq!((b.events[1].x, b.events[1].y), (27, 23));
        assert_eq!(xform.oob_dropped.load(Ordering::Relaxed), 1);

        let mut b = batch_of(&[(40, 0), (5, 30), (10, 7)]);
        xform.apply(&mut b);
        assert_eq!(b.events.len(), 1, "超出源尺寸的事件必须被丢弃");
        assert_eq!((b.events[0].x, b.events[0].y), (6, 7));
        assert_eq!(xform.oob_dropped.load(Ordering::Relaxed), 3, "计数器单调累计(1+2),录制 worker 用快照差");

        xform.set(None);
        let mut b = batch_of(&[(2, 5)]);
        xform.apply(&mut b);
        assert_eq!(b.events.len(), 1);
        assert_eq!(xform.oob_dropped.load(Ordering::Relaxed), 3);
    }

    #[test]
    #[should_panic(expected = "映射表长度不符")]
    fn event_lut_construction_asserts_area() {
        let (w, h) = (32u32, 24u32);
        let active = active_fixture(w, h, 100.0, w as f64 / 2.0, h as f64 / 2.0);
        let pair = pair_for(&active, GeomMode::Undistort);
        let _ = EventLut::from_cam_maps(&pair.1.maps, (16, 16));
    }

    #[test]
    fn record_tap_counts_drops_when_full() {
        use fs_core::{Event, EventBatch};
        let batch = |n: usize| EventBatch {
            events: (0..n).map(|i| Event { t_us: i as i64, x: 1, y: 2, p: 1 }).collect(),
        };

        let tap = EventRecordTap::default();
        tap.offer(&batch(5));
        assert_eq!(tap.dropped.load(Ordering::Relaxed), 0);

        let (tx, rx) = bounded::<fs_core::EventBatch>(2);
        tap.arm(tx);
        tap.offer(&batch(5));
        tap.offer(&batch(7));
        tap.offer(&batch(9));
        assert_eq!(tap.dropped.load(Ordering::Relaxed), 9, "按事件数计,不是按批数");
        let got = rx.try_recv().expect("第一批必须送达");
        assert_eq!(got.events.len(), 5);
        assert_eq!(got.events[3].t_us, 3, "送达的是同一批事件,不是空壳");
        assert_eq!(rx.try_recv().expect("第二批必须送达").events.len(), 7);
        assert!(rx.try_recv().is_err(), "第三批已被丢弃");

        tap.offer(&batch(3));
        assert_eq!(rx.try_recv().expect("腾出空间后恢复送达").events.len(), 3);
        assert_eq!(tap.dropped.load(Ordering::Relaxed), 9);

        drop(rx);
        tap.offer(&batch(4));
        assert_eq!(tap.dropped.load(Ordering::Relaxed), 13);

        tap.disarm();
        tap.offer(&batch(100));
        assert_eq!(tap.dropped.load(Ordering::Relaxed), 13);
    }


    #[test]
    fn bad_frames_drop_without_killing_the_thread() {
        let (w, h) = (32u32, 24u32);
        let active = active_fixture(w, h, 100.0, w as f64 / 2.0, h as f64 / 2.0);
        let r = rig(Some(pair_for(&active, GeomMode::Undistort)));

        r.send(synced_frame(PixelFormat::Bayer8, 8, 8, vec![0u8; 5]));
        r.send(synced_frame(PixelFormat::Gray8, 16, 16, vec![0u8; 256]));
        r.send(synced_frame(PixelFormat::Gray8, w, h, vec![0u8; (w * h) as usize]));

        let good = r.synced_rx.recv_timeout(RECV).expect("坏帧之后线程必须还活着,好帧照常流出");
        assert_eq!((good.frame.w, good.frame.h), (w, h));
        assert!(r.synced_rx.try_recv().is_err(), "synced 只该收到那一帧好帧");

        let first = r.calib_rx.try_recv().expect("尺寸不符的帧仍应到达 calib");
        assert_eq!((first.frame.w, first.frame.h), (16, 16));
        let second = r.calib_rx.try_recv().expect("好帧也到达 calib");
        assert_eq!((second.frame.w, second.frame.h), (w, h));
        assert!(r.calib_rx.try_recv().is_err(), "转换失败的帧不该到达任何 tap");

        let record = r.record_rx.take().expect("好帧应到达 record");
        assert_eq!((record.frame.w, record.frame.h), (w, h));
        assert!(r.record_rx.take().is_err(), "record 只该收到那一帧好帧");
    }
}
