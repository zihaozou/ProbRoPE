
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;

use crossbeam_channel::{Receiver, RecvTimeoutError, Sender};
use fs_core::bus::{send_latest, EvkMsg, EvkRawMsg};
use fs_core::{Event, EventBatch, TriggerEvent};
use fs_recon::{CarryState, CudaStream, DecodedEventsDevice, GpuEventDecoder};
use metavision_sys::evt3_raw::bytes_to_words;

use crate::pipeline::SHUTDOWN_POLL;
use crate::transform::{EventRecordTap, EventXform};

const GPU_RAW_CHUNK_WORDS: usize = 2048;


pub struct GpuRawIntake {

    pub rx_raw: Receiver<EvkRawMsg>,

    pub trig_tx: Sender<EvkMsg>,
    pub decoder: GpuEventDecoder,
}


pub struct DecodedBatch {
    pub events: DecodedEventsDevice,
    pub stream: Arc<CudaStream>,
    pub t_latest_us: i64,
}


pub struct DecodeFanout {
    pub dev_tx: Sender<DecodedBatch>,
    pub dev_rx_for_drop: Receiver<DecodedBatch>,
    pub host_recon_tx: Sender<EventBatch>,
    pub host_recon_rx_for_drop: Receiver<EventBatch>,

    pub tap: EventRecordTap,

    pub trig_tx: Sender<EvkMsg>,

    pub xform: EventXform,

    pub events_dropped: Arc<AtomicU64>,

    pub host_downloads: Arc<AtomicU64>,
}

pub fn spawn_decode_stage(
    rx_raw: Receiver<EvkRawMsg>,
    decoder: GpuEventDecoder,
    out: DecodeFanout,
    shutdown: Arc<AtomicBool>,
) -> JoinHandle<()> {
    std::thread::spawn(move || run(rx_raw, decoder, out, shutdown))
}

fn run(rx_raw: Receiver<EvkRawMsg>, mut decoder: GpuEventDecoder, out: DecodeFanout, shutdown: Arc<AtomicBool>) {
    let mut carry = CarryState::default();
    let mut words: Vec<u16> = Vec::new();
    loop {
        if shutdown.load(Ordering::Relaxed) {
            return;
        }
        let first = match rx_raw.recv_timeout(SHUTDOWN_POLL) {
            Ok(m) => m,
            Err(RecvTimeoutError::Timeout) => continue,
            Err(RecvTimeoutError::Disconnected) => return,
        };
        words.clear();
        let mut msg = Some(first);
        loop {
            match msg.take() {
                Some(EvkRawMsg::Bytes(bytes)) => {
                    debug_assert!(bytes.len() % 2 == 0, "EvkRawMsg::Bytes must be whole EVT3 words (producer contract)");
                    bytes_to_words(&mut words, &mut None, &bytes);
                }
                Some(EvkRawMsg::ResetState) => {
                    flush(&mut decoder, &mut carry, &mut words, &out);
                    carry = CarryState::default();
                }
                None => unreachable!("msg is Some on every iteration entry"),
            }
            match rx_raw.try_recv() {
                Ok(m) => msg = Some(m),
                Err(_) => break,
            }
        }
        flush(&mut decoder, &mut carry, &mut words, &out);
    }
}


fn flush(decoder: &mut GpuEventDecoder, carry: &mut CarryState, words: &mut Vec<u16>, out: &DecodeFanout) {
    if words.is_empty() {
        return;
    }
    static DECODE_LOGGED: AtomicBool = AtomicBool::new(false);
    static DOWNLOAD_LOGGED: AtomicBool = AtomicBool::new(false);

    let (dev, triggers, _stats) = match decoder.decode_with_state(words, GPU_RAW_CHUNK_WORDS, carry) {
        Ok(v) => v,
        Err(e) => {
            if !DECODE_LOGGED.swap(true, Ordering::Relaxed) {
                eprintln!("decode: GPU 解码失败({e});丢弃本批,解码级继续");
            }
            words.clear();
            return;
        }
    };
    words.clear();

    for tr in triggers {
        let _ = out
            .trig_tx
            .send(EvkMsg::Trigger(TriggerEvent { t_us: tr.t_us, polarity: tr.polarity as i8 }));
    }

    let lut_active = out.xform.is_active();
    let mut host_batch = None;
    if dev.len > 0 && (lut_active || out.tap.is_armed()) {
        out.host_downloads.fetch_add(1, Ordering::Relaxed);
        match dev.download(decoder.stream()) {
            Ok(decoded) => {
                let mut batch = EventBatch {
                    events: decoded.into_iter().map(|e| Event { t_us: e.t_us, x: e.x, y: e.y, p: e.p as i8 }).collect(),
                };
                out.xform.apply(&mut batch);
                out.tap.offer(&batch);
                host_batch = Some(batch);
            }
            Err(e) => {
                if !DOWNLOAD_LOGGED.swap(true, Ordering::Relaxed) {
                    eprintln!("decode: 宿主 download 失败({e});本批不进录制/宿主路");
                }
            }
        }
    }

    if lut_active {
        if let Some(batch) = host_batch {
            if !batch.events.is_empty()
                && send_latest(&out.host_recon_tx, &out.host_recon_rx_for_drop, batch)
            {
                out.events_dropped.fetch_add(1, Ordering::Relaxed);
            }
        }
    } else {
        let b = DecodedBatch { events: dev, stream: decoder.stream().clone(), t_latest_us: carry.time_us() };
        if send_latest(&out.dev_tx, &out.dev_rx_for_drop, b) {
            out.events_dropped.fetch_add(1, Ordering::Relaxed);
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    use crossbeam_channel::bounded;
    use fs_calib::{ActiveCalibration, CalibSource, GeomMode};
    use fs_recon::DecodedEvent;
    use serde_json::json;

    use crate::transform::EventLut;

    const RECV: Duration = Duration::from_secs(5);

    fn word(ty: u16, payload: u16) -> u16 {
        (ty << 12) | (payload & 0x0FFF)
    }
    fn time_high(th: u16) -> u16 {
        word(0x8, th)
    }
    fn time_low(tl: u16) -> u16 {
        word(0x6, tl)
    }
    fn addr_y(y: u16) -> u16 {
        word(0x0, y)
    }
    fn addr_x(x: u16, pol: u16) -> u16 {
        word(0x2, (pol << 11) | x)
    }
    fn ext_trigger(id: u16, pol: u16) -> u16 {
        word(0xA, (id << 8) | pol)
    }


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

    struct Rig {
        raw_tx: Option<Sender<EvkRawMsg>>,
        dev_rx: Receiver<DecodedBatch>,
        host_rx: Receiver<EventBatch>,
        trig_rx: Receiver<EvkMsg>,
        tap: EventRecordTap,
        xform: EventXform,
        events_dropped: Arc<AtomicU64>,
        host_downloads: Arc<AtomicU64>,
        shutdown: Arc<AtomicBool>,
        handle: Option<JoinHandle<()>>,
    }

    fn rig() -> Option<Rig> {
        let decoder = match GpuEventDecoder::new() {
            Ok(d) => d,
            Err(_) => {
                eprintln!("skip: no CUDA");
                return None;
            }
        };
        let (raw_tx, raw_rx) = bounded::<EvkRawMsg>(64);
        let dev = bounded::<DecodedBatch>(16);
        let host = bounded::<EventBatch>(16);
        let (trig_tx, trig_rx) = bounded::<EvkMsg>(64);
        let tap = EventRecordTap::default();
        let xform = EventXform::default();
        let events_dropped = Arc::new(AtomicU64::new(0));
        let host_downloads = Arc::new(AtomicU64::new(0));
        let shutdown = Arc::new(AtomicBool::new(false));
        let handle = spawn_decode_stage(
            raw_rx,
            decoder,
            DecodeFanout {
                dev_tx: dev.0,
                dev_rx_for_drop: dev.1.clone(),
                host_recon_tx: host.0,
                host_recon_rx_for_drop: host.1.clone(),
                tap: tap.clone(),
                trig_tx,
                xform: xform.clone(),
                events_dropped: events_dropped.clone(),
                host_downloads: host_downloads.clone(),
            },
            shutdown.clone(),
        );
        Some(Rig {
            raw_tx: Some(raw_tx),
            dev_rx: dev.1,
            host_rx: host.1,
            trig_rx,
            tap,
            xform,
            events_dropped,
            host_downloads,
            shutdown,
            handle: Some(handle),
        })
    }

    impl Rig {
        fn send_words(&self, words: &[u16]) {
            let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
            self.raw_tx.as_ref().expect("rig 输入端仍在").send(EvkRawMsg::Bytes(bytes)).unwrap();
        }
    }

    impl Drop for Rig {
        fn drop(&mut self) {
            self.raw_tx.take();
            if let Some(h) = self.handle.take() {
                let _ = h.join();
            }
        }
    }

    #[test]
    fn decode_stage_forwards_device_batches() {
        let Some(r) = rig() else { return };
        r.send_words(&[time_high(5), time_low(100), addr_y(10), addr_x(20, 1)]);
        let b = r.dev_rx.recv_timeout(RECV).expect("解码级必须转发设备批");
        assert_eq!(b.events.len, 1);
        let ev = b.events.download(&b.stream).expect("download 校验");
        assert_eq!(ev, vec![DecodedEvent { x: 20, y: 10, p: 1, t_us: 5 * 4096 + 100 }]);
        assert!(b.t_latest_us >= ev[0].t_us, "设备批携带的解码时钟不得落后于批内最后事件");
        assert_eq!(r.host_downloads.load(Ordering::Relaxed), 0, "tap 未挂、LUT 未装:零下载");
        assert!(r.host_rx.try_recv().is_err(), "几何 Off 时宿主通道必须静默");
        assert_eq!(r.events_dropped.load(Ordering::Relaxed), 0);
    }


    #[test]
    fn host_download_only_when_tapped() {
        let Some(r) = rig() else { return };
        r.send_words(&[time_high(1), time_low(0), addr_y(7), addr_x(10, 1)]);
        let b = r.dev_rx.recv_timeout(RECV).expect("设备批照发");
        assert_eq!(b.events.len, 1);
        assert_eq!(r.host_downloads.load(Ordering::Relaxed), 0, "没人要宿主流就不许 download");

        let (tap_tx, tap_rx) = bounded::<EventBatch>(16);
        r.tap.arm(tap_tx);
        r.send_words(&[time_low(50), addr_y(7), addr_x(10, 1), addr_y(5), addr_x(2, 0)]);
        let b = r.dev_rx.recv_timeout(RECV).expect("tap 只是加了一路宿主分接,设备批照发");
        assert_eq!(b.events.len, 2);
        let got = tap_rx.recv_timeout(RECV).expect("tap 必须收到宿主批");
        assert_eq!(r.host_downloads.load(Ordering::Relaxed), 1);
        let dev_events = b.events.download(&b.stream).expect("download 校验");
        assert_eq!(got.events.len(), 2);
        for (h, d) in got.events.iter().zip(&dev_events) {
            assert_eq!((h.x, h.y, h.p as u8, h.t_us), (d.x, d.y, d.p, d.t_us), "tap 的宿主批必须与设备批是同一批事件");
        }
        assert!(r.host_rx.try_recv().is_err(), "LUT 未装时宿主批不喂 recon");

        let (w, h) = (32u32, 24u32);
        let active = active_fixture(w, h, 100.0, w as f64 / 2.0 + 4.0, h as f64 / 2.0);
        let (_m0, m1) = fs_calib::derive(&active, GeomMode::Undistort).expect("derive 必须成功");
        r.xform.set(Some(EventLut::from_cam_maps(&m1, active.cam1.image_size)));
        r.send_words(&[time_low(60), addr_y(7), addr_x(10, 1), addr_y(5), addr_x(2, 0)]);
        let host = r.host_rx.recv_timeout(RECV).expect("LUT On 时宿主批喂 recon");
        assert_eq!(host.events.len(), 1, "x=2 平移后 x'=-2,应被 DROP 哨兵滤掉");
        assert_eq!((host.events[0].x, host.events[0].y), (6, 7));
        assert_eq!(r.xform.oob_dropped.load(Ordering::Relaxed), 1);
        let tapped = tap_rx.recv_timeout(RECV).expect("tap 收到的与喂 recon 的是同一批 LUT 后事件");
        assert_eq!(tapped.events.len(), 1);
        assert_eq!((tapped.events[0].x, tapped.events[0].y), (6, 7));
        assert_eq!(r.host_downloads.load(Ordering::Relaxed), 2);
        assert!(r.dev_rx.try_recv().is_err(), "LUT On 时设备路暂停 —— 决不双喂");
    }

    #[test]
    fn triggers_reach_the_sync_channel() {
        let Some(r) = rig() else { return };
        r.send_words(&[time_high(2), time_low(200), ext_trigger(0, 1)]);
        match r.trig_rx.recv_timeout(RECV).expect("trigger 必须到达同步通道") {
            EvkMsg::Trigger(t) => {
                assert_eq!(t.t_us, 2 * 4096 + 200);
                assert_eq!(t.polarity, 1);
            }
            _ => panic!("同步通道上应是 Trigger 消息"),
        }
        let b = r.dev_rx.recv_timeout(RECV).expect("空设备批也发 —— 它是解码时钟的载体");
        assert_eq!(b.events.len, 0);
        assert_eq!(b.t_latest_us, 2 * 4096 + 200);
    }


    #[test]
    fn reset_state_clears_the_carry() {
        let Some(r) = rig() else { return };
        r.send_words(&[time_high(3000), time_low(0), addr_y(1), addr_x(1, 1)]);
        let b = r.dev_rx.recv_timeout(RECV).expect("重置前的批");
        let ev = b.events.download(&b.stream).expect("download");
        assert_eq!(ev[0].t_us, 3000 * 4096);

        r.raw_tx.as_ref().expect("rig 输入端仍在").send(EvkRawMsg::ResetState).unwrap();
        r.send_words(&[time_high(1), time_low(5), addr_y(2), addr_x(3, 0)]);
        let b = r.dev_rx.recv_timeout(RECV).expect("重置后的批");
        let ev = b.events.download(&b.stream).expect("download");
        assert_eq!(
            ev,
            vec![DecodedEvent { x: 3, y: 2, p: 0, t_us: 4096 + 5 }],
            "重置后必须从新纪元起算,不得继承 2^24 翻卷"
        );
    }

    #[test]
    fn triggers_survive_recon_overload() {
        let Some(r) = rig() else { return };
        const N: usize = 24;
        const DEV_CAP: usize = 16;
        for i in 0..N {
            r.send_words(&[time_high(1), time_low(i as u16), ext_trigger(0, 1)]);
            r.raw_tx.as_ref().expect("rig 输入端仍在").send(EvkRawMsg::ResetState).unwrap();
        }
        for i in 0..N {
            match r.trig_rx.recv_timeout(RECV).expect("recon 过载时 trigger 也必须无损到达") {
                EvkMsg::Trigger(t) => {
                    assert_eq!(t.t_us, 4096 + i as i64, "第 {i} 个 trigger 的时间戳");
                    assert_eq!(t.polarity, 1);
                }
                _ => panic!("同步通道上应是 Trigger 消息"),
            }
        }
        let want_dropped = (N - DEV_CAP) as u64;
        let deadline = std::time::Instant::now() + Duration::from_secs(3);
        while r.events_dropped.load(Ordering::Relaxed) < want_dropped && std::time::Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(10));
        }
        assert_eq!(r.events_dropped.load(Ordering::Relaxed), want_dropped, "24 次发送对容量 16 应恰好丢 8 批");
        let mut kept = 0usize;
        let mut oldest_kept_t = None;
        while let Ok(b) = r.dev_rx.try_recv() {
            if oldest_kept_t.is_none() {
                oldest_kept_t = Some(b.t_latest_us);
            }
            kept += 1;
        }
        assert_eq!(kept, DEV_CAP, "dev 通道应驻留满容量的批数");
        assert_eq!(oldest_kept_t, Some(4096 + (N - DEV_CAP) as i64), "驻留的最旧批应是第 8 批 —— 被丢的是更旧的");
    }


    #[test]
    fn stage_exits_on_shutdown_or_disconnect() {
        let Some(mut r) = rig() else { return };
        r.shutdown.store(true, Ordering::Relaxed);
        let h = r.handle.take().expect("handle 仍在");
        let deadline = std::time::Instant::now() + Duration::from_secs(3);
        while !h.is_finished() && std::time::Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(20));
        }
        assert!(h.is_finished(), "shutdown 置位后解码级必须退出,哪怕发送端还活着");
        h.join().unwrap();

        let Some(mut r) = rig() else { return };
        r.raw_tx.take();
        let h = r.handle.take().expect("handle 仍在");
        let deadline = std::time::Instant::now() + Duration::from_secs(3);
        while !h.is_finished() && std::time::Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(20));
        }
        assert!(h.is_finished(), "输入断开后解码级必须退出");
        h.join().unwrap();
    }
}
