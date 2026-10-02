use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crossbeam_channel::{bounded, Sender};
use fs_core::bus::{EvkMsg, FlirMsg};
use fs_core::{Event, EventBatch, PixelFormat, RgbFrame, TriggerEvent};
use fs_recon::{CarryState, GpuEventDecoder};
use metavision_sys::evt3_raw::{time_shift_us, Evt3RawFile};
use metavision_sys::MvCamera;
use opencv::prelude::*;
use opencv::videoio::{VideoCapture, CAP_ANY};

use super::evk_thread::{self, EvkCmd};
use crate::settings::EvkSettingsSnapshot;


pub fn spawn_replay_evk(
    path: String,
    realtime: bool,
    tx: Sender<EvkMsg>,
    snapshot: Arc<Mutex<EvkSettingsSnapshot>>,
    rates: Arc<crate::stream_rates::StreamRates>,
    xform: crate::transform::EventXform,
    record_tap: crate::transform::EventRecordTap,
    shutdown: Arc<std::sync::atomic::AtomicBool>,
) -> (JoinHandle<()>, Sender<EvkCmd>, (u32, u32)) {
    let (cmd_tx, cmd_rx) = bounded::<EvkCmd>(16);
    let cam = MvCamera::open_file(&path, realtime).expect("open raw");
    let geometry = match cam.geometry() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("warn: EVK4 geometry query failed ({e}); falling back to 1280x720");
            (1280, 720)
        }
    };
    let handle = std::thread::spawn(move || {
        evk_thread::run(cam, tx, cmd_rx, snapshot, rates, xform, record_tap, shutdown);
    });
    (handle, cmd_tx, geometry)
}

pub fn spawn_gpu_replay_evk(path: String, realtime: bool, tx: Sender<EvkMsg>) -> (JoinHandle<()>, (u32, u32)) {
    let file = Evt3RawFile::open(&path).expect("open raw");
    let geometry = (file.header.width, file.header.height);
    let handle = std::thread::spawn(move || {
        let mut decoder = match GpuEventDecoder::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("gpu replay: GpuEventDecoder init failed ({e}); no events will be emitted");
                let _ = tx.send(EvkMsg::Eof);
                return;
            }
        };
        let shift = time_shift_us(&file.words).unwrap_or(0);
        eprintln!("gpu replay: SDK-equivalent time shift = {shift} us");

        const SLICE_WORDS: usize = 32_000_000;
        const BATCH_EVENTS: usize = 65_536;

        let mut carry = CarryState::default();
        let mut first_t: Option<i64> = None;
        let t0_wall = Instant::now();
        for slice in file.words.chunks(SLICE_WORDS) {
            let (dev, triggers, _stats) = match decoder.decode_with_state(slice, 65_536, &mut carry) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("gpu replay: decode failed ({e}); ending stream early");
                    break;
                }
            };
            let events = match dev.download(decoder.stream()) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("gpu replay: download failed ({e}); ending stream early");
                    break;
                }
            };
            let mut trig_iter = triggers.into_iter().peekable();
            let mut i = 0usize;
            while i < events.len() {
                let end = (i + BATCH_EVENTS).min(events.len());
                let chunk = &events[i..end];
                let last_t = chunk.last().expect("chunk is non-empty").t_us;
                if realtime {
                    let ft = *first_t.get_or_insert(chunk[0].t_us);
                    let due = t0_wall + Duration::from_micros((last_t - ft).max(0) as u64);
                    if let Some(wait) = due.checked_duration_since(Instant::now()) {
                        std::thread::sleep(wait);
                    }
                }
                while let Some(tr) = trig_iter.peek() {
                    if tr.t_us > last_t {
                        break;
                    }
                    let tr = trig_iter.next().expect("peeked");
                    if tx.send(EvkMsg::Trigger(TriggerEvent { t_us: tr.t_us - shift, polarity: tr.polarity as i8 })).is_err() {
                        return;
                    }
                }
                let batch = EventBatch {
                    events: chunk.iter().map(|e| Event { t_us: e.t_us - shift, x: e.x, y: e.y, p: e.p as i8 }).collect(),
                };
                if tx.send(EvkMsg::Events(batch)).is_err() {
                    return;
                }
                i = end;
            }
            for tr in trig_iter {
                if tx.send(EvkMsg::Trigger(TriggerEvent { t_us: tr.t_us - shift, polarity: tr.polarity as i8 })).is_err() {
                    return;
                }
            }
        }
        let _ = tx.send(EvkMsg::Eof);
    });
    (handle, geometry)
}

pub fn spawn_replay_flir(
    path: String,
    fps: f64,
    realtime: bool,
    tx: Sender<FlirMsg>,
    last_error: Arc<Mutex<Option<String>>>,
) -> JoinHandle<()> {
    std::thread::spawn(move || {
        let mut cap = match VideoCapture::from_file(&path, CAP_ANY) {
            Ok(c) => c,
            Err(e) => {
                let msg = format!("replay: failed to open avi '{path}': {e}");
                eprintln!("{msg}");
                *last_error.lock().unwrap() = Some(msg);
                let _ = tx.send(FlirMsg::Eof);
                return;
            }
        };
        let interval = Duration::from_secs_f64(1.0 / fps);
        let t0 = Instant::now();
        let mut seq: u64 = 0;
        let mut mat = Mat::default();
        loop {
            match cap.read(&mut mat) {
                Ok(true) if !mat.empty() => {}
                Ok(_) => break,
                Err(e) => {
                    let msg = format!("replay: avi read error: {e}");
                    eprintln!("{msg}");
                    *last_error.lock().unwrap() = Some(msg);
                    break;
                }
            }
            if realtime {
                let due = t0 + interval * seq as u32;
                if let Some(wait) = due.checked_duration_since(Instant::now()) {
                    std::thread::sleep(wait);
                }
            }
            let (w, h) = (mat.cols() as u32, mat.rows() as u32);
            let data = match mat.data_bytes() {
                Ok(b) => b.to_vec(),
                Err(e) => {
                    let msg = format!("replay: avi frame read failed: {e}");
                    eprintln!("{msg}");
                    *last_error.lock().unwrap() = Some(msg);
                    break;
                }
            };
            let frame = RgbFrame {
                seq,
                t_cam_us: (seq as f64 * 1e6 / fps) as i64,
                w, h, data,
                format: PixelFormat::Bgr8,
            };
            if tx.send(FlirMsg::Frame(frame)).is_err() { return; }
            seq += 1;
        }
        let _ = tx.send(FlirMsg::Eof);
    })
}
