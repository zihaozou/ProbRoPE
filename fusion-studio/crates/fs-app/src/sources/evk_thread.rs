use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use crossbeam_channel::{Receiver, RecvTimeoutError, Sender};
use fs_core::bus::{send_latest, EvkMsg, EvkRawMsg};
use fs_core::{Event, EventBatch, TriggerEvent};
use metavision_sys::{MvCamera, MvEventCD, MvEventTrigger};

use crate::pipeline::SHUTDOWN_POLL;
use crate::settings::{AntiflickerMode, EvkSettingsSnapshot, TrailFilterType, CONNECT_REVISION, EVK_BIAS_NAMES};
use crate::stream_rates::StreamRates;
use crate::transform::{EventRecordTap, EventXform};

pub enum EvkCmd {
    SetBias { name: &'static str, value: i32 },
    SetErcEnable(bool),
    SetErcCdEventCount(u32),
    SetAntiflickerEnable(bool),
    SetAntiflickerBand { low_hz: u32, high_hz: u32 },
    SetAntiflickerMode(AntiflickerMode),
    SetTrailFilterEnable(bool),
    SetTrailFilterType(TrailFilterType),
    SetTrailFilterThresholdUs(u32),
}

struct Ctx {
    tx: Sender<EvkMsg>,
    rates: Arc<StreamRates>,

    xform: EventXform,

    record_tap: EventRecordTap,
}

unsafe extern "C" fn on_cd(e: *const MvEventCD, n: usize, user: *mut c_void) {
    let ctx = &*(user as *const Ctx);
    let evs = std::slice::from_raw_parts(e, n);
    let mut batch = EventBatch {
        events: evs.iter().map(|e| Event { t_us: e.t, x: e.x, y: e.y, p: e.p as i8 }).collect(),
    };
    ctx.xform.apply(&mut batch);
    ctx.rates.add_events(batch.events.len() as u64);
    ctx.record_tap.offer(&batch);
    let _ = ctx.tx.send(EvkMsg::Events(batch));
}

unsafe extern "C" fn on_trig(e: *const MvEventTrigger, n: usize, user: *mut c_void) {
    let ctx = &*(user as *const Ctx);
    for ev in std::slice::from_raw_parts(e, n) {
        let _ = ctx.tx.send(EvkMsg::Trigger(TriggerEvent { t_us: ev.t, polarity: ev.p as i8 }));
    }
}

unsafe extern "C" fn on_status(is_eof: i32, user: *mut c_void) {
    if is_eof != 0 {
        let ctx = &*(user as *const Ctx);
        let _ = ctx.tx.send(EvkMsg::Eof);
    }
}


fn read_evk_snapshot(cam: &MvCamera) -> EvkSettingsSnapshot {
    let mut s = EvkSettingsSnapshot::default();
    let avail = match cam.probe_facilities() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("evk settings: facility probe failed ({e}); panel will show all-unavailable");
            s.populated = true;
            s.revision = CONNECT_REVISION;
            return s;
        }
    };
    s.biases_available = avail.ll_biases;
    s.erc_available = avail.erc;
    s.af_available = avail.antiflicker;
    s.trail_available = avail.trail_filter;

    if s.biases_available {
        for (i, name) in EVK_BIAS_NAMES.iter().enumerate() {
            match (cam.get_bias(name), cam.bias_info(name)) {
                (Ok(v), Ok(info)) => {
                    s.biases[i].available = true;
                    s.biases[i].value = v;
                    s.biases[i].recommended = (info.min_recommended, info.max_recommended);
                    s.biases[i].allowed = (info.min_allowed, info.max_allowed);
                }
                (v, info) => {
                    eprintln!("evk settings: bias '{name}' read failed ({v:?} / {info:?})");
                }
            }
        }
    }
    if s.erc_available {
        s.erc_enabled = cam.erc_is_enabled().unwrap_or(false);
        s.erc_event_count = cam.erc_cd_event_count().unwrap_or(0);
        s.erc_count_range = cam.erc_cd_event_count_range().unwrap_or((0, 0));
        s.erc_period_us = cam.erc_count_period_us().unwrap_or(0);
    }
    if s.af_available {
        s.af_enabled = cam.af_is_enabled().unwrap_or(false);
        s.af_band_hz = cam.af_frequency_band().unwrap_or((0, 0));
        s.af_freq_range_hz = cam.af_supported_frequency_range().unwrap_or((0, 0));
        s.af_mode = cam.af_mode().ok().and_then(AntiflickerMode::from_shim).unwrap_or_default();
    }
    if s.trail_available {
        s.trail_enabled = cam.trail_is_enabled().unwrap_or(false);
        s.trail_type = cam.trail_type().ok().and_then(TrailFilterType::from_shim).unwrap_or_default();
        s.trail_available_types = cam.trail_available_types().unwrap_or(0);
        s.trail_threshold_us = cam.trail_threshold_us().unwrap_or(0);
        s.trail_threshold_range_us = cam.trail_threshold_range_us().unwrap_or((0, 0));
    }
    s.populated = true;
    s.revision = CONNECT_REVISION;
    s
}

fn apply_evk_cmd(cam: &MvCamera, cmd: EvkCmd, snapshot: &Arc<Mutex<EvkSettingsSnapshot>>) {
    enum ReadBack {
        Bias(Option<usize>, Option<i32>),
        Erc { enabled: Option<bool>, count: Option<u32> },
        Af { enabled: Option<bool>, band: Option<(u32, u32)>, mode: Option<AntiflickerMode> },
        Trail { enabled: Option<bool>, ty: Option<TrailFilterType>, threshold: Option<u32> },
    }

    let result: Result<(), String> = match &cmd {
        EvkCmd::SetBias { name, value } => cam.set_bias(name, *value),
        EvkCmd::SetErcEnable(on) => cam.erc_enable(*on),
        EvkCmd::SetErcCdEventCount(c) => cam.erc_set_cd_event_count(*c),
        EvkCmd::SetAntiflickerEnable(on) => cam.af_enable(*on),
        EvkCmd::SetAntiflickerBand { low_hz, high_hz } => cam.af_set_frequency_band(*low_hz, *high_hz),
        EvkCmd::SetAntiflickerMode(m) => cam.af_set_mode(m.to_shim()),
        EvkCmd::SetTrailFilterEnable(on) => cam.trail_enable(*on),
        EvkCmd::SetTrailFilterType(t) => cam.trail_set_type(t.to_shim()),
        EvkCmd::SetTrailFilterThresholdUs(us) => cam.trail_set_threshold_us(*us),
    };

    let rb = match &cmd {
        EvkCmd::SetBias { name, .. } => {
            ReadBack::Bias(EVK_BIAS_NAMES.iter().position(|n| n == name), cam.get_bias(name).ok())
        }
        EvkCmd::SetErcEnable(_) | EvkCmd::SetErcCdEventCount(_) => ReadBack::Erc {
            enabled: cam.erc_is_enabled().ok(),
            count: cam.erc_cd_event_count().ok(),
        },
        EvkCmd::SetAntiflickerEnable(_) | EvkCmd::SetAntiflickerBand { .. } | EvkCmd::SetAntiflickerMode(_) => {
            ReadBack::Af {
                enabled: cam.af_is_enabled().ok(),
                band: cam.af_frequency_band().ok(),
                mode: cam.af_mode().ok().and_then(AntiflickerMode::from_shim),
            }
        }
        EvkCmd::SetTrailFilterEnable(_) | EvkCmd::SetTrailFilterType(_) | EvkCmd::SetTrailFilterThresholdUs(_) => {
            ReadBack::Trail {
                enabled: cam.trail_is_enabled().ok(),
                ty: cam.trail_type().ok().and_then(TrailFilterType::from_shim),
                threshold: cam.trail_threshold_us().ok(),
            }
        }
    };

    let mut s = snapshot.lock().unwrap();
    match result {
        Ok(()) => s.last_error = None,
        Err(e) => s.last_error = Some(e),
    }
    match rb {
        ReadBack::Bias(Some(i), Some(v)) => s.biases[i].value = v,
        ReadBack::Bias(..) => {}
        ReadBack::Erc { enabled, count } => {
            if let Some(v) = enabled { s.erc_enabled = v; }
            if let Some(v) = count { s.erc_event_count = v; }
        }
        ReadBack::Af { enabled, band, mode } => {
            if let Some(v) = enabled { s.af_enabled = v; }
            if let Some(v) = band { s.af_band_hz = v; }
            if let Some(v) = mode { s.af_mode = v; }
        }
        ReadBack::Trail { enabled, ty, threshold } => {
            if let Some(v) = enabled { s.trail_enabled = v; }
            if let Some(v) = ty { s.trail_type = v; }
            if let Some(v) = threshold { s.trail_threshold_us = v; }
        }
    }
    s.revision += 1;
}


pub fn run(
    cam: MvCamera,
    tx: Sender<EvkMsg>,
    cmd_rx: Receiver<EvkCmd>,
    snapshot: Arc<Mutex<EvkSettingsSnapshot>>,
    rates: Arc<StreamRates>,
    xform: EventXform,
    record_tap: EventRecordTap,
    shutdown: Arc<AtomicBool>,
) {
    *snapshot.lock().unwrap() = read_evk_snapshot(&cam);

    let ctx = Box::leak(Box::new(Ctx { tx, rates, xform, record_tap }));
    let user = ctx as *mut Ctx as *mut c_void;
    unsafe {
        cam.set_cd_callback(on_cd, user).expect("cd cb");
        cam.set_trigger_callback(on_trig, user).expect("trig cb");
        cam.set_status_callback(on_status, user).expect("status cb");
    }
    cam.start().expect("evk start");
    loop {
        if shutdown.load(Ordering::Relaxed) {
            break;
        }
        match cmd_rx.recv_timeout(SHUTDOWN_POLL) {
            Ok(cmd) => apply_evk_cmd(&cam, cmd, &snapshot),
            Err(RecvTimeoutError::Timeout) => continue,
            Err(RecvTimeoutError::Disconnected) => break,
        }
    }
    let _ = cam.stop();
}


pub(crate) fn realign_to_words(carry: &mut Option<u8>, incoming: &[u8]) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(incoming.len() + 1);
    if let Some(b) = carry.take() {
        out.push(b);
    }
    out.extend_from_slice(incoming);
    if out.len() % 2 != 0 {
        *carry = out.pop();
    }
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

pub fn run_raw(
    cam: MvCamera,
    raw_tx: Sender<EvkRawMsg>,
    raw_rx_for_drop: Receiver<EvkRawMsg>,
    tx: Sender<EvkMsg>,
    cmd_rx: Receiver<EvkCmd>,
    snapshot: Arc<Mutex<EvkSettingsSnapshot>>,
    rates: Arc<StreamRates>,
    shutdown: Arc<AtomicBool>,
) {
    struct RawCtx {
        raw_tx: Sender<EvkRawMsg>,
        raw_rx_for_drop: Receiver<EvkRawMsg>,
        tx: Sender<EvkMsg>,
        byte_carry: Option<u8>,
        rates: Arc<StreamRates>,
    }

    unsafe extern "C" fn on_raw(data: *const u8, n: usize, user: *mut c_void) {
        let ctx = &mut *(user as *mut RawCtx);
        ctx.rates.add_event_bytes(n as u64);
        let bytes = std::slice::from_raw_parts(data, n);
        if let Some(whole) = realign_to_words(&mut ctx.byte_carry, bytes) {
            send_latest(&ctx.raw_tx, &ctx.raw_rx_for_drop, EvkRawMsg::Bytes(whole));
        }
    }

    unsafe extern "C" fn on_raw_status(is_eof: i32, user: *mut c_void) {
        if is_eof != 0 {
            let ctx = &*(user as *const RawCtx);
            let _ = ctx.tx.send(EvkMsg::Eof);
        }
    }

    *snapshot.lock().unwrap() = read_evk_snapshot(&cam);

    let ctx = Box::leak(Box::new(RawCtx { raw_tx, raw_rx_for_drop, tx, byte_carry: None, rates }));
    let user = ctx as *mut RawCtx as *mut c_void;
    unsafe {
        cam.set_raw_callback(on_raw, user).expect("raw cb");
        cam.set_status_callback(on_raw_status, user).expect("status cb");
    }
    let _ = ctx.raw_tx.send(EvkRawMsg::ResetState);
    cam.start().expect("evk start");
    loop {
        if shutdown.load(Ordering::Relaxed) {
            break;
        }
        match cmd_rx.recv_timeout(SHUTDOWN_POLL) {
            Ok(cmd) => apply_evk_cmd(&cam, cmd, &snapshot),
            Err(RecvTimeoutError::Timeout) => continue,
            Err(RecvTimeoutError::Disconnected) => break,
        }
    }
    let _ = cam.stop();
}

#[cfg(test)]
mod tests {
    use super::realign_to_words;

    #[test]
    fn realign_to_words_reassembles_odd_buffers() {
        let stream: Vec<u8> = (0u8..=41).collect();
        let splits = [1usize, 3, 1, 0, 5, 7, 2, 1, 1, 9, 12];
        assert_eq!(splits.iter().sum::<usize>(), stream.len());

        let mut carry = None;
        let mut emitted: Vec<u8> = Vec::new();
        let mut pos = 0usize;
        for &len in &splits {
            let buf = &stream[pos..pos + len];
            pos += len;
            if let Some(batch) = realign_to_words(&mut carry, buf) {
                assert_eq!(batch.len() % 2, 0, "every emitted batch must be whole words");
                emitted.extend_from_slice(&batch);
            }
        }
        assert_eq!(emitted, stream, "reassembled bytes must equal the original stream, in order");
        assert!(carry.is_none(), "even-length stream must leave no dangling byte");

        let mut carry = None;
        let mut emitted: Vec<u8> = Vec::new();
        for buf in [&stream[..7], &stream[7..15], &stream[15..21]] {
            if let Some(batch) = realign_to_words(&mut carry, buf) {
                assert_eq!(batch.len() % 2, 0);
                emitted.extend_from_slice(&batch);
            }
        }
        assert_eq!(emitted, &stream[..20]);
        assert_eq!(carry, Some(stream[20]));

        let mut carry = None;
        assert_eq!(realign_to_words(&mut carry, &[0xAB]), None);
        assert_eq!(carry, Some(0xAB));
        assert_eq!(realign_to_words(&mut carry, &[0xCD]), Some(vec![0xAB, 0xCD]));
        assert!(carry.is_none());
    }
}
