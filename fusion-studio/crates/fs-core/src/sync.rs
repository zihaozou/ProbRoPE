use std::collections::VecDeque;

use crate::clock::{ClockModel, ClockParams};
use crate::types::{RgbFrame, SyncSource, SyncedFrame, TriggerEvent};

#[derive(Clone, Copy)]
pub struct SyncConfig {
    pub fps: f64,

    pub gate_frac: f64,

    pub trigger_polarity: i8,

    pub max_pending: usize,
}

#[derive(Clone, Copy, Default, Debug)]
pub struct SyncStats {
    pub matched: u64,
    pub interpolated: u64,
    pub spurious: u64,
    pub drift_ppm: f64,
    pub offset_us: f64,
}

pub struct SyncEngine {
    cfg: SyncConfig,
    clock: ClockModel,
    triggers: VecDeque<i64>,
    frames: VecDeque<RgbFrame>,
    last_trigger_seen_us: Option<i64>,
    seed_t_evk: Option<i64>,
    stats: SyncStats,
}

impl SyncEngine {
    pub fn new(cfg: SyncConfig) -> Self {
        Self { cfg, clock: ClockModel::new(), triggers: VecDeque::new(),
               frames: VecDeque::new(), last_trigger_seen_us: None, seed_t_evk: None,
               stats: SyncStats::default() }
    }


    pub fn seed(&mut self, t_flir_us: i64, t_evk_us: i64) {
        self.clock.seed(t_flir_us, t_evk_us);
        self.seed_t_evk = Some(t_evk_us);
    }
    pub fn is_seeded(&self) -> bool { self.clock.is_seeded() }


    pub fn config(&self) -> SyncConfig { self.cfg }


    pub fn trigger_polarity(&self) -> i8 { self.cfg.trigger_polarity }

    pub fn stats(&self) -> SyncStats {
        let mut s = self.stats;
        s.drift_ppm = self.clock.drift_ppm();
        s.offset_us = self.clock.offset_us();
        s
    }


    pub fn clock_model(&self) -> ClockParams { self.clock.model() }

    pub fn push_trigger(&mut self, tr: TriggerEvent) -> Vec<SyncedFrame> {
        if tr.polarity != self.cfg.trigger_polarity { return vec![]; }
        if self.seed_t_evk.is_some_and(|s| tr.t_us <= s) { return vec![]; }
        self.last_trigger_seen_us = Some(tr.t_us);
        self.triggers.push_back(tr.t_us);
        self.drain()
    }

    pub fn push_frame(&mut self, f: RgbFrame) -> Vec<SyncedFrame> {
        self.frames.push_back(f);
        self.drain()
    }

    fn gate_us(&self) -> i64 { (self.cfg.gate_frac * 1e6 / self.cfg.fps) as i64 }

    fn drain(&mut self) -> Vec<SyncedFrame> {
        let mut out = Vec::new();
        if !self.clock.is_seeded() { return out; }
        let gate = self.gate_us();
        loop {
            let Some(front) = self.frames.front() else { break };
            let pred = self.clock.predict(front.t_cam_us);
            while matches!(self.triggers.front(), Some(&t) if t < pred - gate) {
                self.triggers.pop_front();
                self.stats.spurious += 1;
            }
            match self.triggers.front().copied() {
                Some(t) if t <= pred + gate => {
                    self.triggers.pop_front();
                    let frame = self.frames.pop_front().unwrap();
                    self.clock.update(frame.t_cam_us, t);
                    self.stats.matched += 1;
                    out.push(SyncedFrame { frame, t_evk_us: t, source: SyncSource::Matched });
                }
                _ => {
                    let evk_moved_on = self.last_trigger_seen_us.map_or(false, |t| t > pred + gate);
                    if evk_moved_on || self.frames.len() > self.cfg.max_pending {
                        let frame = self.frames.pop_front().unwrap();
                        self.stats.interpolated += 1;
                        out.push(SyncedFrame { frame, t_evk_us: pred, source: SyncSource::Interpolated });
                    } else {
                        break;
                    }
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::*;

    fn frame(seq: u64, t: i64) -> RgbFrame {
        RgbFrame { seq, t_cam_us: t, w: 1, h: 1, data: vec![0], format: PixelFormat::Gray8 }
    }
    fn trig(t: i64) -> TriggerEvent { TriggerEvent { t_us: t, polarity: 1 } }

    const FPS: f64 = 30.0;
    const DT: i64 = 33_333;

    fn engine() -> SyncEngine {
        let mut e = SyncEngine::new(SyncConfig { fps: FPS, gate_frac: 0.2, trigger_polarity: 1, max_pending: 120 });
        e.seed(0, 3000);
        e
    }

    #[test]
    fn clock_model_forwards_the_underlying_fit() {
        let mut e = engine();
        for i in 1..20i64 {
            e.push_trigger(trig(i * DT + 3000));
            e.push_frame(frame(i as u64, i * DT));
        }
        let params = e.clock_model();
        let stats = e.stats();
        assert_eq!(params.x0, 0, "x0 is the seed's t_flir");
        assert_eq!(params.drift * 1e6, stats.drift_ppm);
        assert_eq!(params.offset_us, stats.offset_us);
    }

    #[test]
    fn perfect_cadence_all_matched() {
        let mut e = engine();
        let mut out = vec![];
        for i in 1..50i64 {
            out.extend(e.push_trigger(trig(i * DT + 3000)));
            out.extend(e.push_frame(frame(i as u64, i * DT)));
        }
        assert_eq!(out.len(), 49);
        assert!(out.iter().all(|s| s.source == SyncSource::Matched));
        assert_eq!(e.stats().matched, 49);
        assert_eq!(out[0].t_evk_us, DT + 3000);
    }

    #[test]
    fn missing_trigger_interpolates() {
        let mut e = engine();
        let mut out = vec![];
        for i in 1..10i64 {
            if i != 5 { out.extend(e.push_trigger(trig(i * DT + 3000))); }
            out.extend(e.push_frame(frame(i as u64, i * DT)));
        }
        let s5 = out.iter().find(|s| s.frame.seq == 5).unwrap();
        assert_eq!(s5.source, SyncSource::Interpolated);
        assert!((s5.t_evk_us - (5 * DT + 3000)).abs() < 700);
        assert_eq!(e.stats().interpolated, 1);
        assert_eq!(e.stats().matched, 8);
    }

    #[test]
    fn spurious_trigger_rejected() {
        let mut e = engine();
        let mut out = vec![];
        for i in 1..10i64 {
            out.extend(e.push_trigger(trig(i * DT + 3000)));
            if i == 4 { out.extend(e.push_trigger(trig(i * DT + 3000 + 9000))); }
            out.extend(e.push_frame(frame(i as u64, i * DT)));
        }
        assert_eq!(e.stats().spurious, 1);
        assert_eq!(e.stats().matched, 9);
        assert!(out.iter().all(|s| s.source == SyncSource::Matched));
    }

    #[test]
    fn tracks_drift() {
        let mut e = engine();
        for i in 1..300i64 {
            let t_evk = (i * DT) as f64 * 1.0008 + 3000.0;
            e.push_trigger(trig(t_evk as i64));
            e.push_frame(frame(i as u64, i * DT));
        }
        let st = e.stats();
        assert_eq!(st.interpolated, 0, "drift must be absorbed by RLS, not interpolation");
        assert_eq!(st.spurious, 0);
        assert_eq!(st.matched, 299);
        assert!((st.drift_ppm - 800.0).abs() < 100.0, "drift_ppm={}", st.drift_ppm);
    }

    #[test]
    fn falling_edge_ignored() {
        let mut e = engine();
        e.push_trigger(TriggerEvent { t_us: DT + 3000, polarity: 0 });
        let out = e.push_frame(frame(1, DT));
        assert!(out.is_empty(), "polarity-0 trigger must not match; frame waits");
    }

    #[test]
    fn handshake_echo_trigger_dropped() {
        let mut e = engine();
        assert!(e.push_trigger(trig(3000)).is_empty(), "the seed's own edge");
        assert!(e.push_trigger(trig(2900)).is_empty(), "pre-seed stragglers too");
        let mut out = e.push_frame(frame(1, DT));
        assert!(out.is_empty(), "no echo queued -- the frame must wait for its REAL trigger");
        out.extend(e.push_trigger(trig(DT + 3000)));
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].source, SyncSource::Matched);
        assert_eq!(out[0].t_evk_us, DT + 3000);
        let st = e.stats();
        assert_eq!((st.matched, st.interpolated, st.spurious), (1, 0, 0));
    }
}
