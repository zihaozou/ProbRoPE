
use std::collections::VecDeque;

use crate::SyncSource;


pub const LOST_CONSECUTIVE: u32 = 15;

pub const WINDOW_MS: u64 = 2_000;

pub const LOST_RATE: f64 = 0.5;

pub const DEGRADED_RATE: f64 = 0.9;

pub const MIN_SAMPLES: usize = 15;
pub const NO_FRAME_MS: u64 = 1_500;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SyncState {
    Healthy,
    Degraded,
    Lost,
}

#[derive(Clone, Copy, Debug)]
pub struct SyncHealth {
    pub state: SyncState,

    pub match_rate_window: Option<f64>,
    pub consecutive_non_matched: u32,
    pub window_samples: usize,

    pub since_ms: u64,

    pub desynced: bool,
}

pub struct SyncHealthMonitor {
    window: VecDeque<(u64, bool)>,
    consecutive_non_matched: u32,
    last_frame_ms: Option<u64>,
    now_ms: u64,
    state: SyncState,
    since_ms: u64,
    desynced: bool,
}

impl SyncHealthMonitor {
    pub fn new() -> Self {
        SyncHealthMonitor {
            window: VecDeque::new(),
            consecutive_non_matched: 0,
            last_frame_ms: None,
            now_ms: 0,
            state: SyncState::Healthy,
            since_ms: 0,
            desynced: false,
        }
    }


    pub fn push(&mut self, source: SyncSource, now_ms: u64) {
        let matched = source == SyncSource::Matched;
        if matched {
            self.consecutive_non_matched = 0;
        } else {
            self.consecutive_non_matched += 1;
        }
        self.window.push_back((now_ms, matched));
        self.last_frame_ms = Some(now_ms);
        self.advance(now_ms);
    }

    pub fn tick(&mut self, now_ms: u64) {
        self.advance(now_ms);
    }


    pub fn reset(&mut self, now_ms: u64) {
        self.window.clear();
        self.consecutive_non_matched = 0;
        self.last_frame_ms = Some(now_ms);
        self.now_ms = now_ms;
        self.state = SyncState::Healthy;
        self.since_ms = now_ms;
        self.desynced = false;
    }


    pub fn set_desynced(&mut self, desynced: bool) {
        self.desynced = desynced;
    }

    pub fn health(&self) -> SyncHealth {
        let (samples, matched) = self.window_counts();
        let match_rate_window =
            if samples < MIN_SAMPLES { None } else { Some(matched as f64 / samples as f64) };
        SyncHealth {
            state: self.state,
            match_rate_window,
            consecutive_non_matched: self.consecutive_non_matched,
            window_samples: samples,
            since_ms: self.since_ms,
            desynced: self.desynced,
        }
    }

    fn window_counts(&self) -> (usize, usize) {
        let matched = self.window.iter().filter(|(_, m)| *m).count();
        (self.window.len(), matched)
    }

    fn advance(&mut self, now_ms: u64) {
        self.now_ms = now_ms;
        let cutoff = now_ms.saturating_sub(WINDOW_MS);
        while matches!(self.window.front(), Some(&(t, _)) if t < cutoff) {
            self.window.pop_front();
        }
        let next = self.evaluate();
        if next != self.state {
            self.state = next;
            self.since_ms = now_ms;
        }
    }

    fn evaluate(&self) -> SyncState {
        let starved = self
            .last_frame_ms
            .is_some_and(|t| self.now_ms.saturating_sub(t) >= NO_FRAME_MS);
        if starved || self.consecutive_non_matched >= LOST_CONSECUTIVE {
            return SyncState::Lost;
        }
        let (samples, matched) = self.window_counts();
        if samples < MIN_SAMPLES {
            return self.state;
        }
        let rate = matched as f64 / samples as f64;
        if rate < LOST_RATE {
            SyncState::Lost
        } else if rate < DEGRADED_RATE {
            SyncState::Degraded
        } else {
            SyncState::Healthy
        }
    }
}

impl Default for SyncHealthMonitor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SyncSource::{Interpolated, Matched};


    #[test]
    fn all_matched_is_healthy() {
        let mut m = SyncHealthMonitor::new();
        for i in 0..30 {
            m.push(Matched, 1_000 + i * 33);
        }
        let h = m.health();
        assert_eq!(h.state, SyncState::Healthy);
        assert!((h.match_rate_window.unwrap() - 1.0).abs() < 1e-9);
        assert_eq!(h.consecutive_non_matched, 0);
    }

    #[test]
    fn fifteen_consecutive_non_matched_is_lost() {
        let mut m = SyncHealthMonitor::new();
        for i in 0..14 {
            m.push(Interpolated, 1_000 + i * 33);
        }
        assert_ne!(m.health().state, SyncState::Lost, "14 帧还不够");
        m.push(Interpolated, 1_000 + 14 * 33);
        assert_eq!(m.health().state, SyncState::Lost);
        assert_eq!(m.health().consecutive_non_matched, 15);
    }


    #[test]
    fn one_match_clears_the_consecutive_counter() {
        let mut m = SyncHealthMonitor::new();
        for i in 0..14 {
            m.push(Interpolated, 1_000 + i * 33);
        }
        m.push(Matched, 1_500);
        assert_eq!(m.health().consecutive_non_matched, 0);
    }

    #[test]
    fn alternating_half_broken_is_caught_by_the_window_rate() {
        let mut m = SyncHealthMonitor::new();
        for i in 0..40 {
            let src = if i % 3 == 0 { Matched } else { Interpolated };
            m.push(src, 1_000 + i * 33);
        }
        let h = m.health();
        assert!(h.consecutive_non_matched < LOST_CONSECUTIVE, "连续计数够不着");
        assert!(h.match_rate_window.unwrap() < LOST_RATE);
        assert_eq!(h.state, SyncState::Lost);
    }

    #[test]
    fn too_few_samples_never_reports_lost_by_rate() {
        let mut m = SyncHealthMonitor::new();
        for i in 0..3 {
            m.push(Interpolated, 1_000 + i * 33);
        }
        assert_ne!(m.health().state, SyncState::Lost);
    }

    #[test]
    fn window_is_trimmed_by_time_not_by_count() {
        let mut m = SyncHealthMonitor::new();
        for i in 0..20 {
            m.push(Interpolated, 1_000 + i * 33);
        }
        for i in 0..20 {
            m.push(Matched, 4_000 + i * 33);
        }
        let h = m.health();
        assert!((h.match_rate_window.unwrap() - 1.0).abs() < 1e-9, "旧样本必须已被裁掉");
        assert_eq!(h.window_samples, 20);
    }

    #[test]
    fn no_frames_for_1500ms_is_lost() {
        let mut m = SyncHealthMonitor::new();
        m.push(Matched, 1_000);
        m.tick(2_400);
        assert_eq!(m.health().state, SyncState::Healthy, "1.4 秒还不算");
        m.tick(2_600);
        assert_eq!(m.health().state, SyncState::Lost);
    }


    #[test]
    fn mildly_degraded_is_not_lost() {
        let mut m = SyncHealthMonitor::new();
        for i in 0..40 {
            let src = if i % 5 == 0 { Interpolated } else { Matched };
            m.push(src, 1_000 + i * 33);
        }
        let h = m.health();
        assert!(h.match_rate_window.unwrap() < DEGRADED_RATE);
        assert!(h.match_rate_window.unwrap() > LOST_RATE);
        assert_eq!(h.state, SyncState::Degraded);
    }

    #[test]
    fn reset_clears_history_and_the_no_frame_clock() {
        let mut m = SyncHealthMonitor::new();
        for i in 0..20 {
            m.push(Interpolated, 1_000 + i * 33);
        }
        assert_eq!(m.health().state, SyncState::Lost);
        m.reset(5_000);
        let h = m.health();
        assert_eq!(h.state, SyncState::Healthy);
        assert_eq!(h.consecutive_non_matched, 0);
        assert_eq!(h.window_samples, 0);
        m.tick(6_000);
        assert_eq!(m.health().state, SyncState::Healthy);
    }


    #[test]
    fn desynced_is_reported_independently_of_state() {
        let mut m = SyncHealthMonitor::new();
        m.push(Matched, 1_000);
        assert!(!m.health().desynced);
        m.set_desynced(true);
        let h = m.health();
        assert!(h.desynced);
        assert_eq!(h.state, SyncState::Healthy, "desynced 不改写统计判定");
    }

    #[test]
    fn rate_lost_never_flickers_healthy_while_starving() {
        let mut m = SyncHealthMonitor::new();
        for i in 0..40 {
            let src = if i % 3 == 0 { Matched } else { Interpolated };
            m.push(src, 1_000 + i * 50);
        }
        assert_eq!(m.health().state, SyncState::Lost, "先靠滑窗率判 Lost");

        let last_frame_ms = 1_000 + 39 * 50;
        let mut t = last_frame_ms;
        while t <= last_frame_ms + NO_FRAME_MS + 200 {
            m.tick(t);
            assert_eq!(m.health().state, SyncState::Lost, "t={t} 不许闪回 Healthy");
            t += 50;
        }
    }

    #[test]
    fn single_match_does_not_clear_rate_based_lost() {
        let mut m = SyncHealthMonitor::new();
        for i in 0..20 {
            m.push(Interpolated, 1_000 + i * 33);
        }
        assert_eq!(m.health().state, SyncState::Lost);

        m.push(Matched, 1_000 + 20 * 33);
        let h = m.health();
        assert_eq!(h.consecutive_non_matched, 0, "连续计数照常清零");
        assert_eq!(h.state, SyncState::Lost, "但滑窗判定不因一帧翻盘");
    }

    #[test]
    fn sustained_good_frames_recover_from_lost() {
        let mut m = SyncHealthMonitor::new();
        for i in 0..20 {
            m.push(Interpolated, 1_000 + i * 33);
        }
        assert_eq!(m.health().state, SyncState::Lost);

        for i in 0..100 {
            m.push(Matched, 1_700 + i * 33);
        }
        let h = m.health();
        assert_eq!(h.state, SyncState::Healthy, "坏样本出窗后应当自愈");
        assert_eq!(h.consecutive_non_matched, 0);
    }
}
