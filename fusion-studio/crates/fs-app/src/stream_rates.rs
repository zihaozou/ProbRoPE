
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use fs_core::{PixelFormat, RgbFrame};


pub const EVT3_BYTES_PER_EVENT: f64 = 4.83;


pub const EVENTS_BIN_BYTES_PER_EVENT: f64 = 8.0;

const SAMPLE_INTERVAL: Duration = Duration::from_millis(250);

const RATE_WINDOW: Duration = Duration::from_secs(5);

const MIN_SPAN: Duration = Duration::from_millis(750);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameSpec {
    pub w: u32,
    pub h: u32,
    pub format: PixelFormat,
    pub bytes: u64,
}

impl FrameSpec {

    pub fn format_name(&self) -> String {
        format!("{:?}", self.format)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RateBasis {

    RawEvt3Bytes,

    DecodedEvents,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EventRate {
    pub bytes_per_s: f64,
    pub basis: RateBasis,
}


#[derive(Default)]
pub struct StreamRates {
    frame: Mutex<Option<FrameSpec>>,

    event_bytes: AtomicU64,

    event_count: AtomicU64,
}

impl StreamRates {

    pub fn observe_frame(&self, f: &RgbFrame) {
        let spec = FrameSpec { w: f.w, h: f.h, format: f.format, bytes: f.data.len() as u64 };
        *self.frame.lock().unwrap_or_else(|p| p.into_inner()) = Some(spec);
    }


    pub fn add_event_bytes(&self, n: u64) {
        self.event_bytes.fetch_add(n, Ordering::Relaxed);
    }

    pub fn add_events(&self, n: u64) {
        self.event_count.fetch_add(n, Ordering::Relaxed);
    }

    pub fn frame_spec(&self) -> Option<FrameSpec> {
        *self.frame.lock().unwrap_or_else(|p| p.into_inner())
    }

    pub fn event_totals(&self) -> (u64, u64) {
        (self.event_bytes.load(Ordering::Relaxed), self.event_count.load(Ordering::Relaxed))
    }
}


#[derive(Default)]
pub struct RateWindow {
    samples: VecDeque<Sample>,
}

struct Sample {
    at: Instant,
    bytes: u64,
    events: u64,
}

impl RateWindow {
    pub fn observe(&mut self, at: Instant, bytes: u64, events: u64) {
        if self.samples.back().is_some_and(|s| at.duration_since(s.at) < SAMPLE_INTERVAL) {
            return;
        }
        self.samples.push_back(Sample { at, bytes, events });
        while self.samples.len() > 2 && at.duration_since(self.samples[1].at) > RATE_WINDOW {
            self.samples.pop_front();
        }
    }

    pub fn rate(&self) -> Option<EventRate> {
        let first = self.samples.front()?;
        let last = self.samples.back()?;
        let span = last.at.duration_since(first.at);
        if span < MIN_SPAN {
            return None;
        }
        let basis = if last.bytes > 0 {
            RateBasis::RawEvt3Bytes
        } else if last.events > 0 {
            RateBasis::DecodedEvents
        } else {
            return None;
        };
        let dt = span.as_secs_f64();
        let events_per_s = match basis {
            RateBasis::RawEvt3Bytes => (last.bytes - first.bytes) as f64 / EVT3_BYTES_PER_EVENT / dt,
            RateBasis::DecodedEvents => (last.events - first.events) as f64 / dt,
        };
        Some(EventRate { bytes_per_s: events_per_s * EVENTS_BIN_BYTES_PER_EVENT, basis })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(w: u32, h: u32, format: PixelFormat) -> RgbFrame {
        let bpp = match format {
            PixelFormat::Bgr8 | PixelFormat::Rgb8 => 3,
            PixelFormat::Bayer8 | PixelFormat::Gray8 => 1,
        };
        RgbFrame { seq: 0, t_cam_us: 0, w, h, data: vec![0u8; (w * h * bpp) as usize], format }
    }

    #[test]
    fn frame_spec_reports_the_bytes_that_will_be_written() {
        let r = StreamRates::default();
        assert_eq!(r.frame_spec(), None, "还没见过同步帧时必须是 None,不能是一个零尺寸的假值");
        r.observe_frame(&frame(1280, 1024, PixelFormat::Bayer8));
        let s = r.frame_spec().unwrap();
        assert_eq!((s.w, s.h, s.bytes), (1280, 1024, 1_310_720));
        assert_eq!(s.format_name(), "Bayer8");
        r.observe_frame(&frame(640, 480, PixelFormat::Bgr8));
        let s = r.frame_spec().unwrap();
        assert_eq!((s.w, s.h, s.bytes), (640, 480, 921_600));
        assert_eq!(s.format_name(), "Bgr8");
    }

    #[test]
    fn no_reading_until_the_window_has_enough_history() {
        let t0 = Instant::now();
        let mut w = RateWindow::default();
        w.observe(t0, 0, 0);
        assert_eq!(w.rate(), None, "只有一个采样点算不出速率");
        w.observe(t0 + Duration::from_millis(300), 3_000_000, 0);
        assert_eq!(w.rate(), None, "跨度不足 MIN_SPAN 时不许给读数");
    }

    #[test]
    fn an_unobserved_event_stream_is_unknown_not_zero() {
        let t0 = Instant::now();
        let mut w = RateWindow::default();
        w.observe(t0, 0, 0);
        w.observe(t0 + Duration::from_secs(2), 0, 0);
        assert_eq!(w.rate(), None);
    }


    #[test]
    fn raw_bytes_are_converted_to_the_events_bin_disk_cost() {
        let t0 = Instant::now();
        let mut w = RateWindow::default();
        w.observe(t0, 0, 0);
        w.observe(t0 + Duration::from_secs(2), 46_600_000, 0);
        let r = w.rate().unwrap();
        assert_eq!(r.basis, RateBasis::RawEvt3Bytes);
        let want = 23_300_000.0 / EVT3_BYTES_PER_EVENT * EVENTS_BIN_BYTES_PER_EVENT;
        assert!((r.bytes_per_s - want).abs() < 1.0, "{} != {want}", r.bytes_per_s);
        assert!(r.bytes_per_s > 23_300_000.0, "落盘口径必然大于线上口径 —— 相等说明换算被跳过了");
    }


    #[test]
    fn decoded_events_are_converted_with_the_events_bin_cost() {
        let t0 = Instant::now();
        let mut w = RateWindow::default();
        w.observe(t0, 0, 0);
        w.observe(t0 + Duration::from_secs(2), 0, 4_000_000);
        let r = w.rate().unwrap();
        assert_eq!(r.basis, RateBasis::DecodedEvents);
        assert!((r.bytes_per_s - 2_000_000.0 * EVENTS_BIN_BYTES_PER_EVENT).abs() < 1.0, "{}", r.bytes_per_s);
    }

    #[test]
    fn the_reading_follows_the_current_rate_not_the_session_average() {
        let t0 = Instant::now();
        let mut w = RateWindow::default();
        let mut events = 0u64;
        for i in 0..=24u64 {
            events = i * 312_500;
            w.observe(t0 + Duration::from_millis(250 * i), 0, events);
        }
        assert!((w.rate().unwrap().bytes_per_s - 10_000_000.0).abs() < 200_000.0);
        for i in 1..=24u64 {
            events += 1_250_000;
            w.observe(t0 + Duration::from_millis(6000 + 250 * i), 0, events);
        }
        let r = w.rate().unwrap().bytes_per_s;
        assert!((r - 40_000_000.0).abs() < 2_000_000.0, "窗口没跟上当前码率:{r}");
    }

    #[test]
    fn per_frame_calls_are_throttled_but_still_fill_the_window() {
        let t0 = Instant::now();
        let mut w = RateWindow::default();
        for i in 0..600u64 {
            let at = t0 + Duration::from_millis(16 * i);
            w.observe(at, 0, (16 * i) * 2_500);
        }
        assert!(w.samples.len() <= 24, "250ms 节流之后不该攒下 {} 个采样点", w.samples.len());
        let r = w.rate().unwrap().bytes_per_s;
        assert!((r - 20_000_000.0).abs() < 1_000_000.0, "{r}");
    }
}
