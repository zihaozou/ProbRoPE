use std::collections::VecDeque;

use fs_core::{Event, EventBatch, GrayImage};

use crate::Reconstructor;


pub struct Accumulator {
    w: u32,
    h: u32,
    pub window_us: i64,
    pub gain: f32,
    events: VecDeque<Event>,
    cap: usize,
}

impl Accumulator {
    pub fn new(w: u32, h: u32) -> Self {
        Self { w, h, window_us: 33_000, gain: 40.0, events: VecDeque::new(), cap: 4_000_000 }
    }
}

impl Reconstructor for Accumulator {
    fn name(&self) -> &'static str { "accumulator" }

    fn push_events(&mut self, batch: &EventBatch) {
        self.events.extend(batch.events.iter().copied());
        while self.events.len() > self.cap { self.events.pop_front(); }
    }

    fn render_at(&mut self, t_us: i64, out: &mut GrayImage) {
        debug_assert_eq!((out.w, out.h), (self.w, self.h));
        let horizon = t_us - 4 * self.window_us;
        while matches!(self.events.front(), Some(e) if e.t_us < horizon) { self.events.pop_front(); }

        let mut acc = vec![0.0f32; (self.w * self.h) as usize];
        let lo = t_us - self.window_us;
        for e in &self.events {
            if e.t_us < lo || e.t_us > t_us { continue; }
            if (e.x as u32) < self.w && (e.y as u32) < self.h {
                let idx = e.y as usize * self.w as usize + e.x as usize;
                acc[idx] += if e.p == 1 { 1.0 } else { -1.0 };
            }
        }
        for (o, a) in out.data.iter_mut().zip(&acc) {
            *o = (128.0 + self.gain * a).clamp(0.0, 255.0) as u8;
        }
    }

    fn reset(&mut self) { self.events.clear(); }

    fn params_ui(&mut self, ui: &mut egui::Ui) {
        ui.add(egui::Slider::new(&mut self.window_us, 1_000..=500_000).logarithmic(true).text("window µs"));
        ui.add(egui::Slider::new(&mut self.gain, 1.0..=200.0).text("gain"));
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fs_core::{Event, EventBatch};

    fn ev(t: i64, x: u16, y: u16, p: i8) -> Event { Event { t_us: t, x, y, p } }

    #[test]
    fn accumulates_in_window_only() {
        let mut a = Accumulator::new(64, 64);
        a.window_us = 10_000;
        a.push_events(&EventBatch { events: vec![
            ev(1_000, 10, 20, 1),
            ev(45_000, 10, 20, 1),
            ev(46_000, 30, 40, 0),
        ]});
        let mut img = fs_core::GrayImage::new(64, 64);
        a.render_at(50_000, &mut img);
        let px = |x: usize, y: usize| img.data[y * 64 + x];
        assert!(px(10, 20) > 128, "positive event must brighten");
        assert!(px(30, 40) < 128, "negative event must darken");
        assert_eq!(px(5, 5), 128, "no-event pixel stays mid-gray");
    }

    #[test]
    fn reset_clears() {
        let mut a = Accumulator::new(8, 8);
        a.push_events(&EventBatch { events: vec![ev(100, 1, 1, 1)] });
        a.reset();
        let mut img = fs_core::GrayImage::new(8, 8);
        a.render_at(200, &mut img);
        assert_eq!(img.data[8 + 1], 128);
    }

    #[test]
    fn out_of_bounds_events_ignored() {
        let mut a = Accumulator::new(8, 8);
        a.push_events(&EventBatch { events: vec![ev(100, 200, 200, 1)] });
        let mut img = fs_core::GrayImage::new(8, 8);
        a.render_at(200, &mut img);
        assert!(img.data.iter().all(|&v| v == 128));
    }
}
