use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use crossbeam_channel::{Receiver, Sender};
use fs_core::bus::send_latest;
use fs_core::{PixelFormat, RgbFrame};


pub fn run(
    rx: Receiver<RgbFrame>,
    tx: Sender<egui::ColorImage>,
    tx_rx: Receiver<egui::ColorImage>,
    last_error: Arc<Mutex<Option<String>>>,
) {
    static FORMAT_LOGGED: AtomicBool = AtomicBool::new(false);
    static SIZE_LOGGED: AtomicBool = AtomicBool::new(false);

    let report = |logged: &AtomicBool, msg: String| {
        if !logged.swap(true, Ordering::Relaxed) {
            eprintln!("preview: {msg}");
        }
        *last_error.lock().unwrap() = Some(msg);
    };

    while let Ok(f) = rx.recv() {
        if f.format != PixelFormat::Rgb8 {
            report(&FORMAT_LOGGED, format!("非 Rgb8 帧({:?})到达预览 —— 上游变换级 bug;丢帧", f.format));
            continue;
        }
        let need = (f.w * f.h * 3) as usize;
        if f.data.len() < need {
            report(&SIZE_LOGGED, format!("Rgb8 帧太小({} < {need});丢帧", f.data.len()));
            continue;
        }
        let img = egui::ColorImage::from_rgb([f.w as usize, f.h as usize], &f.data[..need]);
        send_latest(&tx, &tx_rx, img);
    }
}
