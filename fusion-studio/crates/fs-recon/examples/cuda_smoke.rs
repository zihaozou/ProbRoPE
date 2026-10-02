use std::time::Instant;

use fs_core::{Event, EventBatch, GrayImage};
use fs_recon::{CudaManifold, Reconstructor};

fn main() {
    let mut r = match CudaManifold::new(1280, 720) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("CudaManifold::new failed: {e}");
            std::process::exit(1);
        }
    };

    let mut img = GrayImage::new(1280, 720);
    let start = Instant::now();
    r.render_at(0, &mut img);
    let elapsed_us = start.elapsed().as_micros();

    let min = *img.data.iter().min().unwrap();
    let max = *img.data.iter().max().unwrap();
    let mean = img.data.iter().map(|&v| v as f64).sum::<f64>() / img.data.len() as f64;
    println!("fill:      min={min} max={max} mean={mean:.2} elapsed_us={elapsed_us}");
    assert!(img.data.iter().all(|&v| v == 128), "expected uniform mid-gray fill");

    r.reset();
    let mut events = Vec::with_capacity(100_000);
    let mut t = 0i64;
    for i in 0..50_000u32 {
        let x = 200 + (i % 100) as u16;
        let y = 200 + ((i / 100) % 100) as u16;
        events.push(Event { t_us: t, x, y, p: 1 });
        t += 1;
    }
    for i in 0..50_000u32 {
        let x = 900 + (i % 100) as u16;
        let y = 400 + ((i / 100) % 100) as u16;
        events.push(Event { t_us: t, x, y, p: 0 });
        t += 1;
    }
    let render_t = t;
    r.push_events(&EventBatch { events });

    let start = Instant::now();
    r.render_at(render_t, &mut img);
    let elapsed_us = start.elapsed().as_micros();

    let idx = |x: usize, y: usize| y * 1280 + x;
    let bright = img.data[idx(250, 250)];
    let dark = img.data[idx(950, 450)];
    let bg = img.data[idx(10, 10)];
    println!(
        "synthetic: bright={bright} dark={dark} bg={bg} elapsed_us={elapsed_us} (render_at @ 1280x720, 100k events)"
    );
    assert!(bright > 128, "bright square should be lighter than mid-gray, got {bright}");
    assert!(dark < 128, "dark square should be darker than mid-gray, got {dark}");
    assert_eq!(bg, 128, "untouched background should stay mid-gray, got {bg}");

    println!("PASS");

    for &iters in &[10u32, 50u32] {
        r.reset();
        r.iters = iters;
        let mut events = Vec::with_capacity(100_000);
        let mut t = 0i64;
        for i in 0..50_000u32 {
            let x = 200 + (i % 100) as u16;
            let y = 200 + ((i / 100) % 100) as u16;
            events.push(Event { t_us: t, x, y, p: 1 });
            t += 1;
        }
        for i in 0..50_000u32 {
            let x = 900 + (i % 100) as u16;
            let y = 400 + ((i / 100) % 100) as u16;
            events.push(Event { t_us: t, x, y, p: 0 });
            t += 1;
        }
        let render_t = t;
        r.push_events(&EventBatch { events });

        let start = Instant::now();
        r.render_at(render_t, &mut img);
        let elapsed_us = start.elapsed().as_micros();

        let bright = img.data[idx(250, 250)];
        let dark = img.data[idx(950, 450)];
        println!(
            "denoise:   iters={iters} bright={bright} dark={dark} elapsed_us={elapsed_us} (render_at @ 1280x720, 100k events, PD-TV)"
        );
    }

    println!("PASS");
}
