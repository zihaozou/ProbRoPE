use std::ffi::c_void;
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};

static N_EVENTS: AtomicU64 = AtomicU64::new(0);
static N_TRIGGERS: AtomicU64 = AtomicU64::new(0);
static LAST_T: AtomicI64 = AtomicI64::new(0);

unsafe extern "C" fn on_cd(_e: *const metavision_sys::MvEventCD, n: usize, _u: *mut c_void) {
    N_EVENTS.fetch_add(n as u64, Ordering::Relaxed);
}
unsafe extern "C" fn on_trig(e: *const metavision_sys::MvEventTrigger, n: usize, _u: *mut c_void) {
    let evs = std::slice::from_raw_parts(e, n);
    for ev in evs {
        if ev.p == 1 { N_TRIGGERS.fetch_add(1, Ordering::Relaxed); }
        LAST_T.store(ev.t, Ordering::Relaxed);
    }
}

fn main() {
    let path = std::env::args().nth(1).expect("usage: read_raw <file.raw>");
    let cam = metavision_sys::MvCamera::open_file(&path, false).unwrap();
    let (w, h) = cam.geometry().unwrap();
    println!("geometry {w}x{h}");
    unsafe {
        cam.set_cd_callback(on_cd, std::ptr::null_mut()).unwrap();
        cam.set_trigger_callback(on_trig, std::ptr::null_mut()).unwrap();
    }
    cam.start().unwrap();
    let mut prev = u64::MAX;
    loop {
        std::thread::sleep(std::time::Duration::from_millis(300));
        let n = N_EVENTS.load(Ordering::Relaxed);
        if n == prev { break; }
        prev = n;
    }
    cam.stop().unwrap();
    println!("events={} rising_triggers={} last_trigger_t_us={}",
             N_EVENTS.load(Ordering::Relaxed), N_TRIGGERS.load(Ordering::Relaxed),
             LAST_T.load(Ordering::Relaxed));
}
