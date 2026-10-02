use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Condvar, Mutex};

use crate::board::{BoardConfig, Detection};
use crate::engine::{intrinsics_pass, stereo_pass, CalibSession, CamCalib, IntrinsicsOutcome, StereoOutcome};

struct IntrinsicsJob {
    generation: u64,
    cam: usize,
    cfg: BoardConfig,
    views: Vec<(u64, Detection)>,
    image_size: (u32, u32),
    warm: Option<CamCalib>,
}

struct StereoJob {
    generation: u64,
    cfg: BoardConfig,
    pairs: Vec<(u64, Detection, Detection)>,
    cam0: CamCalib,
    cam1: CamCalib,
    size0: (u32, u32),
}

#[derive(Default)]
struct Slots {
    cam: [Option<IntrinsicsJob>; 2],
    stereo: Option<StereoJob>,
    shutdown: bool,
}

impl Slots {
    fn any_queued(&self) -> bool {
        self.cam[0].is_some() || self.cam[1].is_some() || self.stereo.is_some()
    }
}

struct Shared {
    slots: Mutex<Slots>,
    cv: Condvar,
    running: AtomicBool,
}

enum ResultMsg {
    Intrinsics { cam: usize, generation: u64, outcome: IntrinsicsOutcome },
    Stereo { generation: u64, outcome: StereoOutcome },
}


pub struct BackgroundSession {
    inner: CalibSession,
    shared: Arc<Shared>,
    rx_results: mpsc::Receiver<ResultMsg>,
    gens: [u64; 3],
}

impl BackgroundSession {
    pub fn new(cfg: BoardConfig, cap: usize, size0: (u32, u32), size1: (u32, u32)) -> Self {
        let inner = CalibSession::new(cfg, cap, size0, size1);
        let shared = Arc::new(Shared {
            slots: Mutex::new(Slots::default()),
            cv: Condvar::new(),
            running: AtomicBool::new(false),
        });
        let (tx, rx) = mpsc::channel();
        {
            let shared = shared.clone();
            std::thread::Builder::new()
                .name("calib-background".into())
                .spawn(move || worker(&shared, &tx))
                .expect("spawn calib background thread");
        }
        BackgroundSession { inner, shared, rx_results: rx, gens: [0; 3] }
    }


    pub fn observe(&mut self, det0: Option<Detection>, det1: Option<Detection>) {
        let stereo_input = match (&det0, &det1) {
            (Some(d0), Some(d1)) => Some((d0.clone(), d1.clone())),
            _ => None,
        };

        if let Some(d0) = det0 {
            if self.inner.cam0.accept(d0) {
                self.post_intrinsics(0);
            }
        }
        if let Some(d1) = det1 {
            if self.inner.cam1.accept(d1) {
                self.post_intrinsics(1);
            }
        }

        if let Some((d0, d1)) = stereo_input {
            if self.inner.cam0.state.is_some() && self.inner.cam1.state.is_some() {
                self.inner.stereo.accept_pair(d0, d1);
                self.post_stereo();
            }
        }
    }

    pub fn pump(&mut self) -> usize {
        let mut applied = 0;
        while let Ok(msg) = self.rx_results.try_recv() {
            match msg {
                ResultMsg::Intrinsics { cam, generation, outcome } => {
                    if generation != self.gens[cam] {
                        continue;
                    }
                    let engine = if cam == 0 { &mut self.inner.cam0 } else { &mut self.inner.cam1 };
                    engine.apply_outcome(outcome);
                    applied += 1;
                    self.post_stereo();
                }
                ResultMsg::Stereo { generation, outcome } => {
                    if generation != self.gens[2] {
                        continue;
                    }
                    self.inner.stereo.apply_outcome(outcome);
                    applied += 1;
                }
            }
        }
        applied
    }

    pub fn busy(&self) -> bool {
        let slots = self.shared.slots.lock().unwrap();
        slots.any_queued() || self.shared.running.load(Ordering::SeqCst)
    }


    pub fn reset_stereo(&mut self) {
        self.gens[2] += 1;
        self.shared.slots.lock().unwrap().stereo = None;
        self.inner.reset_stereo();
    }

    pub fn reset_camera(&mut self, idx: usize) {
        if idx > 1 {
            return;
        }
        self.gens[idx] += 1;
        self.gens[2] += 1;
        {
            let mut slots = self.shared.slots.lock().unwrap();
            slots.cam[idx] = None;
            slots.stereo = None;
        }
        self.inner.reset_camera(idx);
    }

    pub fn reset_all(&mut self) {
        for g in &mut self.gens {
            *g += 1;
        }
        {
            let mut slots = self.shared.slots.lock().unwrap();
            slots.cam = [None, None];
            slots.stereo = None;
        }
        self.inner.reset_all();
    }

    pub fn session(&self) -> &CalibSession {
        &self.inner
    }

    fn post_intrinsics(&self, cam: usize) {
        let engine = if cam == 0 { &self.inner.cam0 } else { &self.inner.cam1 };
        let job = IntrinsicsJob {
            generation: self.gens[cam],
            cam,
            cfg: self.inner.cfg.clone(),
            views: engine.snapshot_views(),
            image_size: engine.image_size(),
            warm: engine.state.clone(),
        };
        let mut slots = self.shared.slots.lock().unwrap();
        slots.cam[cam] = Some(job);
        self.shared.cv.notify_one();
    }

    fn post_stereo(&self) {
        let (Some(cam0), Some(cam1)) = (self.inner.cam0.state.clone(), self.inner.cam1.state.clone()) else {
            return;
        };
        if self.inner.stereo.pair_count() == 0 {
            return;
        }
        let job = StereoJob {
            generation: self.gens[2],
            cfg: self.inner.cfg.clone(),
            pairs: self.inner.stereo.snapshot_pairs(),
            cam0,
            cam1,
            size0: self.inner.cam0.image_size(),
        };
        let mut slots = self.shared.slots.lock().unwrap();
        slots.stereo = Some(job);
        self.shared.cv.notify_one();
    }
}

impl Drop for BackgroundSession {
    fn drop(&mut self) {
        let mut slots = self.shared.slots.lock().unwrap();
        slots.shutdown = true;
        slots.cam = [None, None];
        slots.stereo = None;
        self.shared.cv.notify_all();
    }
}


fn worker(shared: &Shared, tx: &mpsc::Sender<ResultMsg>) {
    let stats = std::env::var("FS_CALIB_BG_STATS").is_ok();
    loop {
        let (cam_jobs, stereo_job) = {
            let mut slots = shared.slots.lock().unwrap();
            loop {
                if slots.shutdown {
                    return;
                }
                if slots.any_queued() {
                    break;
                }
                slots = shared.cv.wait(slots).unwrap();
            }
            shared.running.store(true, Ordering::SeqCst);
            ([slots.cam[0].take(), slots.cam[1].take()], slots.stereo.take())
        };

        for job in cam_jobs.into_iter().flatten() {
            let t0 = std::time::Instant::now();
            let n_views = job.views.len();
            let outcome = intrinsics_pass(&job.cfg, &job.views, job.image_size, job.warm.as_ref());
            if stats {
                eprintln!(
                    "[calib-bg] cam{} intrinsics: {} views, warm={}, {:.1}ms",
                    job.cam,
                    n_views,
                    job.warm.is_some(),
                    t0.elapsed().as_secs_f64() * 1000.0
                );
            }
            if tx.send(ResultMsg::Intrinsics { cam: job.cam, generation: job.generation, outcome }).is_err() {
                return;
            }
        }
        if let Some(job) = stereo_job {
            let t0 = std::time::Instant::now();
            let n_pairs = job.pairs.len();
            let outcome = stereo_pass(&job.pairs, &job.cam0, &job.cam1, &job.cfg, job.size0);
            if stats {
                eprintln!("[calib-bg] stereo: {} pairs, {:.1}ms", n_pairs, t0.elapsed().as_secs_f64() * 1000.0);
            }
            if tx.send(ResultMsg::Stereo { generation: job.generation, outcome }).is_err() {
                return;
            }
        }
        shared.running.store(false, Ordering::SeqCst);
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::BoardKind;
    use crate::sim::SimRig;
    use std::time::{Duration, Instant};

    fn test_board() -> BoardConfig {
        BoardConfig {
            kind: BoardKind::Checkerboard,
            inner_cols: 7,
            inner_rows: 6,
            rect_w_mm: 82.0,
            rect_h_mm: 98.0,
        }
    }

    fn wait_idle(sess: &mut BackgroundSession, timeout: Duration) {
        let start = Instant::now();
        loop {
            let applied = sess.pump();
            if applied == 0 && !sess.busy() && sess.pump() == 0 {
                return;
            }
            assert!(start.elapsed() < timeout, "background calibration did not go idle within {timeout:?}");
            std::thread::sleep(Duration::from_millis(2));
        }
    }


    #[test]
    fn background_session_recovers_ground_truth() {
        let board = test_board();
        let rig = SimRig::realistic();
        let views = rig.gen_views(&board, 30, 2);
        assert_eq!(views.len(), 30);

        let mut sess = BackgroundSession::new(board, 40, rig.size0, rig.size1);
        for (d0, d1) in views {
            sess.observe(Some(d0), Some(d1));
            wait_idle(&mut sess, Duration::from_secs(60));
        }

        let cam0 = sess.session().cam0.state.clone().expect("cam0 should have calibrated");
        let cam1 = sess.session().cam1.state.clone().expect("cam1 should have calibrated");
        let stereo = sess.session().stereo.state.clone().expect("stereo should have calibrated");

        for (label, calib, k_truth) in [("cam0", &cam0, &rig.k0), ("cam1", &cam1, &rig.k1)] {
            for (name, est, truth) in [
                ("fx", calib.k[0][0], k_truth[0][0]),
                ("fy", calib.k[1][1], k_truth[1][1]),
                ("cx", calib.k[0][2], k_truth[0][2]),
                ("cy", calib.k[1][2], k_truth[1][2]),
            ] {
                assert!((est - truth).abs() / truth < 0.01, "{label}.{name} off: {est} vs {truth}");
            }
        }

        let mut trace = 0.0;
        for i in 0..3 {
            for k in 0..3 {
                trace += rig.r[k][i] * stereo.r[k][i];
            }
        }
        let angle_deg = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0).acos().to_degrees();
        assert!(angle_deg < 0.2, "stereo rotation error too large: {angle_deg} deg");

        println!(
            "[background_session_recovers_ground_truth] cam0 fx={:.2} cam1 fx={:.2} stereo rot_err={:.4}deg pairs={}",
            cam0.k[0][0],
            cam1.k[0][0],
            angle_deg,
            sess.session().stereo.pair_count()
        );
    }


    #[test]
    fn background_coalesces_and_final_state_is_correct() {
        let board = test_board();
        let rig = SimRig::realistic();
        let views = rig.gen_views(&board, 30, 7);
        assert_eq!(views.len(), 30);

        let mut sess = BackgroundSession::new(board, 40, rig.size0, rig.size1);
        let mut accepted0 = 0usize;
        for (d0, _d1) in views {
            let before = sess.session().cam0.pool_len();
            sess.observe(Some(d0), None);
            if sess.session().cam0.pool_len() != before {
                accepted0 += 1;
            }
        }
        wait_idle(&mut sess, Duration::from_secs(60));

        let publishes = sess.session().cam0.history().count();
        println!("[background_coalesces] cam0 accepts={accepted0} publishes={publishes}");
        assert!(publishes >= 1, "at least the final snapshot must have been calibrated");
        assert!(
            publishes < accepted0,
            "expected coalescing to publish fewer results ({publishes}) than accepted keyframes ({accepted0})"
        );

        let calib = sess.session().cam0.state.clone().expect("cam0 should have calibrated");
        let fx_truth = rig.k0[0][0];
        assert!((calib.k[0][0] - fx_truth).abs() / fx_truth < 0.01, "fx off after coalesced publish: {} vs {}", calib.k[0][0], fx_truth);
    }


    #[test]
    fn reset_discards_stale_inflight_publish() {
        let board = test_board();
        let rig = SimRig::realistic();
        let views = rig.gen_views(&board, 12, 4);
        assert_eq!(views.len(), 12);

        let mut sess = BackgroundSession::new(board, 40, rig.size0, rig.size1);
        for (d0, d1) in &views {
            sess.observe(Some(d0.clone()), Some(d1.clone()));
        }
        assert!(sess.session().cam0.pool_len() > 0);

        sess.reset_camera(0);
        assert!(sess.session().cam0.state.is_none());
        assert_eq!(sess.session().cam0.pool_len(), 0);

        wait_idle(&mut sess, Duration::from_secs(60));
        assert!(sess.session().cam0.state.is_none(), "stale publish must not resurrect a reset camera");
        assert_eq!(sess.session().cam0.pool_len(), 0, "stale eviction verdicts must not touch a cleared pool");
        assert!(sess.session().stereo.state.is_none(), "camera reset cascades to stereo; stale stereo publish must stay discarded");

        assert!(sess.session().cam1.state.is_some(), "cam1 must be unaffected by cam0's reset");

        let board2 = test_board();
        for (d0, _d1) in rig.gen_views(&board2, 10, 5) {
            sess.observe(Some(d0), None);
            wait_idle(&mut sess, Duration::from_secs(60));
        }
        assert!(sess.session().cam0.state.is_some(), "cam0 should recalibrate from post-reset observations");
    }
}
