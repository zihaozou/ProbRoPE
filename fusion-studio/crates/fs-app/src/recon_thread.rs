use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crossbeam_channel::{select, Receiver, RecvTimeoutError, Sender};
use fs_core::bus::send_latest;
use fs_core::{EventBatch, GrayImage};
use fs_recon::{Accumulator, CudaManifold, Normalization, Reconstructor};

use crate::decode_stage::DecodedBatch;

#[derive(Clone, Copy, Debug, Default)]
pub struct ReconStats {
    pub render_ms_avg: f64,
    pub paired_coverage_pct: f64,

    pub render_overloaded: bool,
    pub coverage_overloaded: bool,
    pub overloaded: bool,

    pub effective_iters: u32,

    pub median_active: bool,
    pub events_dropped: u64,
}


pub struct ReconOutputs {

    pub img: (Sender<egui::ColorImage>, Receiver<egui::ColorImage>),
    pub gray: (Sender<(GrayImage, i64)>, Receiver<(GrayImage, i64)>),
    pub stats: Arc<Mutex<ReconStats>>,
}


pub struct DecodedIntake {

    pub rx_dev: Receiver<DecodedBatch>,

    pub rx_host: Receiver<EventBatch>,
}


pub enum EvkIntake {
    Sdk(Receiver<EventBatch>),
    GpuRaw(DecodedIntake),
}


fn consume_device_batch(recon: &mut dyn Reconstructor, last_t: &mut i64, b: DecodedBatch) {
    if b.events.len > 0 {
        recon.push_events_device(&b.events, &b.stream);
    }
    *last_t = b.t_latest_us;
}

fn consume_host_batch(recon: &mut dyn Reconstructor, last_t: &mut i64, batch: EventBatch) {
    if let Some(e) = batch.events.last() {
        *last_t = e.t_us;
    }
    recon.push_events(&batch);
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ReconChoice {
    CudaManifold,
    Accumulator,
}


pub enum ParamsMsg {

    SetIters(u32),

    SetTauLeak(f32),

    SetMedian(bool),

    SetWindowUs(i64),
}

fn apply_params(recon: &mut dyn Reconstructor, msg: ParamsMsg) {
    match msg {
        ParamsMsg::SetIters(v) => {
            if let Some(c) = recon.as_any_mut().downcast_mut::<CudaManifold>() {
                c.iters = v;
            }
        }
        ParamsMsg::SetTauLeak(v) => {
            if let Some(c) = recon.as_any_mut().downcast_mut::<CudaManifold>() {
                c.tau_leak_us = v;
            }
        }
        ParamsMsg::SetMedian(v) => {
            if let Some(c) = recon.as_any_mut().downcast_mut::<CudaManifold>() {
                c.median_filter = v;
            }
        }
        ParamsMsg::SetWindowUs(v) => {
            if let Some(a) = recon.as_any_mut().downcast_mut::<Accumulator>() {
                a.window_us = v;
            }
        }
    }
}

pub struct ReconSwap {
    pub recon: Box<dyn Reconstructor>,

    pub dims: (u32, u32),
}

pub struct ReconCtrl {

    pub rx_swap: Receiver<ReconSwap>,
    pub rx_params: Receiver<ParamsMsg>,

    pub shutdown: Arc<AtomicBool>,

    pub events_dropped: Arc<AtomicU64>,

    pub auto_degrade: Arc<AtomicBool>,
    pub paired_only: Arc<AtomicBool>,
}

#[derive(Clone)]
pub struct ReconHandles {
    pub tx_swap: Sender<ReconSwap>,
    pub tx_params: Sender<ParamsMsg>,
    pub paired_only: Arc<AtomicBool>,
    pub auto_degrade: Arc<AtomicBool>,
    pub stats: Arc<Mutex<ReconStats>>,
    pub initial_choice: ReconChoice,
    pub initial_error: Option<String>,
}


pub fn build_cuda_manifold(w: u32, h: u32) -> Result<CudaManifold, String> {
    let mut m = CudaManifold::new(w, h)?;
    m.iters = 2;
    m.tau_leak_us = 2_000_000.0;
    m.tonemap_scale = 400.0;
    m.normalization = Normalization::TileLocal;
    m.median_filter = true;
    Ok(m)
}


pub fn build_accumulator(w: u32, h: u32) -> Accumulator {
    let mut a = Accumulator::new(w, h);
    a.window_us = 150_000;
    a
}

pub fn default_reconstructor(w: u32, h: u32) -> (Box<dyn Reconstructor>, ReconChoice, Option<String>) {
    match build_cuda_manifold(w, h) {
        Ok(m) => (Box::new(m), ReconChoice::CudaManifold, None),
        Err(e) => (Box::new(build_accumulator(w, h)), ReconChoice::Accumulator, Some(e)),
    }
}

const RENDER_MS_EMA_ALPHA: f64 = 0.2;

const COVERAGE_WINDOW: Duration = Duration::from_secs(3);

const COVERAGE_STARTUP_GRACE: Duration = Duration::from_secs(1);

const DEGRADE_STEP_COOLDOWN: Duration = Duration::from_secs(2);


const RECOVERY_HEADROOM_FRAC: f64 = 0.5;

fn cuda_manifold_knobs(recon: &mut dyn Reconstructor) -> Option<(u32, bool)> {
    recon.as_any_mut().downcast_mut::<CudaManifold>().map(|c| (c.iters, c.median_filter))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DegradeAction {
    None,
    DegradeIters(u32),
    DegradeMedianOff,
    RecoverIters(u32),
    RecoverMedianOn,
}

fn decide_degrade_action(
    render_overloaded: bool,
    render_ms_avg: f64,
    frame_interval_ms: f64,
    current_iters: u32,
    current_median: bool,
    target_iters: u32,
    target_median: bool,
) -> DegradeAction {
    if render_overloaded {
        if current_iters > 0 {
            DegradeAction::DegradeIters(current_iters / 2)
        } else if current_median {
            DegradeAction::DegradeMedianOff
        } else {
            DegradeAction::None
        }
    } else if render_ms_avg < RECOVERY_HEADROOM_FRAC * frame_interval_ms {
        if current_iters < target_iters {
            let new_iters = if current_iters == 0 { target_iters.min(1) } else { (current_iters * 2).min(target_iters) };
            DegradeAction::RecoverIters(new_iters)
        } else if !current_median && target_median {
            DegradeAction::RecoverMedianOn
        } else {
            DegradeAction::None
        }
    } else {
        DegradeAction::None
    }
}

#[cfg(test)]
mod degrade_tests {
    use super::*;

    #[test]
    fn coverage_only_overload_does_not_degrade() {
        let action = decide_degrade_action(
            false,
            5.0,
            33.3,
            2,
            true,
            2,
            true,
        );
        assert_eq!(action, DegradeAction::None, "already at target with headroom to spare -- nothing to do, and definitely no degrade");
    }


    #[test]
    fn coverage_only_overload_still_allows_recovery() {
        let action = decide_degrade_action(false, 5.0, 33.3, 0, false, 2, true);
        assert_eq!(action, DegradeAction::RecoverIters(1), "render is healthy -- should recover iters regardless of coverage");
    }

    #[test]
    fn render_overloaded_degrades_iters_then_median() {
        assert_eq!(decide_degrade_action(true, 100.0, 33.3, 4, true, 4, true), DegradeAction::DegradeIters(2));
        assert_eq!(decide_degrade_action(true, 100.0, 33.3, 0, true, 4, true), DegradeAction::DegradeMedianOff);
        assert_eq!(
            decide_degrade_action(true, 100.0, 33.3, 0, false, 4, true),
            DegradeAction::None,
            "already fully degraded -- nothing left to do"
        );
    }

    #[test]
    fn mid_budget_is_a_no_op() {
        let action = decide_degrade_action(false, 25.0, 33.3, 1, false, 4, true);
        assert_eq!(action, DegradeAction::None, "75% of budget is neither overloaded nor comfortably under 50% -- hold steady");
    }

    #[test]
    fn recovery_caps_at_target_then_recovers_median() {
        assert_eq!(decide_degrade_action(false, 1.0, 33.3, 1, false, 4, true), DegradeAction::RecoverIters(2));
        assert_eq!(decide_degrade_action(false, 1.0, 33.3, 3, false, 4, true), DegradeAction::RecoverIters(4), "should cap at target_iters, not overshoot");
        assert_eq!(decide_degrade_action(false, 1.0, 33.3, 4, false, 4, true), DegradeAction::RecoverMedianOn);
        assert_eq!(
            decide_degrade_action(false, 1.0, 33.3, 4, true, 4, true),
            DegradeAction::None,
            "already fully recovered to target -- nothing left to do"
        );
    }
}

pub fn run(
    intake: EvkIntake,
    rx_req: Receiver<i64>,
    ctrl: ReconCtrl,
    out: ReconOutputs,
    mut recon: Box<dyn Reconstructor>,
    mut w: u32,
    mut h: u32,
    tuning: crate::settings::SharedTuning,
) {
    let (sdk_rx, gpu_raw): (Option<Receiver<EventBatch>>, Option<DecodedIntake>) = match intake {
        EvkIntake::Sdk(rx) => (Some(rx), None),
        EvkIntake::GpuRaw(d) => {
            eprintln!("recon: decoded-batch intake active(事件来自独立解码级)");
            (None, Some(d))
        }
    };
    let (tx_img, tx_img_rx) = out.img;
    let (tx_gray, tx_gray_rx) = out.gray;
    let mut img = GrayImage::new(w, h);
    let mut last_t: i64 = 0;
    let mut last_render = Instant::now();
    let mut last_request: Option<Instant> = None;

    let stats_diag = std::env::var("FS_RECON_STATS").is_ok();
    if stats_diag {
        eprintln!("recon: FS_RECON_STATS enabled -- printing every 10 paired renders");
    }

    let mut render_count: u64 = 0;
    let mut render_ms_avg: f64 = 0.0;
    let mut render_times: VecDeque<Instant> = VecDeque::new();
    let mut first_render_at: Option<Instant> = None;
    let mut last_degrade_change: Option<Instant> = None;
    let mut coalesced_requests: u64 = 0;
    let mut last_serviced_t: Option<i64> = None;
    let (mut target_iters, mut target_median) = cuda_manifold_knobs(&mut *recon).unwrap_or((2, true));

    loop {
        if ctrl.shutdown.load(Ordering::Relaxed) {
            return;
        }
        let mut swap_to = None;
        while let Ok(nc) = ctrl.rx_swap.try_recv() {
            swap_to = Some(nc);
        }
        if let Some(ReconSwap { recon: mut nc, dims }) = swap_to {
            nc.reset();
            recon = nc;
            if dims != (w, h) {
                (w, h) = dims;
                img = GrayImage::new(w, h);
            }
            (target_iters, target_median) = cuda_manifold_knobs(&mut *recon).unwrap_or((2, true));
            last_degrade_change = None;
        }

        while let Ok(msg) = ctrl.rx_params.try_recv() {
            match &msg {
                ParamsMsg::SetIters(v) => target_iters = *v,
                ParamsMsg::SetMedian(v) => target_median = *v,
                ParamsMsg::SetTauLeak(_) | ParamsMsg::SetWindowUs(_) => {}
            }
            apply_params(&mut *recon, msg);
        }

        if let Some(rx_events) = &sdk_rx {
            match rx_events.recv_timeout(Duration::from_millis(20)) {
                Ok(batch) => {
                    if let Some(e) = batch.events.last() {
                        last_t = e.t_us;
                    }
                    recon.push_events(&batch);
                    while let Ok(batch) = rx_events.try_recv() {
                        if let Some(e) = batch.events.last() {
                            last_t = e.t_us;
                        }
                        recon.push_events(&batch);
                    }
                }
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => break,
            }
        } else if let Some(g) = gpu_raw.as_ref() {
            let mut disconnected = false;
            select! {
                recv(g.rx_dev) -> m => match m {
                    Ok(b) => consume_device_batch(&mut *recon, &mut last_t, b),
                    Err(_) => disconnected = true,
                },
                recv(g.rx_host) -> m => match m {
                    Ok(batch) => consume_host_batch(&mut *recon, &mut last_t, batch),
                    Err(_) => disconnected = true,
                },
                default(Duration::from_millis(20)) => {}
            }
            if disconnected {
                break;
            }
            while let Ok(b) = g.rx_dev.try_recv() {
                consume_device_batch(&mut *recon, &mut last_t, b);
            }
            while let Ok(batch) = g.rx_host.try_recv() {
                consume_host_batch(&mut *recon, &mut last_t, batch);
            }
        }

        let mut requested_t = None;
        let mut seen_this_tick = 0u32;
        while let Ok(t) = rx_req.try_recv() {
            requested_t = Some(t);
            seen_this_tick += 1;
        }
        if seen_this_tick > 1 {
            coalesced_requests += (seen_this_tick - 1) as u64;
        }
        if let Some(t) = requested_t {
            let fps = tuning.fps();
            let frame_interval_ms = if fps > 0.0 { 1000.0 / fps } else { 33.3 };

            last_request = Some(Instant::now());
            let request_gap_us = last_serviced_t.map(|lt| t - lt);
            last_serviced_t = Some(t);

            let r0 = Instant::now();
            recon.render_at(t, &mut img);
            let render_ms = r0.elapsed().as_secs_f64() * 1000.0;

            let ci = egui::ColorImage::from_gray([w as usize, h as usize], &img.data);
            send_latest(&tx_img, &tx_img_rx, ci);
            send_latest(&tx_gray, &tx_gray_rx, (img.clone(), t));
            last_render = Instant::now();

            render_count += 1;
            render_ms_avg =
                if render_count == 1 { render_ms } else { RENDER_MS_EMA_ALPHA * render_ms + (1.0 - RENDER_MS_EMA_ALPHA) * render_ms_avg };

            let now = Instant::now();
            let first_render_at = *first_render_at.get_or_insert(now);
            render_times.push_back(now);
            while let Some(&front) = render_times.front() {
                if now.duration_since(front) > COVERAGE_WINDOW {
                    render_times.pop_front();
                } else {
                    break;
                }
            }
            let elapsed_since_first = now.duration_since(first_render_at).as_secs_f64();
            let paired_coverage_pct = if elapsed_since_first < COVERAGE_STARTUP_GRACE.as_secs_f64() {
                100.0
            } else {
                let window_secs = elapsed_since_first.min(COVERAGE_WINDOW.as_secs_f64());
                (100.0 * render_times.len() as f64 / (window_secs * fps.max(0.001))).min(999.0)
            };

            let render_overloaded = render_ms_avg > frame_interval_ms;
            let coverage_overloaded = paired_coverage_pct < 80.0;
            let overloaded = render_overloaded || coverage_overloaded;

            let (mut effective_iters, mut median_active) = cuda_manifold_knobs(&mut *recon).unwrap_or((0, false));

            if ctrl.auto_degrade.load(Ordering::Relaxed) {
                let cooldown_elapsed = last_degrade_change.map_or(true, |t| t.elapsed() >= DEGRADE_STEP_COOLDOWN);
                if cooldown_elapsed {
                    if let Some(c) = recon.as_any_mut().downcast_mut::<CudaManifold>() {
                        let action = decide_degrade_action(
                            render_overloaded,
                            render_ms_avg,
                            frame_interval_ms,
                            c.iters,
                            c.median_filter,
                            target_iters,
                            target_median,
                        );
                        match action {
                            DegradeAction::DegradeIters(new_iters) => {
                                eprintln!(
                                    "recon: RENDER OVERLOADED (render_ms_avg={render_ms_avg:.1}ms > budget={frame_interval_ms:.1}ms) -- degrading iters {} -> {new_iters}",
                                    c.iters
                                );
                                c.iters = new_iters;
                                last_degrade_change = Some(now);
                            }
                            DegradeAction::DegradeMedianOff => {
                                eprintln!(
                                    "recon: RENDER OVERLOADED (render_ms_avg={render_ms_avg:.1}ms > budget={frame_interval_ms:.1}ms) -- disabling median filter"
                                );
                                c.median_filter = false;
                                last_degrade_change = Some(now);
                            }
                            DegradeAction::RecoverIters(new_iters) => {
                                eprintln!("recon: comfortably under budget -- recovering iters {} -> {new_iters}", c.iters);
                                c.iters = new_iters;
                                last_degrade_change = Some(now);
                            }
                            DegradeAction::RecoverMedianOn => {
                                eprintln!("recon: comfortably under budget -- re-enabling median filter");
                                c.median_filter = true;
                                last_degrade_change = Some(now);
                            }
                            DegradeAction::None => {}
                        }
                        effective_iters = c.iters;
                        median_active = c.median_filter;
                    }
                }
            }

            let events_dropped = ctrl.events_dropped.load(Ordering::Relaxed);
            *out.stats.lock().unwrap() = ReconStats {
                render_ms_avg,
                paired_coverage_pct,
                render_overloaded,
                coverage_overloaded,
                overloaded,
                effective_iters,
                median_active,
                events_dropped,
            };

            if stats_diag && render_count % 10 == 0 {
                if let Some(c) = recon.as_any_mut().downcast_mut::<CudaManifold>() {
                    let s = c.last_render_stats();
                    eprintln!(
                        "[recon-stats] #{render_count} render={render_ms:.2}ms (avg={render_ms_avg:.2}ms) \
                         pack={:.2}ms upload={:.2}ms kernel={:.2}ms download={:.2}ms events={} integrated={} decim_p={:.3} pending_after={} \
                         coverage={paired_coverage_pct:.1}% render_overloaded={render_overloaded} coverage_overloaded={coverage_overloaded} \
                         iters={effective_iters} median={median_active} \
                         events_dropped={events_dropped} coalesced_requests={coalesced_requests} request_gap_us={request_gap_us:?}",
                        s.pack_ms,
                        s.upload_ms,
                        s.kernel_ms,
                        s.download_ms,
                        s.events_this_render,
                        s.events_integrated,
                        s.decimation_p,
                        s.pending_len
                    );
                } else {
                    eprintln!(
                        "[recon-stats] #{render_count} render={render_ms:.2}ms (avg={render_ms_avg:.2}ms) \
                         coverage={paired_coverage_pct:.1}% render_overloaded={render_overloaded} coverage_overloaded={coverage_overloaded} \
                         events_dropped={events_dropped} \
                         coalesced_requests={coalesced_requests} request_gap_us={request_gap_us:?} (non-CudaManifold plugin, no RenderStats)"
                    );
                }
            }

            continue;
        }

        if ctrl.paired_only.load(Ordering::Relaxed) {
            continue;
        }

        let paired_recently = last_request.map_or(false, |t| t.elapsed() < Duration::from_millis(500));
        if !paired_recently && last_render.elapsed() >= Duration::from_millis(33) && last_t > 0 {
            recon.render_at(last_t, &mut img);
            let ci = egui::ColorImage::from_gray([w as usize, h as usize], &img.data);
            send_latest(&tx_img, &tx_img_rx, ci);
            last_render = Instant::now();
        }
    }
}
