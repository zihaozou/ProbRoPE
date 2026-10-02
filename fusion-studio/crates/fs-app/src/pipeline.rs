use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crossbeam_channel::{never, select, Receiver, RecvTimeoutError, Sender};
use fs_core::bus::{send_latest, EvkMsg, FlirMsg};
use fs_core::clock::ClockParams;
use fs_core::health::{SyncHealth, SyncHealthMonitor, SyncState};
use fs_core::sync::{SyncEngine, SyncStats};
use fs_core::{EventBatch, RgbFrame, SyncSource, SyncedFrame};

pub enum PipelineCmd {

    Reseed {
        fps: f64,

        seed: (i64, i64),
    },
    Desync,
}

pub struct PipelineOut {

    pub flir_preview: (Sender<RgbFrame>, Receiver<RgbFrame>),
    pub events: (Sender<EventBatch>, Receiver<EventBatch>),

    pub transform: (Sender<Arc<SyncedFrame>>, Receiver<Arc<SyncedFrame>>),
    pub stats: Arc<Mutex<SyncStats>>,
    pub events_dropped: Arc<AtomicU64>,
    pub frames_dropped_pre_transform: Arc<AtomicU64>,

    pub health: Arc<Mutex<SyncHealth>>,
    pub clock: Arc<Mutex<Option<ClockParams>>>,
}

pub const SHUTDOWN_POLL: Duration = Duration::from_millis(200);


struct HealthCtx {
    monitor: SyncHealthMonitor,
    t0: Instant,
}

impl HealthCtx {
    fn new() -> Self {
        HealthCtx { monitor: SyncHealthMonitor::new(), t0: Instant::now() }
    }

    fn now_ms(&self) -> u64 {
        self.t0.elapsed().as_millis() as u64
    }


    fn push(&mut self, source: SyncSource) {
        let now = self.now_ms();
        self.monitor.push(source, now);
    }


    fn tick(&mut self) {
        let now = self.now_ms();
        self.monitor.tick(now);
    }

    fn reset(&mut self) {
        let now = self.now_ms();
        self.monitor.reset(now);
    }

    fn set_desynced(&mut self, desynced: bool) {
        self.monitor.set_desynced(desynced);
    }

    fn health(&self) -> SyncHealth {
        self.monitor.health()
    }
}

pub fn sync_loop(
    rx_flir: Receiver<FlirMsg>,
    rx_evk: Receiver<EvkMsg>,
    rx_cmd: Receiver<PipelineCmd>,
    mut engine: SyncEngine,
    out: PipelineOut,
    shutdown: Arc<AtomicBool>,
) {
    let mut health = HealthCtx::new();
    let mut last_health_state = SyncState::Healthy;

    let mut first_frame_t: Option<i64> = None;
    let mut first_trigger_t: Option<i64> = None;
    let mut debug_count: u32 = 0;

    let mut flir_done = false;
    let mut evk_done = false;
    let mut desynced = false;
    let mut rx_cmd = rx_cmd;

    while !flir_done && !evk_done {
        select! {
            recv(rx_flir) -> msg => {
                flir_done = handle_flir(msg, &mut engine, &mut first_frame_t, first_trigger_t, &out, &mut debug_count, desynced, &mut health);
            }
            recv(rx_evk) -> msg => {
                evk_done = handle_evk(msg, &mut engine, &mut first_trigger_t, first_frame_t, &out, &mut debug_count, desynced, &mut health);
            }
            recv(rx_cmd) -> msg => match msg {
                Ok(cmd) => handle_cmd(cmd, &mut engine, &mut first_frame_t, &mut first_trigger_t, &mut desynced, &mut health),
                Err(_) => rx_cmd = never(),
            },
            default(SHUTDOWN_POLL) => {
                if shutdown.load(Ordering::Relaxed) { return; }
                health.tick();
            }
        }
        publish(&engine, &health, &out, &mut last_health_state);
    }

    if !flir_done {
        loop {
            match rx_flir.recv_timeout(SHUTDOWN_POLL) {
                Ok(msg) => {
                    let done = handle_flir(Ok(msg), &mut engine, &mut first_frame_t, first_trigger_t, &out, &mut debug_count, desynced, &mut health);
                    publish(&engine, &health, &out, &mut last_health_state);
                    if done { break; }
                }
                Err(RecvTimeoutError::Timeout) => {
                    if shutdown.load(Ordering::Relaxed) { return; }
                    health.tick();
                    publish(&engine, &health, &out, &mut last_health_state);
                }
                Err(RecvTimeoutError::Disconnected) => break,
            }
        }
    } else if !evk_done {
        loop {
            match rx_evk.recv_timeout(SHUTDOWN_POLL) {
                Ok(msg) => {
                    let done = handle_evk(Ok(msg), &mut engine, &mut first_trigger_t, first_frame_t, &out, &mut debug_count, desynced, &mut health);
                    publish(&engine, &health, &out, &mut last_health_state);
                    if done { break; }
                }
                Err(RecvTimeoutError::Timeout) => {
                    if shutdown.load(Ordering::Relaxed) { return; }
                    health.tick();
                    publish(&engine, &health, &out, &mut last_health_state);
                }
                Err(RecvTimeoutError::Disconnected) => break,
            }
        }
    }
}


fn handle_flir(
    msg: Result<FlirMsg, crossbeam_channel::RecvError>,
    engine: &mut SyncEngine,
    first_frame_t: &mut Option<i64>,
    first_trigger_t: Option<i64>,
    out: &PipelineOut,
    debug_count: &mut u32,
    desynced: bool,
    health: &mut HealthCtx,
) -> bool {
    match msg {
        Ok(FlirMsg::Frame(f)) => {
            send_latest(&out.flir_preview.0, &out.flir_preview.1, f.clone());
            if !desynced {
                if first_frame_t.is_none() { *first_frame_t = Some(f.t_cam_us); }
                maybe_seed(engine, *first_frame_t, first_trigger_t);
                emit(engine.push_frame(f), out, debug_count, health);
            }
            false
        }
        Ok(FlirMsg::Eof) | Err(_) => true,
    }
}


fn handle_evk(
    msg: Result<EvkMsg, crossbeam_channel::RecvError>,
    engine: &mut SyncEngine,
    first_trigger_t: &mut Option<i64>,
    first_frame_t: Option<i64>,
    out: &PipelineOut,
    debug_count: &mut u32,
    desynced: bool,
    health: &mut HealthCtx,
) -> bool {
    match msg {
        Ok(EvkMsg::Events(batch)) => {
            if send_latest(&out.events.0, &out.events.1, batch) {
                out.events_dropped.fetch_add(1, Ordering::Relaxed);
            }
            false
        }
        Ok(EvkMsg::Trigger(tr)) => {
            if !desynced {
                if tr.polarity == engine.trigger_polarity() && first_trigger_t.is_none() {
                    *first_trigger_t = Some(tr.t_us);
                }
                maybe_seed(engine, first_frame_t, *first_trigger_t);
                emit(engine.push_trigger(tr), out, debug_count, health);
            }
            false
        }
        Ok(EvkMsg::Eof) | Err(_) => true,
    }
}


fn handle_cmd(
    cmd: PipelineCmd,
    engine: &mut SyncEngine,
    first_frame_t: &mut Option<i64>,
    first_trigger_t: &mut Option<i64>,
    desynced: &mut bool,
    health: &mut HealthCtx,
) {
    match cmd {
        PipelineCmd::Reseed { fps, seed } => {
            let mut cfg = engine.config();
            cfg.fps = fps;
            *engine = SyncEngine::new(cfg);
            engine.seed(seed.0, seed.1);
            *first_frame_t = None;
            *first_trigger_t = None;
            *desynced = false;
            health.reset();
            eprintln!("sync: reseeded at fps={fps} seed=({}, {})", seed.0, seed.1);
        }
        PipelineCmd::Desync => {
            *first_frame_t = None;
            *first_trigger_t = None;
            *desynced = true;
            health.set_desynced(true);
            eprintln!("sync: DESYNCED (fps apply failed after clock reset) -- no paired frames until a successful Apply");
        }
    }
}

fn emit(synced: Vec<SyncedFrame>, out: &PipelineOut, debug_count: &mut u32, health: &mut HealthCtx) {
    for s in synced {
        if *debug_count < 5 {
            eprintln!(
                "sync-debug: seq={} t_flir_us={} t_evk_us={} source={:?}",
                s.frame.seq, s.frame.t_cam_us, s.t_evk_us, s.source
            );
            *debug_count += 1;
        }
        health.push(s.source);
        if send_latest(&out.transform.0, &out.transform.1, Arc::new(s)) {
            out.frames_dropped_pre_transform.fetch_add(1, Ordering::Relaxed);
        }
    }
}

fn publish(engine: &SyncEngine, health_ctx: &HealthCtx, out: &PipelineOut, last_health_state: &mut SyncState) {
    if let Ok(mut s) = out.stats.lock() { *s = engine.stats(); }
    if let Ok(mut c) = out.clock.lock() {
        *c = engine.is_seeded().then(|| engine.clock_model());
    }
    let health = health_ctx.health();
    if health.state == SyncState::Lost && *last_health_state != SyncState::Lost {
        let stats = engine.stats();
        let rate = match health.match_rate_window {
            Some(r) => format!("{r:.2}"),
            None => "n/a".to_string(),
        };
        eprintln!(
            "sync-lost: consecutive={} rate={} matched={} interpolated={} spurious={} drift_ppm={:.1} offset_us={:.0}",
            health.consecutive_non_matched, rate, stats.matched, stats.interpolated, stats.spurious, stats.drift_ppm, stats.offset_us
        );
    }
    *last_health_state = health.state;
    if let Ok(mut h) = out.health.lock() { *h = health; }
}

fn maybe_seed(engine: &mut SyncEngine, frame_t: Option<i64>, trigger_t: Option<i64>) {
    if !engine.is_seeded() {
        if let (Some(f), Some(t)) = (frame_t, trigger_t) { engine.seed(f, t); }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossbeam_channel::bounded;
    use fs_core::sync::SyncConfig;
    use fs_core::{PixelFormat, SyncSource, TriggerEvent};

    fn frame(seq: u64, t: i64) -> RgbFrame {
        RgbFrame { seq, t_cam_us: t, w: 1, h: 1, data: vec![0], format: PixelFormat::Gray8 }
    }
    fn trig(t: i64) -> TriggerEvent {
        TriggerEvent { t_us: t, polarity: 1 }
    }


    fn no_shutdown() -> Arc<AtomicBool> {
        Arc::new(AtomicBool::new(false))
    }

    fn test_out() -> (PipelineOut, Receiver<Arc<SyncedFrame>>) {
        let flir_preview = bounded(64);
        let events = bounded(64);
        let transform = bounded(256);
        let synced_rx = transform.1.clone();
        let out = PipelineOut {
            flir_preview,
            events,
            transform,
            stats: Arc::new(Mutex::new(SyncStats::default())),
            events_dropped: Arc::new(AtomicU64::new(0)),
            frames_dropped_pre_transform: Arc::new(AtomicU64::new(0)),
            health: Arc::new(Mutex::new(SyncHealthMonitor::new().health())),
            clock: Arc::new(Mutex::new(None)),
        };
        (out, synced_rx)
    }

    #[test]
    fn transform_intake_overflow_is_counted() {
        let transform = bounded(1);
        let transform_rx = transform.1.clone();
        let out = PipelineOut {
            flir_preview: bounded(4),
            events: bounded(4),
            transform,
            stats: Arc::new(Mutex::new(SyncStats::default())),
            events_dropped: Arc::new(AtomicU64::new(0)),
            frames_dropped_pre_transform: Arc::new(AtomicU64::new(0)),
            health: Arc::new(Mutex::new(SyncHealthMonitor::new().health())),
            clock: Arc::new(Mutex::new(None)),
        };
        let mut health = HealthCtx::new();
        let mut dbg = 0u32;
        let mk = |seq: u64| SyncedFrame {
            frame: frame(seq, seq as i64 * 33_333),
            t_evk_us: seq as i64 * 33_333 + 3_000,
            source: SyncSource::Matched,
        };

        emit(vec![mk(1), mk(2), mk(3)], &out, &mut dbg, &mut health);
        assert_eq!(
            out.frames_dropped_pre_transform.load(Ordering::Relaxed),
            2,
            "容量 1 的进料口收 3 帧:丢掉的 2 帧必须计数"
        );
        let got: Vec<_> = transform_rx.try_iter().collect();
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].frame.seq, 3, "丢最旧,留最新");
    }

    #[test]
    fn publish_writes_clock_model_only_after_seeding() {
        let mut engine = SyncEngine::new(SyncConfig { fps: 30.0, gate_frac: 0.2, trigger_polarity: 1, max_pending: 120 });
        let (out, _synced_rx) = test_out();
        let health = HealthCtx::new();
        let mut last_health_state = SyncState::Healthy;

        publish(&engine, &health, &out, &mut last_health_state);
        assert!(
            out.clock.lock().unwrap().is_none(),
            "an unseeded engine must publish null, not a fabricated zero-drift/zero-offset mapping"
        );

        engine.seed(1_000, 4_000);
        publish(&engine, &health, &out, &mut last_health_state);
        let got = out.clock.lock().unwrap().expect("a seeded engine must publish Some");
        assert_eq!(got.x0, 1_000, "x0 must be the handshake frame's t_flir");
        assert_eq!(got, engine.clock_model(), "published value must be the engine's own model, not a re-derived copy");
    }

    #[test]
    fn reseed_replaces_engine_and_mapping() {
        let mut engine = SyncEngine::new(SyncConfig { fps: 30.0, gate_frac: 0.2, trigger_polarity: 1, max_pending: 120 });
        engine.seed(0, 3_000);
        let dt30 = 33_333;
        for i in 1..5i64 {
            engine.push_trigger(trig(i * dt30 + 3_000));
            let out = engine.push_frame(frame(i as u64, i * dt30));
            assert_eq!(out.len(), 1);
        }

        let (mut ff, mut ft) = (Some(123), Some(456));
        let mut desynced = true;
        let mut health = HealthCtx::new();
        handle_cmd(PipelineCmd::Reseed { fps: 60.0, seed: (0, 900_000) }, &mut engine, &mut ff, &mut ft, &mut desynced, &mut health);
        assert_eq!((ff, ft), (None, None), "auto-seed trackers cleared");
        assert!(!desynced, "successful reseed clears a standing desync");
        assert_eq!(engine.config().fps, 60.0);
        assert!(engine.is_seeded());

        let dt60 = 16_667;
        engine.push_trigger(trig(dt60 + 900_000));
        let out = engine.push_frame(frame(100, dt60));
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].source, SyncSource::Matched);
        assert_eq!(out[0].t_evk_us, dt60 + 900_000);
    }


    #[test]
    fn sync_loop_survives_cmd_disconnect() {
        let (flir_tx, flir_rx) = bounded(16);
        let (evk_tx, evk_rx) = bounded(16);
        let (cmd_tx, cmd_rx) = bounded::<PipelineCmd>(1);
        drop(cmd_tx);
        let mut engine = SyncEngine::new(SyncConfig { fps: 30.0, gate_frac: 0.2, trigger_polarity: 1, max_pending: 120 });
        engine.seed(0, 3_000);
        let (out, synced_rx) = test_out();
        let h = std::thread::spawn(move || sync_loop(flir_rx, evk_rx, cmd_rx, engine, out, no_shutdown()));

        let dt = 33_333;
        for i in 1..10i64 {
            evk_tx.send(EvkMsg::Trigger(trig(i * dt + 3_000))).unwrap();
            flir_tx.send(FlirMsg::Frame(frame(i as u64, i * dt))).unwrap();
        }
        flir_tx.send(FlirMsg::Eof).unwrap();
        evk_tx.send(EvkMsg::Eof).unwrap();
        h.join().unwrap();

        let got: Vec<_> = synced_rx.try_iter().collect();
        assert_eq!(got.len(), 9, "all 9 pairs must flow despite the dead cmd channel");
        assert!(got.iter().all(|s| s.source == SyncSource::Matched));
    }

    #[test]
    fn sync_loop_processes_reseed() {
        let (flir_tx, flir_rx) = bounded(16);
        let (evk_tx, evk_rx) = bounded(16);
        let (cmd_tx, cmd_rx) = bounded::<PipelineCmd>(1);
        let mut engine = SyncEngine::new(SyncConfig { fps: 30.0, gate_frac: 0.2, trigger_polarity: 1, max_pending: 120 });
        engine.seed(0, 3_000);
        let (out, synced_rx) = test_out();
        let h = std::thread::spawn(move || sync_loop(flir_rx, evk_rx, cmd_rx, engine, out, no_shutdown()));

        cmd_tx.send(PipelineCmd::Reseed { fps: 60.0, seed: (0, 900_000) }).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(100));

        let dt = 16_667;
        for i in 1..10i64 {
            evk_tx.send(EvkMsg::Trigger(trig(i * dt + 900_000))).unwrap();
            flir_tx.send(FlirMsg::Frame(frame(i as u64, i * dt))).unwrap();
        }
        flir_tx.send(FlirMsg::Eof).unwrap();
        evk_tx.send(EvkMsg::Eof).unwrap();
        h.join().unwrap();

        let got: Vec<_> = synced_rx.try_iter().collect();
        assert_eq!(got.len(), 9);
        assert!(
            got.iter().all(|s| s.source == SyncSource::Matched),
            "post-reseed pairs must match under the NEW mapping (got {:?})",
            got.iter().map(|s| s.source).collect::<Vec<_>>()
        );
        assert_eq!(got[0].t_evk_us, dt + 900_000);
    }

    #[test]
    fn desync_stops_synced_output_until_reseed() {
        let (flir_tx, flir_rx) = bounded(64);
        let (evk_tx, evk_rx) = bounded(64);
        let (cmd_tx, cmd_rx) = bounded::<PipelineCmd>(4);
        let mut engine = SyncEngine::new(SyncConfig { fps: 30.0, gate_frac: 0.2, trigger_polarity: 1, max_pending: 120 });
        engine.seed(0, 3_000);
        let (out, synced_rx) = test_out();
        let preview_rx = out.flir_preview.1.clone();
        let h = std::thread::spawn(move || sync_loop(flir_rx, evk_rx, cmd_rx, engine, out, no_shutdown()));

        let dt = 33_333;
        for i in 1..4i64 {
            evk_tx.send(EvkMsg::Trigger(trig(i * dt + 3_000))).unwrap();
            flir_tx.send(FlirMsg::Frame(frame(i as u64, i * dt))).unwrap();
        }
        for _ in 0..3 {
            synced_rx.recv_timeout(std::time::Duration::from_secs(2)).expect("healthy pair must emit");
        }
        while preview_rx.try_recv().is_ok() {}

        cmd_tx.send(PipelineCmd::Desync).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(100));
        for i in 1..6i64 {
            evk_tx.send(EvkMsg::Trigger(trig(500_000 + i * dt))).unwrap();
            flir_tx.send(FlirMsg::Frame(frame(100 + i as u64, i * dt))).unwrap();
        }
        std::thread::sleep(std::time::Duration::from_millis(200));
        assert!(synced_rx.try_recv().is_err(), "desynced: no SyncedFrame may be emitted, not even Interpolated");
        assert!(preview_rx.try_recv().is_ok(), "desynced: raw preview must KEEP flowing");

        cmd_tx.send(PipelineCmd::Reseed { fps: 30.0, seed: (200_000, 800_000) }).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(100));
        for i in 1..4i64 {
            evk_tx.send(EvkMsg::Trigger(trig(800_000 + i * dt))).unwrap();
            flir_tx.send(FlirMsg::Frame(frame(200 + i as u64, 200_000 + i * dt))).unwrap();
        }
        flir_tx.send(FlirMsg::Eof).unwrap();
        evk_tx.send(EvkMsg::Eof).unwrap();
        h.join().unwrap();

        let got: Vec<_> = synced_rx.try_iter().collect();
        assert_eq!(got.len(), 3, "paired output resumes after the reseed");
        assert!(got.iter().all(|s| s.source == SyncSource::Matched));
        assert_eq!(got[0].t_evk_us, 800_000 + dt);
    }


    #[test]
    fn sync_loop_exits_on_shutdown_with_senders_still_alive() {
        let (flir_tx, flir_rx) = bounded(16);
        let (evk_tx, evk_rx) = bounded(16);
        let (cmd_tx, cmd_rx) = bounded::<PipelineCmd>(1);
        let mut engine = SyncEngine::new(SyncConfig { fps: 30.0, gate_frac: 0.2, trigger_polarity: 1, max_pending: 120 });
        engine.seed(0, 3_000);
        let (out, _synced_rx) = test_out();
        let shutdown = Arc::new(AtomicBool::new(false));
        let h = {
            let shutdown = shutdown.clone();
            std::thread::spawn(move || sync_loop(flir_rx, evk_rx, cmd_rx, engine, out, shutdown))
        };

        let dt = 33_333;
        for i in 1..4i64 {
            evk_tx.send(EvkMsg::Trigger(trig(i * dt + 3_000))).unwrap();
            flir_tx.send(FlirMsg::Frame(frame(i as u64, i * dt))).unwrap();
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
        assert!(!h.is_finished(), "must still be running while shutdown is clear");

        shutdown.store(true, Ordering::Relaxed);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(3);
        while !h.is_finished() && std::time::Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        assert!(h.is_finished(), "sync_loop must exit on the shutdown flag alone, with every sender still alive");
        h.join().unwrap();
        drop((flir_tx, evk_tx, cmd_tx));
    }


    #[test]
    fn sync_loop_phase2_exits_on_shutdown_with_evk_sender_alive() {
        let (flir_tx, flir_rx) = bounded::<FlirMsg>(16);
        let (evk_tx, evk_rx) = bounded(16);
        let (cmd_tx, cmd_rx) = bounded::<PipelineCmd>(1);
        drop(flir_tx);
        let mut engine = SyncEngine::new(SyncConfig { fps: 30.0, gate_frac: 0.2, trigger_polarity: 1, max_pending: 120 });
        engine.seed(0, 3_000);
        let (out, _synced_rx) = test_out();
        let shutdown = Arc::new(AtomicBool::new(false));
        let h = {
            let shutdown = shutdown.clone();
            std::thread::spawn(move || sync_loop(flir_rx, evk_rx, cmd_rx, engine, out, shutdown))
        };

        evk_tx.send(EvkMsg::Trigger(trig(3_000))).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(100));
        assert!(!h.is_finished(), "phase 2 must still be draining EVK while shutdown is clear");

        shutdown.store(true, Ordering::Relaxed);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(3);
        while !h.is_finished() && std::time::Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        assert!(h.is_finished(), "phase 2's EVK drain must exit on the flag, with the EVK sender still alive");
        h.join().unwrap();
        drop((evk_tx, cmd_tx));
    }


    #[test]
    fn health_is_published_as_frames_flow() {
        let (flir_tx, flir_rx) = bounded(64);
        let (evk_tx, evk_rx) = bounded(64);
        let (_cmd_tx, cmd_rx) = bounded::<PipelineCmd>(1);
        let mut engine = SyncEngine::new(SyncConfig { fps: 30.0, gate_frac: 0.2, trigger_polarity: 1, max_pending: 120 });
        engine.seed(0, 3_000);
        let (out, synced_rx) = test_out();
        let health = out.health.clone();
        let shutdown = Arc::new(AtomicBool::new(false));
        let h = std::thread::spawn({
            let shutdown = shutdown.clone();
            move || sync_loop(flir_rx, evk_rx, cmd_rx, engine, out, shutdown)
        });

        let dt = 33_333;
        for i in 1..20i64 {
            evk_tx.send(EvkMsg::Trigger(trig(i * dt + 3_000))).unwrap();
            flir_tx.send(FlirMsg::Frame(frame(i as u64, i * dt))).unwrap();
        }
        for _ in 0..19 {
            synced_rx.recv_timeout(std::time::Duration::from_secs(2)).expect("paired frames must flow");
        }
        assert_eq!(health.lock().unwrap().state, fs_core::health::SyncState::Healthy);

        shutdown.store(true, Ordering::Relaxed);
        h.join().unwrap();
        drop((flir_tx, evk_tx));
    }


    #[test]
    fn health_goes_lost_when_frames_stop_while_senders_stay_alive() {
        let (flir_tx, flir_rx) = bounded(64);
        let (evk_tx, evk_rx) = bounded(64);
        let (_cmd_tx, cmd_rx) = bounded::<PipelineCmd>(1);
        let mut engine = SyncEngine::new(SyncConfig { fps: 30.0, gate_frac: 0.2, trigger_polarity: 1, max_pending: 120 });
        engine.seed(0, 3_000);
        let (out, synced_rx) = test_out();
        let health = out.health.clone();
        let shutdown = Arc::new(AtomicBool::new(false));
        let h = std::thread::spawn({
            let shutdown = shutdown.clone();
            move || sync_loop(flir_rx, evk_rx, cmd_rx, engine, out, shutdown)
        });

        let dt = 33_333;
        for i in 1..5i64 {
            evk_tx.send(EvkMsg::Trigger(trig(i * dt + 3_000))).unwrap();
            flir_tx.send(FlirMsg::Frame(frame(i as u64, i * dt))).unwrap();
        }
        for _ in 0..4 {
            synced_rx.recv_timeout(std::time::Duration::from_secs(2)).expect("paired frames must flow");
        }
        std::thread::sleep(std::time::Duration::from_millis(
            fs_core::health::NO_FRAME_MS + 500,
        ));
        assert_eq!(
            health.lock().unwrap().state,
            fs_core::health::SyncState::Lost,
            "帧停了就该报 Lost,哪怕通道还开着"
        );

        shutdown.store(true, Ordering::Relaxed);
        h.join().unwrap();
        drop((flir_tx, evk_tx));
    }

    #[test]
    fn health_goes_lost_in_phase2_evk_drain_after_flir_dies() {
        let (flir_tx, flir_rx) = bounded(64);
        let (evk_tx, evk_rx) = bounded(64);
        let (_cmd_tx, cmd_rx) = bounded::<PipelineCmd>(1);
        let mut engine = SyncEngine::new(SyncConfig { fps: 30.0, gate_frac: 0.2, trigger_polarity: 1, max_pending: 120 });
        engine.seed(0, 3_000);
        let (out, synced_rx) = test_out();
        let health = out.health.clone();
        let shutdown = Arc::new(AtomicBool::new(false));
        let h = std::thread::spawn({
            let shutdown = shutdown.clone();
            move || sync_loop(flir_rx, evk_rx, cmd_rx, engine, out, shutdown)
        });

        let dt = 33_333;
        for i in 1..5i64 {
            evk_tx.send(EvkMsg::Trigger(trig(i * dt + 3_000))).unwrap();
            flir_tx.send(FlirMsg::Frame(frame(i as u64, i * dt))).unwrap();
        }
        for _ in 0..4 {
            synced_rx.recv_timeout(std::time::Duration::from_secs(2)).expect("paired frames must flow");
        }

        flir_tx.send(FlirMsg::Eof).unwrap();
        drop(flir_tx);

        for _ in 0..6 {
            std::thread::sleep(std::time::Duration::from_millis(300));
            let _ = evk_tx.send(EvkMsg::Events(fs_core::EventBatch { events: vec![] }));
        }
        std::thread::sleep(std::time::Duration::from_millis(700));

        assert_eq!(
            health.lock().unwrap().state,
            fs_core::health::SyncState::Lost,
            "FLIR died and Phase 2's EVK drain can never push another SyncedFrame -- \
             health must not freeze at whatever it read when Phase 2 was entered"
        );

        shutdown.store(true, Ordering::Relaxed);
        h.join().unwrap();
        drop(evk_tx);
    }
}
