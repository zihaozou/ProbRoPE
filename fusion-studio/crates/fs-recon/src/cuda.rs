use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::{sys, CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

use fs_core::{Event, EventBatch, GrayImage};

use crate::Reconstructor;

const KERNELS_SRC: &str = include_str!("kernels.cu");


fn cuda_include_path() -> String {
    std::env::var("CUDA_PATH")
        .map(|p| format!("{p}\\include"))
        .unwrap_or_else(|_| {
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\include".to_string()
        })
}

fn nvrtc_compile_opts() -> CompileOptions {
    CompileOptions {
        arch: Some("compute_89"),
        ftz: Some(true),
        prec_div: Some(false),
        prec_sqrt: Some(false),
        fmad: Some(true),
        include_paths: vec![cuda_include_path()],
        ..Default::default()
    }
}


const PD_TAU: f32 = 0.25;
const PD_SIGMA: f32 = 0.5;


const PENDING_CAP: usize = 8_000_000;

const DEFAULT_EVENT_DECIMATION_THRESHOLD: usize = 6_000_000;

const F_CLAMP_ABS: f32 = 2.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Normalization {
    Fixed,
    Percentile,
    TileLocal,
}

impl Default for Normalization {
    fn default() -> Self {
        Normalization::Fixed
    }
}


const REBASE_THRESHOLD_US: i64 = 8_000_000;


const REBASE_MARGIN_US: i64 = 1_000_000;


const TILE_PX_W: u32 = 80;
const TILE_PX_H: u32 = 72;


const TILE_REDUCE_THREADS: u32 = 256;


const TILE_ROBUST_STD_K: f32 = 2.5;


#[derive(Clone, Copy, Debug, Default)]
pub struct RenderStats {

    pub pack_ms: f64,
    pub upload_ms: f64,

    pub kernel_ms: f64,

    pub download_ms: f64,
    pub events_this_render: usize,
    pub pending_len: usize,
    pub decimation_p: f64,

    pub events_integrated: usize,
}

#[inline]
fn decim_hash(x: u16, y: u16, p: i8, t_us: i64) -> u64 {
    let mut h = (t_us as u64)
        .wrapping_mul(0x9E3779B97F4A7C15)
        ^ (((x as u64) << 32) | ((y as u64) << 16) | (p as u8 as u64));
    h ^= h >> 30;
    h = h.wrapping_mul(0xBF58476D1CE4E5B9);
    h ^= h >> 27;
    h = h.wrapping_mul(0x94D049BB133111EB);
    h ^= h >> 31;
    h
}

pub struct CudaManifold {
    w: u32,
    h: u32,
    stream: Arc<CudaStream>,
    integrate_events: CudaFunction,
    shift_tmap_fn: CudaFunction,
    decay_state_fn: CudaFunction,
    clamp_f_fn: CudaFunction,
    compute_g: CudaFunction,
    pd_dual: CudaFunction,
    pd_primal: CudaFunction,

    pd_solve_persistent_fn: CudaFunction,
    tonemap: CudaFunction,

    median_filter_fn: CudaFunction,

    tile_stats_fn: CudaFunction,

    tile_bias_scale_fn: CudaFunction,
    tonemap_tiled_fn: CudaFunction,

    f: CudaSlice<f32>,

    t_map: CudaSlice<f32>,
    u: CudaSlice<f32>,

    u_bar: CudaSlice<f32>,
    p_x: CudaSlice<f32>,
    p_y: CudaSlice<f32>,
    g: CudaSlice<f32>,
    out_buf: CudaSlice<u8>,
    percentile_scratch: Vec<f32>,
    packed_scratch: Vec<i32>,
    median_scratch: CudaSlice<u8>,

    tiles_x: u32,
    tiles_y: u32,

    tile_min: CudaSlice<f32>,
    tile_max: CudaSlice<f32>,
    tile_sum: CudaSlice<f32>,
    tile_sumsq: CudaSlice<f32>,
    tile_bias: CudaSlice<f32>,
    tile_scale: CudaSlice<f32>,

    use_persistent_pd: bool,

    persistent_grid_dim: u32,
    persistent_block_dim: u32,
    pending: Vec<Event>,

    last_stats: RenderStats,

    origin_us: Option<i64>,

    last_render_t_us: Option<i64>,


    pub c_pos: f32,

    pub c_neg: f32,
    pub tonemap_scale: f32,

    pub lambda: f32,
    pub iters: u32,

    pub manifold_alpha: f32,
    pub tau_leak_us: f32,

    pub normalization: Normalization,

    pub percentile_lo: f32,
    pub percentile_hi: f32,


    pub median_filter: bool,


    pub event_decimation_threshold: usize,

    soa_ingest_fn: Option<CudaFunction>,
    soa_ingest_quads: Option<CudaSlice<i32>>,
}

impl CudaManifold {
    pub fn new(w: u32, h: u32) -> Result<Self, String> {
        let ctx = CudaContext::new(0).map_err(|e| format!("cuda device init failed: {e}"))?;
        let stream = ctx.default_stream();

        let ptx = compile_ptx_with_opts(KERNELS_SRC, nvrtc_compile_opts())
            .map_err(|e| format!("nvrtc compile failed: {e}"))?;
        let module = ctx.load_module(ptx).map_err(|e| format!("cuda module load failed: {e}"))?;
        let integrate_events = module
            .load_function("integrate_events")
            .map_err(|e| format!("cuda function load failed: {e}"))?;
        let shift_tmap_fn = module
            .load_function("shift_tmap")
            .map_err(|e| format!("cuda function load failed: {e}"))?;
        let decay_state_fn = module
            .load_function("decay_state")
            .map_err(|e| format!("cuda function load failed: {e}"))?;
        let clamp_f_fn = module
            .load_function("clamp_f")
            .map_err(|e| format!("cuda function load failed: {e}"))?;
        let compute_g = module
            .load_function("compute_g")
            .map_err(|e| format!("cuda function load failed: {e}"))?;
        let pd_dual = module
            .load_function("pd_dual")
            .map_err(|e| format!("cuda function load failed: {e}"))?;
        let pd_primal = module
            .load_function("pd_primal")
            .map_err(|e| format!("cuda function load failed: {e}"))?;
        let pd_solve_persistent_fn = module
            .load_function("pd_solve_persistent")
            .map_err(|e| format!("cuda function load failed: {e}"))?;
        let tonemap = module
            .load_function("tonemap")
            .map_err(|e| format!("cuda function load failed: {e}"))?;
        let median_filter_fn = module
            .load_function("median_filter_3x3")
            .map_err(|e| format!("cuda function load failed: {e}"))?;
        let tile_stats_fn = module
            .load_function("compute_tile_stats")
            .map_err(|e| format!("cuda function load failed: {e}"))?;
        let tile_bias_scale_fn = module
            .load_function("compute_tile_bias_scale")
            .map_err(|e| format!("cuda function load failed: {e}"))?;
        let tonemap_tiled_fn = module
            .load_function("tonemap_tiled")
            .map_err(|e| format!("cuda function load failed: {e}"))?;

        let n = (w as usize) * (h as usize);
        let f = stream.alloc_zeros::<f32>(n).map_err(|e| format!("cuda buffer alloc failed: {e}"))?;
        let t_map = stream.alloc_zeros::<f32>(n).map_err(|e| format!("cuda buffer alloc failed: {e}"))?;
        let u = stream.alloc_zeros::<f32>(n).map_err(|e| format!("cuda buffer alloc failed: {e}"))?;
        let u_bar = stream.alloc_zeros::<f32>(n).map_err(|e| format!("cuda buffer alloc failed: {e}"))?;
        let p_x = stream.alloc_zeros::<f32>(n).map_err(|e| format!("cuda buffer alloc failed: {e}"))?;
        let p_y = stream.alloc_zeros::<f32>(n).map_err(|e| format!("cuda buffer alloc failed: {e}"))?;
        let g = stream.alloc_zeros::<f32>(n).map_err(|e| format!("cuda buffer alloc failed: {e}"))?;
        let out_buf = stream.alloc_zeros::<u8>(n).map_err(|e| format!("cuda buffer alloc failed: {e}"))?;
        let median_scratch = stream.alloc_zeros::<u8>(n).map_err(|e| format!("cuda buffer alloc failed: {e}"))?;

        let tiles_x = (w + TILE_PX_W - 1) / TILE_PX_W;
        let tiles_y = (h + TILE_PX_H - 1) / TILE_PX_H;
        let n_tiles = (tiles_x as usize) * (tiles_y as usize);
        let tile_min = stream.alloc_zeros::<f32>(n_tiles).map_err(|e| format!("cuda buffer alloc failed: {e}"))?;
        let tile_max = stream.alloc_zeros::<f32>(n_tiles).map_err(|e| format!("cuda buffer alloc failed: {e}"))?;
        let tile_sum = stream.alloc_zeros::<f32>(n_tiles).map_err(|e| format!("cuda buffer alloc failed: {e}"))?;
        let tile_sumsq = stream.alloc_zeros::<f32>(n_tiles).map_err(|e| format!("cuda buffer alloc failed: {e}"))?;
        let tile_bias = stream.alloc_zeros::<f32>(n_tiles).map_err(|e| format!("cuda buffer alloc failed: {e}"))?;
        let tile_scale = stream.alloc_zeros::<f32>(n_tiles).map_err(|e| format!("cuda buffer alloc failed: {e}"))?;

        const PERSISTENT_BLOCK_DIM: u32 = 256;
        let sm_count = ctx
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
            .unwrap_or(1)
            .max(1) as u32;
        let coop_supported = ctx
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH)
            .unwrap_or(0)
            == 1;
        let occupancy_blocks_per_sm = if coop_supported {
            pd_solve_persistent_fn
                .occupancy_max_active_blocks_per_multiprocessor(PERSISTENT_BLOCK_DIM, 0, None)
                .unwrap_or(0)
        } else {
            0
        };
        let persistent_grid_dim = occupancy_blocks_per_sm * sm_count;
        let use_persistent_pd = if persistent_grid_dim == 0 {
            false
        } else {
            let probe_cfg = LaunchConfig {
                grid_dim: (persistent_grid_dim, 1, 1),
                block_dim: (PERSISTENT_BLOCK_DIM, 1, 1),
                shared_mem_bytes: 0,
            };
            let w_i32 = w as i32;
            let h_i32 = h as i32;
            let alpha0: f32 = 300.0;
            let t_now0: f32 = 0.0;
            let tau0: f32 = PD_TAU;
            let sigma0: f32 = PD_SIGMA;
            let lambda0: f32 = 2.0;
            let iters0: i32 = 0;
            let launch_result: Result<(), String> = (|| {
                let mut u_probe = stream.alloc_zeros::<f32>(n).map_err(|e| e.to_string())?;
                let mut u_bar_probe = stream.alloc_zeros::<f32>(n).map_err(|e| e.to_string())?;
                let mut p_x_probe = stream.alloc_zeros::<f32>(n).map_err(|e| e.to_string())?;
                let mut p_y_probe = stream.alloc_zeros::<f32>(n).map_err(|e| e.to_string())?;
                let mut g_probe = stream.alloc_zeros::<f32>(n).map_err(|e| e.to_string())?;
                let f_probe = stream.alloc_zeros::<f32>(n).map_err(|e| e.to_string())?;
                let t_map_probe = stream.alloc_zeros::<f32>(n).map_err(|e| e.to_string())?;
                let mut args = stream.launch_builder(&pd_solve_persistent_fn);
                args.arg(&mut u_probe);
                args.arg(&mut u_bar_probe);
                args.arg(&mut p_x_probe);
                args.arg(&mut p_y_probe);
                args.arg(&mut g_probe);
                args.arg(&f_probe);
                args.arg(&t_map_probe);
                args.arg(&alpha0);
                args.arg(&t_now0);
                args.arg(&tau0);
                args.arg(&sigma0);
                args.arg(&lambda0);
                args.arg(&w_i32);
                args.arg(&h_i32);
                args.arg(&iters0);
                unsafe { args.launch_cooperative(probe_cfg) }.map_err(|e| e.to_string())?;
                stream.synchronize().map_err(|e| e.to_string())?;
                Ok(())
            })();
            match launch_result {
                Ok(()) => true,
                Err(e) => {
                    eprintln!(
                        "CudaManifold: cooperative-launch smoke test failed ({e}) -- \
                         falling back to the per-iteration PD-TV kernel loop"
                    );
                    false
                }
            }
        };
        if use_persistent_pd {
            eprintln!(
                "CudaManifold: using persistent cooperative-groups PD-TV kernel \
                 (grid={persistent_grid_dim} blocks x {PERSISTENT_BLOCK_DIM} threads, {sm_count} SMs)"
            );
        }

        Ok(Self {
            w,
            h,
            stream,
            integrate_events,
            shift_tmap_fn,
            decay_state_fn,
            clamp_f_fn,
            compute_g,
            pd_dual,
            pd_primal,
            pd_solve_persistent_fn,
            tonemap,
            median_filter_fn,
            tile_stats_fn,
            tile_bias_scale_fn,
            tonemap_tiled_fn,
            f,
            t_map,
            u,
            u_bar,
            p_x,
            p_y,
            g,
            out_buf,
            percentile_scratch: vec![0.0f32; n],
            packed_scratch: Vec::new(),
            median_scratch,
            tiles_x,
            tiles_y,
            tile_min,
            tile_max,
            tile_sum,
            tile_sumsq,
            tile_bias,
            tile_scale,
            use_persistent_pd,
            persistent_grid_dim,
            persistent_block_dim: PERSISTENT_BLOCK_DIM,
            pending: Vec::new(),
            last_stats: RenderStats::default(),
            origin_us: None,
            last_render_t_us: None,
            c_pos: 0.1,
            c_neg: 0.1,
            tonemap_scale: 400.0,
            lambda: 2.0,
            iters: 10,
            manifold_alpha: 300.0,
            tau_leak_us: 200_000.0,
            normalization: Normalization::Fixed,
            percentile_lo: 0.02,
            percentile_hi: 0.98,
            median_filter: false,
            event_decimation_threshold: DEFAULT_EVENT_DECIMATION_THRESHOLD,
            soa_ingest_fn: None,
            soa_ingest_quads: None,
        })
    }

    pub fn last_render_stats(&self) -> RenderStats {
        self.last_stats
    }

    fn launch_shift_tmap(&mut self, delta_us: f32) {
        let n = self.w * self.h;
        let n_i32 = n as i32;
        let cfg = LaunchConfig::for_num_elems(n);
        let mut args = self.stream.launch_builder(&self.shift_tmap_fn);
        args.arg(&mut self.t_map);
        args.arg(&delta_us);
        args.arg(&n_i32);
        unsafe { args.launch(cfg) }.expect("cuda shift_tmap launch failed");
    }
}

impl Reconstructor for CudaManifold {
    fn name(&self) -> &'static str { "cuda-manifold" }

    fn push_events(&mut self, batch: &EventBatch) {
        debug_assert!(
            batch.events.windows(2).all(|w| w[0].t_us <= w[1].t_us),
            "push_events requires events within a batch to be time-ordered"
        );
        debug_assert!(
            match (self.pending.last(), batch.events.first()) {
                (Some(last), Some(first)) => last.t_us <= first.t_us,
                _ => true,
            },
            "push_events requires each batch to start no earlier than the last pending event"
        );
        self.pending.extend(batch.events.iter().copied());
        if self.pending.len() > PENDING_CAP {
            let drop_n = self.pending.len() - PENDING_CAP;
            self.pending.drain(0..drop_n);
        }
    }

    fn render_at(&mut self, t_us: i64, out: &mut GrayImage) {
        debug_assert_eq!((out.w, out.h), (self.w, self.h));

        let split = self.pending.partition_point(|e| e.t_us <= t_us);
        let batch: Vec<Event> = self.pending.drain(0..split).collect();
        let events_this_render = batch.len();
        let pending_len = self.pending.len();

        if self.origin_us.is_none() {
            if let Some(first) = batch.first() {
                self.origin_us = Some(first.t_us);
            }
        }
        if let Some(origin) = self.origin_us {
            if t_us - origin > REBASE_THRESHOLD_US {
                let new_origin = t_us - REBASE_MARGIN_US;
                let delta = (new_origin - origin) as f32;
                self.launch_shift_tmap(delta);
                self.origin_us = Some(new_origin);
            }
        }

        let w_i32 = self.w as i32;
        let h_i32 = self.h as i32;
        let n_px = self.w * self.h;
        let n_px_i32 = n_px as i32;
        let cfg = LaunchConfig::for_num_elems(n_px);

        if let Some(last_t) = self.last_render_t_us {
            let dt_us = (t_us - last_t).max(0) as f32;
            let tau_us = self.tau_leak_us.max(1.0);
            let decay_factor = (-dt_us / tau_us).exp();
            let mut decay_args = self.stream.launch_builder(&self.decay_state_fn);
            decay_args.arg(&mut self.f);
            decay_args.arg(&mut self.u);
            decay_args.arg(&decay_factor);
            decay_args.arg(&n_px_i32);
            unsafe { decay_args.launch(cfg) }.expect("cuda decay_state launch failed");
        }
        self.last_render_t_us = Some(t_us);

        let threshold = self.event_decimation_threshold.max(1);
        let decimation_p: f64 = if batch.len() > threshold { threshold as f64 / batch.len() as f64 } else { 1.0 };
        let keep_bound: u64 = ((decimation_p * (u64::MAX as f64)) as u64).max(1);
        let (eff_c_pos, eff_c_neg) =
            if decimation_p < 1.0 { (self.c_pos / decimation_p as f32, self.c_neg / decimation_p as f32) } else { (self.c_pos, self.c_neg) };

        let mut pack_ms = 0.0f64;
        let mut upload_ms = 0.0f64;
        let mut events_integrated = 0usize;
        if !batch.is_empty() {
            let origin = self.origin_us.expect("origin_us is set above whenever batch is non-empty");
            let t_pack0 = Instant::now();
            self.packed_scratch.clear();
            self.packed_scratch.reserve(batch.len() * 4);
            for e in &batch {
                if decimation_p < 1.0 && decim_hash(e.x, e.y, e.p, e.t_us) >= keep_bound {
                    continue;
                }
                let t_rel = (e.t_us - origin).max(0) as i32;
                self.packed_scratch.push(e.x as i32);
                self.packed_scratch.push(e.y as i32);
                self.packed_scratch.push(e.p as i32);
                self.packed_scratch.push(t_rel);
            }
            events_integrated = self.packed_scratch.len() / 4;
            pack_ms = t_pack0.elapsed().as_secs_f64() * 1000.0;

            if events_integrated > 0 {
                let t_upload0 = Instant::now();
                let d_evs = self.stream.clone_htod(&self.packed_scratch).expect("cuda events upload failed");
                upload_ms = t_upload0.elapsed().as_secs_f64() * 1000.0;
                let n_events = events_integrated as i32;
                let cfg = LaunchConfig::for_num_elems(events_integrated as u32);

                let mut args = self.stream.launch_builder(&self.integrate_events);
                args.arg(&d_evs);
                args.arg(&n_events);
                args.arg(&mut self.f);
                args.arg(&mut self.t_map);
                args.arg(&eff_c_pos);
                args.arg(&eff_c_neg);
                args.arg(&w_i32);
                args.arg(&h_i32);
                unsafe { args.launch(cfg) }.expect("cuda integrate_events launch failed");
            }
        }

        let t_kernel0 = Instant::now();

        {
            let mut clamp_args = self.stream.launch_builder(&self.clamp_f_fn);
            clamp_args.arg(&mut self.f);
            clamp_args.arg(&F_CLAMP_ABS);
            clamp_args.arg(&n_px_i32);
            unsafe { clamp_args.launch(cfg) }.expect("cuda clamp_f launch failed");
        }

        if self.iters > 0 {
            self.stream.memcpy_dtod(&self.u, &mut self.u_bar).expect("cuda u->u_bar copy failed");

            let t_now_rel: f32 = match self.origin_us {
                Some(origin) => (t_us - origin) as f32,
                None => 0.0,
            };

            if self.use_persistent_pd {
                let iters_i32 = self.iters as i32;
                let persistent_cfg = LaunchConfig {
                    grid_dim: (self.persistent_grid_dim, 1, 1),
                    block_dim: (self.persistent_block_dim, 1, 1),
                    shared_mem_bytes: 0,
                };
                let mut args = self.stream.launch_builder(&self.pd_solve_persistent_fn);
                args.arg(&mut self.u);
                args.arg(&mut self.u_bar);
                args.arg(&mut self.p_x);
                args.arg(&mut self.p_y);
                args.arg(&mut self.g);
                args.arg(&self.f);
                args.arg(&self.t_map);
                args.arg(&self.manifold_alpha);
                args.arg(&t_now_rel);
                args.arg(&PD_TAU);
                args.arg(&PD_SIGMA);
                args.arg(&self.lambda);
                args.arg(&w_i32);
                args.arg(&h_i32);
                args.arg(&iters_i32);
                unsafe { args.launch_cooperative(persistent_cfg) }
                    .expect("cuda pd_solve_persistent cooperative launch failed");
            } else {

                let mut g_args = self.stream.launch_builder(&self.compute_g);
                g_args.arg(&self.t_map);
                g_args.arg(&mut self.g);
                g_args.arg(&self.manifold_alpha);
                g_args.arg(&t_now_rel);
                g_args.arg(&n_px_i32);
                unsafe { g_args.launch(cfg) }.expect("cuda compute_g launch failed");

                for _ in 0..self.iters {
                    let mut dual_args = self.stream.launch_builder(&self.pd_dual);
                    dual_args.arg(&mut self.p_x);
                    dual_args.arg(&mut self.p_y);
                    dual_args.arg(&self.u_bar);
                    dual_args.arg(&self.g);
                    dual_args.arg(&PD_SIGMA);
                    dual_args.arg(&w_i32);
                    dual_args.arg(&h_i32);
                    unsafe { dual_args.launch(cfg) }.expect("cuda pd_dual launch failed");

                    let mut primal_args = self.stream.launch_builder(&self.pd_primal);
                    primal_args.arg(&mut self.u);
                    primal_args.arg(&mut self.u_bar);
                    primal_args.arg(&self.p_x);
                    primal_args.arg(&self.p_y);
                    primal_args.arg(&self.g);
                    primal_args.arg(&self.f);
                    primal_args.arg(&PD_TAU);
                    primal_args.arg(&self.lambda);
                    primal_args.arg(&w_i32);
                    primal_args.arg(&h_i32);
                    unsafe { primal_args.launch(cfg) }.expect("cuda pd_primal launch failed");
                }
            }
        }

        let src: &CudaSlice<f32> = if self.iters == 0 { &self.f } else { &self.u };

        match self.normalization {
            Normalization::Fixed => {
                let bias = 128.0f32;
                let scale = self.tonemap_scale;
                let mut tm_args = self.stream.launch_builder(&self.tonemap);
                tm_args.arg(src);
                tm_args.arg(&mut self.out_buf);
                tm_args.arg(&n_px_i32);
                tm_args.arg(&bias);
                tm_args.arg(&scale);
                unsafe { tm_args.launch(cfg) }.expect("cuda tonemap launch failed");
            }
            Normalization::Percentile => {
                self.stream
                    .memcpy_dtoh(src, &mut self.percentile_scratch)
                    .expect("cuda percentile scratch download failed");
                let n = self.percentile_scratch.len();
                let lo_idx = (((n - 1) as f32) * self.percentile_lo.clamp(0.0, 1.0)).round() as usize;
                let hi_idx = (((n - 1) as f32) * self.percentile_hi.clamp(0.0, 1.0)).round() as usize;
                self.percentile_scratch.select_nth_unstable_by(lo_idx, |a, b| a.total_cmp(b));
                let lo_val = self.percentile_scratch[lo_idx];
                self.percentile_scratch.select_nth_unstable_by(hi_idx, |a, b| a.total_cmp(b));
                let hi_val = self.percentile_scratch[hi_idx];
                let range = hi_val - lo_val;
                let (bias, scale) = if range > 1e-6 {
                    let mid_val = 0.5 * (lo_val + hi_val);
                    let scale = (255.0 / range).min(self.tonemap_scale.max(0.0));
                    (128.0 - scale * mid_val, scale)
                } else {
                    (128.0, 0.0)
                };

                let mut tm_args = self.stream.launch_builder(&self.tonemap);
                tm_args.arg(src);
                tm_args.arg(&mut self.out_buf);
                tm_args.arg(&n_px_i32);
                tm_args.arg(&bias);
                tm_args.arg(&scale);
                unsafe { tm_args.launch(cfg) }.expect("cuda tonemap launch failed");
            }
            Normalization::TileLocal => {
                let tile_w_i32 = TILE_PX_W as i32;
                let tile_h_i32 = TILE_PX_H as i32;
                let tiles_x_i32 = self.tiles_x as i32;
                let tiles_y_i32 = self.tiles_y as i32;
                let n_tiles = self.tiles_x * self.tiles_y;

                let tile_cfg = LaunchConfig {
                    grid_dim: (n_tiles, 1, 1),
                    block_dim: (TILE_REDUCE_THREADS, 1, 1),
                    shared_mem_bytes: 0,
                };
                let mut mm_args = self.stream.launch_builder(&self.tile_stats_fn);
                mm_args.arg(src);
                mm_args.arg(&mut self.tile_min);
                mm_args.arg(&mut self.tile_max);
                mm_args.arg(&mut self.tile_sum);
                mm_args.arg(&mut self.tile_sumsq);
                mm_args.arg(&w_i32);
                mm_args.arg(&h_i32);
                mm_args.arg(&tile_w_i32);
                mm_args.arg(&tile_h_i32);
                mm_args.arg(&tiles_x_i32);
                mm_args.arg(&tiles_y_i32);
                unsafe { mm_args.launch(tile_cfg) }.expect("cuda compute_tile_stats launch failed");

                let gain_cap = self.tonemap_scale.max(0.0);
                let bs_cfg = LaunchConfig::for_num_elems(n_tiles);
                let mut bs_args = self.stream.launch_builder(&self.tile_bias_scale_fn);
                bs_args.arg(&self.tile_min);
                bs_args.arg(&self.tile_max);
                bs_args.arg(&self.tile_sum);
                bs_args.arg(&self.tile_sumsq);
                bs_args.arg(&mut self.tile_bias);
                bs_args.arg(&mut self.tile_scale);
                bs_args.arg(&w_i32);
                bs_args.arg(&h_i32);
                bs_args.arg(&tile_w_i32);
                bs_args.arg(&tile_h_i32);
                bs_args.arg(&tiles_x_i32);
                bs_args.arg(&tiles_y_i32);
                bs_args.arg(&TILE_ROBUST_STD_K);
                bs_args.arg(&gain_cap);
                unsafe { bs_args.launch(bs_cfg) }.expect("cuda compute_tile_bias_scale launch failed");

                let mut tm_args = self.stream.launch_builder(&self.tonemap_tiled_fn);
                tm_args.arg(src);
                tm_args.arg(&mut self.out_buf);
                tm_args.arg(&w_i32);
                tm_args.arg(&h_i32);
                tm_args.arg(&self.tile_bias);
                tm_args.arg(&self.tile_scale);
                tm_args.arg(&tile_w_i32);
                tm_args.arg(&tile_h_i32);
                tm_args.arg(&tiles_x_i32);
                tm_args.arg(&tiles_y_i32);
                unsafe { tm_args.launch(cfg) }.expect("cuda tonemap_tiled launch failed");
            }
        }

        let out_src: &CudaSlice<u8> = if self.median_filter {
            let mut med_args = self.stream.launch_builder(&self.median_filter_fn);
            med_args.arg(&self.out_buf);
            med_args.arg(&mut self.median_scratch);
            med_args.arg(&w_i32);
            med_args.arg(&h_i32);
            unsafe { med_args.launch(cfg) }.expect("cuda median_filter_3x3 launch failed");
            &self.median_scratch
        } else {
            &self.out_buf
        };

        if self.iters > 0 {
            self.stream.memcpy_dtod(&self.u, &mut self.f).expect("cuda u->f copy failed");
        }

        self.stream.synchronize().expect("cuda stream sync (render stats) failed");
        let kernel_ms = t_kernel0.elapsed().as_secs_f64() * 1000.0;

        let t_download0 = Instant::now();
        self.stream
            .memcpy_dtoh(out_src, &mut out.data)
            .expect("cuda device->host copy failed");
        let download_ms = t_download0.elapsed().as_secs_f64() * 1000.0;

        self.last_stats =
            RenderStats { pack_ms, upload_ms, kernel_ms, download_ms, events_this_render, pending_len, decimation_p, events_integrated };
    }

    fn reset(&mut self) {
        self.pending.clear();
        self.origin_us = None;
        self.last_render_t_us = None;
        self.stream.memset_zeros(&mut self.f).expect("cuda buffer clear failed");
        self.stream.memset_zeros(&mut self.t_map).expect("cuda buffer clear failed");
        self.stream.memset_zeros(&mut self.u).expect("cuda buffer clear failed");
        self.stream.memset_zeros(&mut self.u_bar).expect("cuda buffer clear failed");
        self.stream.memset_zeros(&mut self.p_x).expect("cuda buffer clear failed");
        self.stream.memset_zeros(&mut self.p_y).expect("cuda buffer clear failed");
        self.stream.memset_zeros(&mut self.g).expect("cuda buffer clear failed");
        self.stream.memset_zeros(&mut self.out_buf).expect("cuda buffer clear failed");
    }

    fn params_ui(&mut self, ui: &mut egui::Ui) {
        ui.add(egui::Slider::new(&mut self.c_pos, 0.01..=1.0).text("c_pos"));
        ui.add(egui::Slider::new(&mut self.c_neg, 0.01..=1.0).text("c_neg"));
        ui.horizontal(|ui| {
            ui.label("tonemap:");
            ui.selectable_value(&mut self.normalization, Normalization::Fixed, "fixed");
            ui.selectable_value(&mut self.normalization, Normalization::Percentile, "percentile");
            ui.selectable_value(&mut self.normalization, Normalization::TileLocal, "tile-local");
        });
        match self.normalization {
            Normalization::Fixed => {
                ui.add(
                    egui::Slider::new(&mut self.tonemap_scale, 10.0..=1000.0)
                        .logarithmic(true)
                        .text("tonemap scale"),
                );
            }
            Normalization::Percentile => {
                ui.add(egui::Slider::new(&mut self.percentile_lo, 0.0..=0.49).text("percentile lo"));
                ui.add(egui::Slider::new(&mut self.percentile_hi, 0.51..=1.0).text("percentile hi"));
            }
            Normalization::TileLocal => {
                ui.add(
                    egui::Slider::new(&mut self.tonemap_scale, 10.0..=1000.0)
                        .logarithmic(true)
                        .text("tile gain cap"),
                );
            }
        }
        ui.checkbox(&mut self.median_filter, "3x3 median filter");
        ui.add(egui::Slider::new(&mut self.lambda, 0.01..=20.0).logarithmic(true).text("lambda"));
        ui.add(egui::Slider::new(&mut self.iters, 0..=100).text("PD iters"));
        ui.add(
            egui::Slider::new(&mut self.manifold_alpha, 1.0..=5000.0)
                .logarithmic(true)
                .text("manifold alpha"),
        );
        ui.add(
            egui::Slider::new(&mut self.tau_leak_us, 1_000.0..=5_000_000.0)
                .logarithmic(true)
                .text("leak time constant (us)"),
        );
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }


    fn push_events_device(&mut self, dev: &crate::gpu_decode::DecodedEventsDevice, stream: &Arc<CudaStream>) {
        if dev.len == 0 {
            return;
        }
        stream.synchronize().expect("cuda decoder-stream sync failed");

        if self.origin_us.is_none() {
            let first =
                self.stream.clone_dtoh(&dev.t_us.slice(0..1)).expect("cuda origin bootstrap download failed");
            self.origin_us = Some(first[0]);
        }
        let origin = self.origin_us.expect("origin_us set above");

        if self.soa_ingest_fn.is_none() {
            let ptx = compile_ptx_with_opts(SOA_INGEST_SRC, nvrtc_compile_opts())
                .expect("nvrtc compile (soa ingest) failed");
            let module = self.stream.context().load_module(ptx).expect("cuda module load (soa ingest) failed");
            self.soa_ingest_fn =
                Some(module.load_function("evt_soa_to_quads").expect("cuda function load (soa ingest) failed"));
        }

        let need = dev.len * 4;
        let realloc = match &self.soa_ingest_quads {
            Some(b) => b.len() < need,
            None => true,
        };
        if realloc {
            self.soa_ingest_quads =
                Some(self.stream.alloc_zeros::<i32>(need).expect("cuda soa ingest scratch alloc failed"));
        }

        let threshold = self.event_decimation_threshold.max(1);
        let decimation_p: f64 = if dev.len > threshold { threshold as f64 / dev.len as f64 } else { 1.0 };
        let keep_bound: u64 = ((decimation_p * (u64::MAX as f64)) as u64).max(1);
        let decimate_i32: i32 = (decimation_p < 1.0) as i32;
        let (eff_c_pos, eff_c_neg) = if decimation_p < 1.0 {
            (self.c_pos / decimation_p as f32, self.c_neg / decimation_p as f32)
        } else {
            (self.c_pos, self.c_neg)
        };

        let n_i32 = dev.len as i32;
        let cfg = LaunchConfig::for_num_elems(dev.len as u32);
        {
            let pack_fn = self.soa_ingest_fn.as_ref().expect("soa_ingest_fn set above");
            let quads = self.soa_ingest_quads.as_mut().expect("soa_ingest_quads set above");
            let mut args = self.stream.launch_builder(pack_fn);
            args.arg(&dev.x);
            args.arg(&dev.y);
            args.arg(&dev.p);
            args.arg(&dev.t_us);
            args.arg(&n_i32);
            args.arg(&origin);
            args.arg(&decimate_i32);
            args.arg(&keep_bound);
            args.arg(quads);
            unsafe { args.launch(cfg) }.expect("cuda evt_soa_to_quads launch failed");
        }
        {
            let quads = self.soa_ingest_quads.as_ref().expect("soa_ingest_quads set above");
            let w_i32 = self.w as i32;
            let h_i32 = self.h as i32;
            let mut args = self.stream.launch_builder(&self.integrate_events);
            args.arg(quads);
            args.arg(&n_i32);
            args.arg(&mut self.f);
            args.arg(&mut self.t_map);
            args.arg(&eff_c_pos);
            args.arg(&eff_c_neg);
            args.arg(&w_i32);
            args.arg(&h_i32);
            unsafe { args.launch(cfg) }.expect("cuda integrate_events (device ingest) launch failed");
        }
        self.stream.synchronize().expect("cuda stream sync (device ingest) failed");
    }
}

const SOA_INGEST_SRC: &str = r#"
extern "C" __global__ void evt_soa_to_quads(
    const unsigned short* xs,
    const unsigned short* ys,
    const unsigned char* ps,
    const long long* ts,
    int n,
    long long origin_us,
    int decimate,
    unsigned long long keep_bound,
    int* out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (decimate) {
        // decim_hash (cuda.rs), bit-for-bit: SplitMix64 finalizer over
        // (t_us, x, y, p).
        unsigned long long h = ((unsigned long long)ts[i]) * 0x9E3779B97F4A7C15ULL
            ^ (((unsigned long long)xs[i] << 32) | ((unsigned long long)ys[i] << 16) | (unsigned long long)ps[i]);
        h ^= h >> 30;
        h *= 0xBF58476D1CE4E5B9ULL;
        h ^= h >> 27;
        h *= 0x94D049BB133111EBULL;
        h ^= h >> 31;
        if (h >= keep_bound) {
            out[4 * i + 0] = -1; // sentinel: integrate_events skips e.x < 0
            return;
        }
    }
    long long tr = ts[i] - origin_us;
    if (tr < 0) tr = 0;
    if (tr > 2147483647LL) tr = 2147483647LL;
    out[4 * i + 0] = (int)xs[i];
    out[4 * i + 1] = (int)ys[i];
    out[4 * i + 2] = (int)ps[i];
    out[4 * i + 3] = (int)tr;
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuda_smoke() {
        let Ok(mut r) = CudaManifold::new(64, 64) else {
            eprintln!("skip: no CUDA");
            return;
        };
        r.iters = 0;
        r.normalization = Normalization::Fixed;
        let mut img = GrayImage::new(64, 64);
        r.render_at(0, &mut img);
        assert!(img.data.iter().all(|&v| v == 128), "expected uniform mid-gray fill");
    }

    fn region_mean(img: &GrayImage, x0: u16, x1: u16, y0: u16, y1: u16) -> f64 {
        let mut sum = 0u64;
        let mut n = 0u64;
        for y in y0..y1 {
            for x in x0..x1 {
                sum += img.data[y as usize * img.w as usize + x as usize] as u64;
                n += 1;
            }
        }
        sum as f64 / n as f64
    }

    #[test]
    fn cuda_integration_synthetic() {
        let Ok(mut r) = CudaManifold::new(64, 64) else {
            eprintln!("skip: no CUDA");
            return;
        };
        r.iters = 0;
        r.normalization = Normalization::Fixed;

        let mut events = Vec::with_capacity(2000);
        let mut t = 0i64;
        for _ in 0..10 {
            for y in 10..20u16 {
                for x in 10..20u16 {
                    events.push(Event { t_us: t, x, y, p: 1 });
                    t += 1;
                }
            }
        }
        for _ in 0..10 {
            for y in 40..50u16 {
                for x in 40..50u16 {
                    events.push(Event { t_us: t, x, y, p: 0 });
                    t += 1;
                }
            }
        }
        r.push_events(&EventBatch { events });

        let mut img = GrayImage::new(64, 64);
        r.render_at(2000, &mut img);

        let bright = region_mean(&img, 10, 20, 10, 20);
        let dark = region_mean(&img, 40, 50, 40, 50);
        let bg = img.data[5 * 64 + 5];

        assert!(bright > 140.0, "bright region mean {bright} should exceed 140");
        assert!(dark < 116.0, "dark region mean {dark} should be below 116");
        assert_eq!(bg, 128, "untouched background pixel should stay mid-gray");
    }

    #[test]
    fn cuda_origin_rebase() {
        let Ok(mut r) = CudaManifold::new(64, 64) else {
            eprintln!("skip: no CUDA");
            return;
        };
        r.iters = 0;
        r.normalization = Normalization::Fixed;
        let mut img = GrayImage::new(64, 64);

        r.push_events(&EventBatch { events: vec![Event { t_us: 0, x: 5, y: 5, p: 1 }] });
        r.render_at(0, &mut img);

        r.push_events(&EventBatch { events: vec![Event { t_us: 20_000_000, x: 30, y: 30, p: 1 }] });
        r.render_at(20_000_100, &mut img);

        let px = img.data[30 * 64 + 30];
        assert!(px > 128, "recent event after rebase should brighten its pixel, got {px}");
    }

    fn region_variance(img: &GrayImage, x0: u16, x1: u16, y0: u16, y1: u16) -> f64 {
        let mean = region_mean(img, x0, x1, y0, y1);
        let mut sq_sum = 0.0f64;
        let mut n = 0u64;
        for y in y0..y1 {
            for x in x0..x1 {
                let v = img.data[y as usize * img.w as usize + x as usize] as f64;
                sq_sum += (v - mean) * (v - mean);
                n += 1;
            }
        }
        sq_sum / n as f64
    }

    fn build_denoise_test_events() -> (Vec<Event>, i64, [(u16, u16); 6]) {
        let salt_positions: [(u16, u16); 6] = [(5, 5), (5, 58), (58, 5), (58, 58), (10, 50), (50, 10)];
        let mut events = Vec::new();
        let mut t = 0i64;
        for y in 20..40u16 {
            for x in 20..40u16 {
                let n = if (x + y) % 2 == 0 { 8 } else { 2 };
                for _ in 0..n {
                    events.push(Event { t_us: t, x, y, p: 1 });
                    t += 1;
                }
            }
        }
        for &(x, y) in &salt_positions {
            for _ in 0..5 {
                events.push(Event { t_us: t, x, y, p: 1 });
                t += 1;
            }
        }
        (events, t, salt_positions)
    }

    #[test]
    fn cuda_denoise_smooths() {
        let (events, t_end, salt_positions) = build_denoise_test_events();
        let batch = EventBatch { events };

        let Ok(mut r0) = CudaManifold::new(64, 64) else {
            eprintln!("skip: no CUDA");
            return;
        };
        r0.tonemap_scale = 100.0;
        r0.normalization = Normalization::Fixed;
        r0.iters = 0;
        r0.push_events(&batch);
        let mut img0 = GrayImage::new(64, 64);
        r0.render_at(t_end, &mut img0);

        let mut r30 = CudaManifold::new(64, 64).expect("cuda available (r0 construction above succeeded)");
        r30.tonemap_scale = 100.0;
        r30.normalization = Normalization::Fixed;
        r30.iters = 30;
        r30.push_events(&batch);
        let mut img30 = GrayImage::new(64, 64);
        r30.render_at(t_end, &mut img30);

        let var0 = region_variance(&img0, 20, 40, 20, 40);
        let var30 = region_variance(&img30, 20, 40, 20, 40);
        assert!(var30 < var0, "square variance should shrink with denoising: {var0} -> {var30}");

        let mean0 = region_mean(&img0, 20, 40, 20, 40);
        let mean30 = region_mean(&img30, 20, 40, 20, 40);
        assert!(
            (mean30 - mean0).abs() <= 20.0,
            "square mean should not wash out: img0={mean0} img30={mean30}"
        );

        for &(x, y) in &salt_positions {
            let dev0 = (img0.data[y as usize * 64 + x as usize] as f64 - 128.0).abs();
            let dev30 = (img30.data[y as usize * 64 + x as usize] as f64 - 128.0).abs();
            assert!(
                dev30 < dev0,
                "salt pixel ({x},{y}) deviation from background should shrink: {dev0} -> {dev30}"
            );
        }
    }


    #[test]
    fn cuda_persistent_pd_matches_fallback_loop() {
        let (events, t_end, _salt_positions) = build_denoise_test_events();
        let batch = EventBatch { events };
        const ITERS: u32 = 20;

        let Ok(mut r_persistent) = CudaManifold::new(64, 64) else {
            eprintln!("skip: no CUDA");
            return;
        };
        r_persistent.tonemap_scale = 100.0;
        r_persistent.normalization = Normalization::Fixed;
        r_persistent.iters = ITERS;
        r_persistent.push_events(&batch);
        let mut img_persistent = GrayImage::new(64, 64);
        r_persistent.render_at(t_end, &mut img_persistent);

        assert!(
            r_persistent.use_persistent_pd,
            "persistent-kernel path should be active on this machine (see CudaManifold::new's \
             feasibility probe) -- if this fails, the parity check below never actually \
             exercises pd_solve_persistent"
        );

        let mut r_fallback = CudaManifold::new(64, 64)
            .expect("cuda available (r_persistent construction above succeeded)");
        r_fallback.tonemap_scale = 100.0;
        r_fallback.normalization = Normalization::Fixed;
        r_fallback.iters = ITERS;
        r_fallback.use_persistent_pd = false;
        r_fallback.push_events(&batch);
        let mut img_fallback = GrayImage::new(64, 64);
        r_fallback.render_at(t_end, &mut img_fallback);

        let max_diff = img_persistent
            .data
            .iter()
            .zip(img_fallback.data.iter())
            .map(|(&a, &b)| (a as i32 - b as i32).abs())
            .max()
            .unwrap_or(0);
        assert!(
            max_diff <= 1,
            "persistent-kernel and fallback-loop outputs should match within 1 LSB, got max_diff={max_diff}"
        );
    }

    #[test]
    fn cuda_adjoint_energy_decreases() {
        let Ok(mut r) = CudaManifold::new(32, 32) else {
            eprintln!("skip: no CUDA");
            return;
        };
        let w = 32usize;
        let h = 32usize;
        let n = w * h;
        r.lambda = 2.0;

        let t_now_rel: f32 = 10_000.0;
        let mut t_map_host = vec![0.0f32; n];
        for y in 0..h {
            for x in 0..w {
                t_map_host[y * w + x] = if x < w / 2 { 0.0 } else { t_now_rel };
            }
        }
        let mut f_host = vec![0.0f32; n];
        for y in 0..h {
            for x in 0..w {
                f_host[y * w + x] = if (x + y) % 2 == 0 { 0.5 } else { -0.5 };
            }
        }

        r.stream.memcpy_htod(&t_map_host, &mut r.t_map).expect("t_map upload failed");
        r.stream.memcpy_htod(&f_host, &mut r.f).expect("f upload failed");

        let w_i32 = w as i32;
        let h_i32 = h as i32;
        let n_i32 = n as i32;
        let cfg = LaunchConfig::for_num_elems(n as u32);

        let mut g_args = r.stream.launch_builder(&r.compute_g);
        g_args.arg(&r.t_map);
        g_args.arg(&mut r.g);
        g_args.arg(&r.manifold_alpha);
        g_args.arg(&t_now_rel);
        g_args.arg(&n_i32);
        unsafe { g_args.launch(cfg) }.expect("compute_g launch failed");

        fn download(stream: &Arc<CudaStream>, buf: &CudaSlice<f32>, n: usize) -> Vec<f32> {
            let mut host = vec![0.0f32; n];
            stream.memcpy_dtoh(buf, &mut host).expect("download failed");
            host
        }

        let g_host = download(&r.stream, &r.g, n);
        assert!(g_host[0] > 0.9, "left (stale) region should have g near 1, got {}", g_host[0]);
        assert!(g_host[w - 1] < 0.1, "right (fresh) region should have g near 0, got {}", g_host[w - 1]);

        fn energy(u: &[f32], g: &[f32], f: &[f32], w: usize, h: usize, lambda: f32) -> f64 {
            let mut tv = 0.0f64;
            let mut data = 0.0f64;
            for y in 0..h {
                for x in 0..w {
                    let i = y * w + x;
                    let gx = if x < w - 1 { u[i + 1] - u[i] } else { 0.0 };
                    let gy = if y < h - 1 { u[i + w] - u[i] } else { 0.0 };
                    tv += (g[i] as f64) * f64::from(gx).hypot(f64::from(gy));
                    let d = (u[i] - f[i]) as f64;
                    data += d * d;
                }
            }
            tv + 0.5 * (lambda as f64) * data
        }

        let mut energies = vec![energy(&download(&r.stream, &r.u, n), &g_host, &f_host, w, h, r.lambda)];

        for _ in 0..4 {
            for _ in 0..50 {
                let mut dual_args = r.stream.launch_builder(&r.pd_dual);
                dual_args.arg(&mut r.p_x);
                dual_args.arg(&mut r.p_y);
                dual_args.arg(&r.u_bar);
                dual_args.arg(&r.g);
                dual_args.arg(&PD_SIGMA);
                dual_args.arg(&w_i32);
                dual_args.arg(&h_i32);
                unsafe { dual_args.launch(cfg) }.expect("pd_dual launch failed");

                let mut primal_args = r.stream.launch_builder(&r.pd_primal);
                primal_args.arg(&mut r.u);
                primal_args.arg(&mut r.u_bar);
                primal_args.arg(&r.p_x);
                primal_args.arg(&r.p_y);
                primal_args.arg(&r.g);
                primal_args.arg(&r.f);
                primal_args.arg(&PD_TAU);
                primal_args.arg(&r.lambda);
                primal_args.arg(&w_i32);
                primal_args.arg(&h_i32);
                unsafe { primal_args.launch(cfg) }.expect("pd_primal launch failed");
            }
            energies.push(energy(&download(&r.stream, &r.u, n), &g_host, &f_host, w, h, r.lambda));
        }

        for pair in energies.windows(2) {
            let (prev, next) = (pair[0], pair[1]);
            assert!(
                next <= prev * (1.0 + 1e-3),
                "primal energy should not increase across 50-iteration checkpoints: {energies:?}"
            );
        }

        let u_probe: Vec<f32> = (0..n).map(|i| ((i * 37 + 11) % 97) as f32 / 97.0 - 0.5).collect();
        let px_probe: Vec<f32> = (0..n).map(|i| ((i * 53 + 7) % 89) as f32 / 89.0 - 0.5).collect();
        let py_probe: Vec<f32> = (0..n).map(|i| ((i * 61 + 3) % 83) as f32 / 83.0 - 0.5).collect();
        r.stream.memcpy_htod(&u_probe, &mut r.u).expect("u probe upload failed");
        r.stream.memcpy_htod(&px_probe, &mut r.p_x).expect("p_x probe upload failed");
        r.stream.memcpy_htod(&py_probe, &mut r.p_y).expect("p_y probe upload failed");

        let mut lhs = 0.0f64;
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                let gx = if x < w - 1 { u_probe[i + 1] - u_probe[i] } else { 0.0 };
                let gy = if y < h - 1 { u_probe[i + w] - u_probe[i] } else { 0.0 };
                lhs += f64::from(g_host[i]) * (f64::from(gx) * f64::from(px_probe[i]) + f64::from(gy) * f64::from(py_probe[i]));
            }
        }

        let zero_f = vec![0.0f32; n];
        r.stream.memcpy_htod(&zero_f, &mut r.f).expect("f probe upload failed");
        let tau1: f32 = 1.0;
        let lambda0: f32 = 0.0;
        let mut probe_args = r.stream.launch_builder(&r.pd_primal);
        probe_args.arg(&mut r.u);
        probe_args.arg(&mut r.u_bar);
        probe_args.arg(&r.p_x);
        probe_args.arg(&r.p_y);
        probe_args.arg(&r.g);
        probe_args.arg(&r.f);
        probe_args.arg(&tau1);
        probe_args.arg(&lambda0);
        probe_args.arg(&w_i32);
        probe_args.arg(&h_i32);
        unsafe { probe_args.launch(cfg) }.expect("pd_primal adjoint-probe launch failed");

        let u_after = download(&r.stream, &r.u, n);
        let mut rhs = 0.0f64;
        for i in 0..n {
            let div_gp_i = f64::from(u_after[i] - u_probe[i]);
            rhs += -f64::from(u_probe[i]) * div_gp_i;
        }

        let reldiff = (lhs - rhs).abs() / lhs.abs().max(1e-9);
        assert!(
            reldiff < 1e-4,
            "adjoint identity <K(u),p> == -<u,div(g*p)> violated (K* must match K=g*grad exactly): lhs={lhs} rhs={rhs} reldiff={reldiff}"
        );
    }

    #[test]
    fn cuda_percentile_normalization_spans_range() {
        let Ok(mut r) = CudaManifold::new(64, 64) else {
            eprintln!("skip: no CUDA");
            return;
        };
        r.iters = 0;
        r.normalization = Normalization::Percentile;
        r.percentile_lo = 0.02;
        r.percentile_hi = 0.98;

        let n = 64 * 64;
        let f_host: Vec<f32> = (0..n).map(|i| -1.5 + 3.0 * (i as f32) / ((n - 1) as f32)).collect();
        r.stream.memcpy_htod(&f_host, &mut r.f).expect("f upload failed");

        let mut img = GrayImage::new(64, 64);
        r.render_at(0, &mut img);

        let min = *img.data.iter().min().expect("non-empty image");
        let max = *img.data.iter().max().expect("non-empty image");
        assert!(min < 20, "percentile-normalized gradient should stretch near black, got min={min}");
        assert!(max > 235, "percentile-normalized gradient should stretch near white, got max={max}");
    }


    #[test]
    fn cuda_median_filter_removes_impulse() {
        let Ok(mut r) = CudaManifold::new(64, 64) else {
            eprintln!("skip: no CUDA");
            return;
        };
        r.iters = 0;
        r.normalization = Normalization::Fixed;
        r.tonemap_scale = 100.0;

        let w = 64usize;
        let n = w * w;
        let mut f_host = vec![0.0f32; n];
        let impulse_idx = 32 * w + 32;
        f_host[impulse_idx] = 1.5;
        r.stream.memcpy_htod(&f_host, &mut r.f).expect("f upload failed");

        let mut img_raw = GrayImage::new(64, 64);
        r.render_at(0, &mut img_raw);
        let raw_val = img_raw.data[impulse_idx];
        let raw_bg = img_raw.data[impulse_idx - 1];
        assert_eq!(raw_bg, 128, "unfiltered background pixel should stay mid-gray");
        assert_eq!(raw_val, 255, "unfiltered impulse should saturate to white");

        r.median_filter = true;
        let mut img_filtered = GrayImage::new(64, 64);
        r.render_at(0, &mut img_filtered);
        let filt_val = img_filtered.data[impulse_idx];
        let filt_bg = img_filtered.data[impulse_idx - 1];
        assert_eq!(
            filt_val, 128,
            "3x3 median filter should fully remove an isolated single-pixel impulse (8/9 neighborhood samples are background)"
        );
        assert_eq!(filt_bg, 128, "neighboring background pixels should be unaffected by median filtering");
    }


    #[test]
    fn cuda_tile_local_normalization_stretches_both_regions() {
        let w = 320u32;
        let h = 72u32;
        let Ok(mut r) = CudaManifold::new(w, h) else {
            eprintln!("skip: no CUDA");
            return;
        };
        r.iters = 0;
        r.normalization = Normalization::TileLocal;
        r.tonemap_scale = 1_000_000.0;

        let n = (w * h) as usize;
        let mut f_host = vec![0.0f32; n];
        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) as usize;
                let local = if x % 2 == 0 { 0.0f32 } else { 0.02f32 };
                f_host[idx] = if x < w / 2 { local } else { 1.0 + local };
            }
        }
        r.stream.memcpy_htod(&f_host, &mut r.f).expect("f upload failed");

        let mut img = GrayImage::new(w, h);
        r.render_at(0, &mut img);

        let region_min_max = |x0: u32, x1: u32| -> (u8, u8) {
            let mut mn = 255u8;
            let mut mx = 0u8;
            for y in 0..h {
                for x in x0..x1 {
                    let v = img.data[(y * w + x) as usize];
                    mn = mn.min(v);
                    mx = mx.max(v);
                }
            }
            (mn, mx)
        };

        let (a_min, a_max) = region_min_max(0, w / 2);
        let (b_min, b_max) = region_min_max(w / 2, w);

        assert!(a_min < 20, "region A (f near 0.0) should stretch near black, got min={a_min}");
        assert!(a_max > 235, "region A (f near 0.0) should stretch near white, got max={a_max}");
        assert!(
            b_min < 20,
            "region B (f near 1.0, offset far from region A) should ALSO stretch near black, got min={b_min}"
        );
        assert!(
            b_max > 235,
            "region B (f near 1.0, offset far from region A) should ALSO stretch near white, got max={b_max}"
        );
    }

    #[test]
    fn cuda_decimation_preserves_expected_contribution() {
        let Ok(mut r_decim) = CudaManifold::new(64, 64) else {
            eprintln!("skip: no CUDA");
            return;
        };
        let mut r_plain = CudaManifold::new(64, 64).expect("cuda available (r_decim construction above succeeded)");

        let n = 500_000;
        let c_pos = 1.0f32 / (n as f32);
        let mut events = Vec::with_capacity(n);
        let mut t = 0i64;
        for _ in 0..n {
            events.push(Event { t_us: t, x: 20, y: 20, p: 1 });
            t += 1;
        }
        let batch = EventBatch { events };

        for r in [&mut r_decim, &mut r_plain] {
            r.iters = 0;
            r.normalization = Normalization::Fixed;
            r.c_pos = c_pos;
            r.tonemap_scale = 100.0;
        }
        r_decim.event_decimation_threshold = 1_000;
        r_plain.event_decimation_threshold = 10_000_000;

        r_decim.push_events(&batch);
        r_plain.push_events(&batch);

        let mut img_decim = GrayImage::new(64, 64);
        let mut img_plain = GrayImage::new(64, 64);
        r_decim.render_at(t, &mut img_decim);
        r_plain.render_at(t, &mut img_plain);

        let stats = r_decim.last_render_stats();
        assert!(stats.decimation_p < 1.0, "decimation should have engaged, got p={}", stats.decimation_p);
        assert!(
            stats.events_integrated < stats.events_this_render,
            "decimation should integrate fewer events than were drained: integrated={} drained={}",
            stats.events_integrated,
            stats.events_this_render
        );
        let plain_stats = r_plain.last_render_stats();
        assert_eq!(plain_stats.decimation_p, 1.0, "huge threshold should never engage the valve");

        let px_decim = img_decim.data[20 * 64 + 20] as i32;
        let px_plain = img_plain.data[20 * 64 + 20] as i32;
        assert!(
            (px_decim - px_plain).abs() <= 40,
            "decimated pixel {px_decim} should be statistically close to undecimated {px_plain} (contrast compensation should preserve expectation)"
        );
    }



    #[test]
    fn push_events_device_override_matches_host_path() {
        use crate::gpu_decode::GpuEventDecoder;

        let Ok(mut dec) = GpuEventDecoder::new() else {
            eprintln!("skip: no CUDA");
            return;
        };
        let (Ok(mut r_dev), Ok(mut r_host)) = (CudaManifold::new(64, 64), CudaManifold::new(64, 64)) else {
            eprintln!("skip: no CUDA");
            return;
        };
        for r in [&mut r_dev, &mut r_host] {
            r.iters = 0;
            r.normalization = Normalization::Fixed;
            r.tonemap_scale = 100.0;
        }

        let word = |ty: u16, payload: u16| (ty << 12) | (payload & 0x0FFF);
        let batch1: Vec<u16> = vec![
            word(0x8, 2),
            word(0x6, 100),
            word(0x0, 10),
            word(0x2, (1 << 11) | 10),
            word(0x3, 20),
            word(0x4, 0b101),
        ];
        let batch2: Vec<u16> = vec![
            word(0x8, 3),
            word(0x6, 7),
            word(0x0, 30),
            word(0x2, (1 << 11) | 31),
        ];

        let mut state = crate::gpu_decode::CarryState::default();
        let mut host_events: Vec<Event> = Vec::new();
        for words in [&batch1, &batch2] {
            let (dev, _trig, _stats) = dec.decode_with_state(words, 2, &mut state).expect("decode failed");
            r_dev.push_events_device(&dev, dec.stream());
            host_events.extend(
                dev.download(dec.stream())
                    .expect("download failed")
                    .into_iter()
                    .map(|e| Event { t_us: e.t_us, x: e.x, y: e.y, p: e.p as i8 }),
            );
        }
        r_host.push_events(&EventBatch { events: host_events });

        let t_render = 3 * 4096 + 7;
        let mut img_dev = GrayImage::new(64, 64);
        let mut img_host = GrayImage::new(64, 64);
        r_dev.render_at(t_render, &mut img_dev);
        r_host.render_at(t_render, &mut img_host);

        assert_eq!(img_dev.data, img_host.data, "device-ingest render must be byte-identical to the host path");
        assert!(img_dev.data[10 * 64 + 10] > 128, "ON event at (10,10) should brighten");
        assert!(img_dev.data[10 * 64 + 20] < 128, "OFF event at (20,10) should darken");
        assert!(img_dev.data[30 * 64 + 31] > 128, "ON event at (31,30) should brighten");
    }


    #[test]
    fn push_events_device_origin_rebootstraps_after_reset() {
        use crate::gpu_decode::GpuEventDecoder;

        let Ok(mut dec) = GpuEventDecoder::new() else {
            eprintln!("skip: no CUDA");
            return;
        };
        let Ok(mut r) = CudaManifold::new(64, 64) else {
            eprintln!("skip: no CUDA");
            return;
        };
        r.iters = 0;
        r.normalization = Normalization::Fixed;

        let word = |ty: u16, payload: u16| (ty << 12) | (payload & 0x0FFF);
        let words: Vec<u16> = vec![word(0x8, 3000), word(0x6, 0), word(0x0, 5), word(0x2, (1 << 11) | 5)];
        let (dev, _t, _s) = dec.decode(&words, 1024).expect("decode failed");
        r.push_events_device(&dev, dec.stream());
        let mut img = GrayImage::new(64, 64);
        r.render_at(3000 * 4096 + 1, &mut img);
        assert!(img.data[5 * 64 + 5] > 128);

        r.reset();
        let words2: Vec<u16> = vec![word(0x8, 1), word(0x6, 0), word(0x0, 7), word(0x2, (1 << 11) | 7)];
        let mut state2 = crate::gpu_decode::CarryState::default();
        let (dev2, _t2, _s2) = dec.decode_with_state(&words2, 1024, &mut state2).expect("decode failed");
        r.push_events_device(&dev2, dec.stream());
        r.render_at(4096 + 1, &mut img);
        assert!(
            img.data[7 * 64 + 7] > 128,
            "post-reset push must re-bootstrap origin from the new stream, not the pre-reset one"
        );
    }

    #[test]
    fn push_events_device_decimation_preserves_expected_contribution() {
        use crate::gpu_decode::GpuEventDecoder;
        use crate::Reconstructor as _;

        let Ok(mut dec) = GpuEventDecoder::new() else {
            eprintln!("skip: no CUDA");
            return;
        };
        let (Ok(mut r_decim), Ok(mut r_plain)) = (CudaManifold::new(64, 64), CudaManifold::new(64, 64)) else {
            eprintln!("skip: no CUDA");
            return;
        };
        let mut r_host = CudaManifold::new(64, 64).expect("cuda available (constructions above succeeded)");

        let n: usize = 200_000;
        let c_pos = 1.0f32 / (n as f32);
        let word = |ty: u16, payload: u16| (ty << 12) | (payload & 0x0FFF);
        let mut words: Vec<u16> = Vec::with_capacity(2 * n + n / 4096 + 2);
        words.push(word(0x0, 20));
        for t in 0..n {
            if t % 4096 == 0 {
                words.push(word(0x8, (t >> 12) as u16));
            }
            words.push(word(0x6, (t & 0xFFF) as u16));
            words.push(word(0x2, (1 << 11) | 20));
        }
        let (dev, _trig, _stats) = dec.decode(&words, 65_536).expect("decode failed");
        assert_eq!(dev.len, n);

        for r in [&mut r_decim, &mut r_plain, &mut r_host] {
            r.iters = 0;
            r.normalization = Normalization::Fixed;
            r.c_pos = c_pos;
            r.tonemap_scale = 100.0;
        }
        r_decim.event_decimation_threshold = 1_000;
        r_host.event_decimation_threshold = 1_000;
        r_plain.event_decimation_threshold = 10_000_000;

        r_decim.push_events_device(&dev, dec.stream());
        r_plain.push_events_device(&dev, dec.stream());
        let host_events: Vec<Event> = dev
            .download(dec.stream())
            .expect("download failed")
            .into_iter()
            .map(|e| Event { t_us: e.t_us, x: e.x, y: e.y, p: e.p as i8 })
            .collect();
        r_host.push_events(&EventBatch { events: host_events });

        let t_end = n as i64;
        let mut img_decim = GrayImage::new(64, 64);
        let mut img_plain = GrayImage::new(64, 64);
        let mut img_host = GrayImage::new(64, 64);
        r_decim.render_at(t_end, &mut img_decim);
        r_plain.render_at(t_end, &mut img_plain);
        r_host.render_at(t_end, &mut img_host);

        let host_stats = r_host.last_render_stats();
        assert!(host_stats.decimation_p < 1.0, "host valve should have engaged, got p={}", host_stats.decimation_p);
        assert!(host_stats.events_integrated < host_stats.events_this_render);

        let px_decim = img_decim.data[20 * 64 + 20] as i32;
        let px_plain = img_plain.data[20 * 64 + 20] as i32;
        let px_host = img_host.data[20 * 64 + 20] as i32;
        assert!(
            (px_decim - px_plain).abs() <= 40,
            "device-decimated pixel {px_decim} should be statistically close to undecimated {px_plain}"
        );
        assert!(
            (px_decim - px_host).abs() <= 2,
            "device valve ({px_decim}) must keep the same subset as the host valve ({px_host})"
        );
    }


    #[test]
    #[ignore = "one-time feasibility probe, not a correctness check"]
    fn cooperative_launch_feasibility_check() {
        use cudarc::driver::sys::CUdevice_attribute;
        let Ok(ctx) = CudaContext::new(0) else {
            eprintln!("skip: no CUDA");
            return;
        };
        let coop = ctx.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH).unwrap();
        let sm_count = ctx.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT).unwrap();
        eprintln!("CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH = {coop} (1 = supported, 0 = not)");
        eprintln!("CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = {sm_count}");
    }

    #[test]
    #[ignore = "profiling tool, not a correctness check -- run explicitly with --ignored --nocapture"]
    fn kernel_profile() {
        use cudarc::driver::sys::CUevent_flags;

        let Ok(mut r) = CudaManifold::new(1280, 720) else {
            eprintln!("skip: no CUDA");
            return;
        };
        r.tau_leak_us = 2_000_000.0;
        r.tonemap_scale = 400.0;
        r.normalization = Normalization::TileLocal;
        r.median_filter = true;

        let w = 1280usize;
        let h = 720usize;
        let n = w * h;
        let w_i32 = w as i32;
        let h_i32 = h as i32;
        let n_i32 = n as i32;
        let cfg = LaunchConfig::for_num_elems(n as u32);
        const ITERS: u32 = 100;

        let ctx = r.stream.context().clone();
        let timed = || ctx.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT)).expect("event create");

        let f_host: Vec<f32> = (0..n).map(|i| ((i % 997) as f32 / 997.0) - 0.5).collect();
        let t_map_host: Vec<f32> = (0..n).map(|i| (i % 500_000) as f32).collect();
        r.stream.memcpy_htod(&f_host, &mut r.f).expect("f upload");
        r.stream.memcpy_htod(&t_map_host, &mut r.t_map).expect("t_map upload");
        r.stream.synchronize().expect("presync");

        const PEAK_EVENTS: usize = 4_272_763;
        let mut packed: Vec<i32> = Vec::with_capacity(PEAK_EVENTS * 4);
        for i in 0..PEAK_EVENTS {
            packed.push((i % w) as i32);
            packed.push((i % h) as i32);
            packed.push((i % 2) as i32);
            packed.push(i as i32);
        }
        let t_upload0 = Instant::now();
        let d_evs = r.stream.clone_htod(&packed).expect("events upload");
        let upload_ms = t_upload0.elapsed().as_secs_f64() * 1000.0;

        let mut out_host = vec![0u8; n];
        let t_dl0 = Instant::now();
        r.stream.memcpy_dtoh(&r.out_buf, &mut out_host).expect("out download");
        let download_ms = t_dl0.elapsed().as_secs_f64() * 1000.0;

        let n_events_i32 = PEAK_EVENTS as i32;
        let evt_cfg = LaunchConfig::for_num_elems(PEAK_EVENTS as u32);
        let (e0, e1) = (timed(), timed());
        e0.record(&r.stream).unwrap();
        {
            let mut args = r.stream.launch_builder(&r.integrate_events);
            args.arg(&d_evs);
            args.arg(&n_events_i32);
            args.arg(&mut r.f);
            args.arg(&mut r.t_map);
            args.arg(&r.c_pos);
            args.arg(&r.c_neg);
            args.arg(&w_i32);
            args.arg(&h_i32);
            unsafe { args.launch(evt_cfg) }.expect("integrate_events launch");
        }
        e1.record(&r.stream).unwrap();
        let integrate_ms = e0.elapsed_ms(&e1).unwrap();

        let (e0, e1) = (timed(), timed());
        e0.record(&r.stream).unwrap();
        {
            let mut args = r.stream.launch_builder(&r.clamp_f_fn);
            args.arg(&mut r.f);
            args.arg(&F_CLAMP_ABS);
            args.arg(&n_i32);
            unsafe { args.launch(cfg) }.expect("clamp_f launch");
        }
        e1.record(&r.stream).unwrap();
        let clamp_ms = e0.elapsed_ms(&e1).unwrap();

        let t_now_rel: f32 = 500_000.0;
        let (e0, e1) = (timed(), timed());
        e0.record(&r.stream).unwrap();
        {
            let mut args = r.stream.launch_builder(&r.compute_g);
            args.arg(&r.t_map);
            args.arg(&mut r.g);
            args.arg(&r.manifold_alpha);
            args.arg(&t_now_rel);
            args.arg(&n_i32);
            unsafe { args.launch(cfg) }.expect("compute_g launch");
        }
        e1.record(&r.stream).unwrap();
        let compute_g_ms = e0.elapsed_ms(&e1).unwrap();

        let (e0, e1) = (timed(), timed());
        e0.record(&r.stream).unwrap();
        for _ in 0..ITERS {
            let mut dual_args = r.stream.launch_builder(&r.pd_dual);
            dual_args.arg(&mut r.p_x);
            dual_args.arg(&mut r.p_y);
            dual_args.arg(&r.u_bar);
            dual_args.arg(&r.g);
            dual_args.arg(&PD_SIGMA);
            dual_args.arg(&w_i32);
            dual_args.arg(&h_i32);
            unsafe { dual_args.launch(cfg) }.expect("pd_dual launch");

            let mut primal_args = r.stream.launch_builder(&r.pd_primal);
            primal_args.arg(&mut r.u);
            primal_args.arg(&mut r.u_bar);
            primal_args.arg(&r.p_x);
            primal_args.arg(&r.p_y);
            primal_args.arg(&r.g);
            primal_args.arg(&r.f);
            primal_args.arg(&PD_TAU);
            primal_args.arg(&r.lambda);
            primal_args.arg(&w_i32);
            primal_args.arg(&h_i32);
            unsafe { primal_args.launch(cfg) }.expect("pd_primal launch");
        }
        e1.record(&r.stream).unwrap();
        let combined_total_ms = e0.elapsed_ms(&e1).unwrap();

        let (e0, e1) = (timed(), timed());
        e0.record(&r.stream).unwrap();
        for _ in 0..ITERS {
            let mut dual_args = r.stream.launch_builder(&r.pd_dual);
            dual_args.arg(&mut r.p_x);
            dual_args.arg(&mut r.p_y);
            dual_args.arg(&r.u_bar);
            dual_args.arg(&r.g);
            dual_args.arg(&PD_SIGMA);
            dual_args.arg(&w_i32);
            dual_args.arg(&h_i32);
            unsafe { dual_args.launch(cfg) }.expect("pd_dual launch");
        }
        e1.record(&r.stream).unwrap();
        let dual_only_total_ms = e0.elapsed_ms(&e1).unwrap();

        let dual_avg_ms = (dual_only_total_ms as f64) / (ITERS as f64);
        let primal_avg_ms = ((combined_total_ms - dual_only_total_ms) as f64) / (ITERS as f64);

        let tile_w_i32 = TILE_PX_W as i32;
        let tile_h_i32 = TILE_PX_H as i32;
        let tiles_x_i32 = r.tiles_x as i32;
        let tiles_y_i32 = r.tiles_y as i32;
        let n_tiles = r.tiles_x * r.tiles_y;
        let tile_cfg = LaunchConfig {
            grid_dim: (n_tiles, 1, 1),
            block_dim: (TILE_REDUCE_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let (e0, e1) = (timed(), timed());
        e0.record(&r.stream).unwrap();
        {
            let mut args = r.stream.launch_builder(&r.tile_stats_fn);
            args.arg(&r.f);
            args.arg(&mut r.tile_min);
            args.arg(&mut r.tile_max);
            args.arg(&mut r.tile_sum);
            args.arg(&mut r.tile_sumsq);
            args.arg(&w_i32);
            args.arg(&h_i32);
            args.arg(&tile_w_i32);
            args.arg(&tile_h_i32);
            args.arg(&tiles_x_i32);
            args.arg(&tiles_y_i32);
            unsafe { args.launch(tile_cfg) }.expect("compute_tile_stats launch");
        }
        e1.record(&r.stream).unwrap();
        let tile_stats_ms = e0.elapsed_ms(&e1).unwrap();

        let (e0, e1) = (timed(), timed());
        e0.record(&r.stream).unwrap();
        {
            let gain_cap: f32 = 200.0;
            let mut args = r.stream.launch_builder(&r.tile_bias_scale_fn);
            args.arg(&r.tile_min);
            args.arg(&r.tile_max);
            args.arg(&r.tile_sum);
            args.arg(&r.tile_sumsq);
            args.arg(&mut r.tile_bias);
            args.arg(&mut r.tile_scale);
            args.arg(&w_i32);
            args.arg(&h_i32);
            args.arg(&tile_w_i32);
            args.arg(&tile_h_i32);
            args.arg(&tiles_x_i32);
            args.arg(&tiles_y_i32);
            args.arg(&TILE_ROBUST_STD_K);
            args.arg(&gain_cap);
            let bs_cfg = LaunchConfig::for_num_elems(n_tiles);
            unsafe { args.launch(bs_cfg) }.expect("compute_tile_bias_scale launch");
        }
        e1.record(&r.stream).unwrap();
        let tile_bias_scale_ms = e0.elapsed_ms(&e1).unwrap();

        let (e0, e1) = (timed(), timed());
        e0.record(&r.stream).unwrap();
        {
            let mut args = r.stream.launch_builder(&r.tonemap_tiled_fn);
            args.arg(&r.f);
            args.arg(&mut r.out_buf);
            args.arg(&w_i32);
            args.arg(&h_i32);
            args.arg(&r.tile_bias);
            args.arg(&r.tile_scale);
            args.arg(&tile_w_i32);
            args.arg(&tile_h_i32);
            args.arg(&tiles_x_i32);
            args.arg(&tiles_y_i32);
            unsafe { args.launch(cfg) }.expect("tonemap_tiled launch");
        }
        e1.record(&r.stream).unwrap();
        let tonemap_tiled_ms = e0.elapsed_ms(&e1).unwrap();

        let (e0, e1) = (timed(), timed());
        e0.record(&r.stream).unwrap();
        {
            let mut args = r.stream.launch_builder(&r.median_filter_fn);
            args.arg(&r.out_buf);
            args.arg(&mut r.median_scratch);
            args.arg(&w_i32);
            args.arg(&h_i32);
            unsafe { args.launch(cfg) }.expect("median_filter_3x3 launch");
        }
        e1.record(&r.stream).unwrap();
        let median_ms = e0.elapsed_ms(&e1).unwrap();

        r.stream.synchronize().unwrap();

        eprintln!();
        eprintln!("=== kernel_profile @ 1280x720, {PEAK_EVENTS} events (peak golden batch), {ITERS} PD iters ===");
        eprintln!("transfers:   upload(events)={upload_ms:.3}ms  download(out u8)={download_ms:.3}ms");
        eprintln!("integrate_events:        {integrate_ms:.3}ms  ({PEAK_EVENTS} events)");
        eprintln!("clamp_f:                 {clamp_ms:.3}ms");
        eprintln!("compute_g:               {compute_g_ms:.3}ms");
        eprintln!("pd_dual   (avg/iter):    {dual_avg_ms:.4}ms  x{ITERS} = {:.3}ms", dual_avg_ms * ITERS as f64);
        eprintln!("pd_primal (avg/iter):    {primal_avg_ms:.4}ms  x{ITERS} = {:.3}ms", primal_avg_ms * ITERS as f64);
        eprintln!("  (combined dual+primal loop total, {ITERS} iters): {combined_total_ms:.3}ms");
        eprintln!("compute_tile_stats:      {tile_stats_ms:.3}ms");
        eprintln!("compute_tile_bias_scale: {tile_bias_scale_ms:.3}ms  (on-device, round 4 -- was a 6-transfer host round trip)");
        eprintln!("tonemap_tiled:           {tonemap_tiled_ms:.3}ms");
        eprintln!("median_filter_3x3:       {median_ms:.3}ms");
        eprintln!("===");
    }
}
