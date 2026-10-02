use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use crossbeam_channel::Sender;

use crate::recon_thread::{self, ParamsMsg, ReconChoice, ReconHandles, ReconStats, ReconSwap};

pub struct ReconPanelState {
    pub choice: ReconChoice,
    pub error: Option<String>,
    tx_swap: Sender<ReconSwap>,
    tx_params: Sender<ParamsMsg>,
    paired_only: Arc<AtomicBool>,

    auto_degrade: Arc<AtomicBool>,

    stats: Arc<Mutex<ReconStats>>,
    recon_dims: (u32, u32),
    iters: u32,
    tau_leak_us: f32,
    median_filter: bool,
    window_us: i64,
}

impl ReconPanelState {

    pub fn new(handles: ReconHandles, recon_dims: (u32, u32)) -> Self {
        ReconPanelState {
            choice: handles.initial_choice,
            error: handles.initial_error,
            tx_swap: handles.tx_swap,
            tx_params: handles.tx_params,
            paired_only: handles.paired_only,
            auto_degrade: handles.auto_degrade,
            stats: handles.stats,
            recon_dims,
            iters: 2,
            tau_leak_us: 2_000_000.0,
            median_filter: true,
            window_us: 150_000,
        }
    }

    pub fn set_recon_dims(&mut self, dims: (u32, u32)) {
        self.recon_dims = dims;
    }
}


pub fn recon_panel(ui: &mut egui::Ui, state: &mut ReconPanelState) {
    let prev_choice = state.choice;
    egui::ComboBox::from_id_salt("recon_choice")
        .selected_text(match state.choice {
            ReconChoice::CudaManifold => "cuda-manifold (recommended)",
            ReconChoice::Accumulator => "accumulator",
        })
        .show_ui(ui, |ui| {
            ui.selectable_value(&mut state.choice, ReconChoice::CudaManifold, "cuda-manifold (recommended)");
            ui.selectable_value(&mut state.choice, ReconChoice::Accumulator, "accumulator");
        });

    if state.choice != prev_choice {
        let (w, h) = state.recon_dims;
        match state.choice {
            ReconChoice::CudaManifold => match recon_thread::build_cuda_manifold(w, h) {
                Ok(m) => {
                    state.error = None;
                    let _ = state.tx_swap.send(ReconSwap { recon: Box::new(m), dims: (w, h) });
                }
                Err(e) => {
                    state.error = Some(e);
                    state.choice = ReconChoice::Accumulator;
                    let _ = state
                        .tx_swap
                        .send(ReconSwap { recon: Box::new(recon_thread::build_accumulator(w, h)), dims: (w, h) });
                }
            },
            ReconChoice::Accumulator => {
                state.error = None;
                let _ = state
                    .tx_swap
                    .send(ReconSwap { recon: Box::new(recon_thread::build_accumulator(w, h)), dims: (w, h) });
            }
        }
    }

    if let Some(err) = &state.error {
        ui.colored_label(egui::Color32::RED, format!("cuda-manifold init failed: {err}"));
    }

    ui.add_space(4.0);
    match state.choice {
        ReconChoice::CudaManifold => {
            if ui.add(egui::Slider::new(&mut state.iters, 0..=20).text("PD iters")).changed() {
                let _ = state.tx_params.send(ParamsMsg::SetIters(state.iters));
            }
            if ui
                .add(egui::Slider::new(&mut state.tau_leak_us, 100_000.0..=5_000_000.0).logarithmic(true).text("leak tau (us)"))
                .changed()
            {
                let _ = state.tx_params.send(ParamsMsg::SetTauLeak(state.tau_leak_us));
            }
            if ui.checkbox(&mut state.median_filter, "3x3 median filter").changed() {
                let _ = state.tx_params.send(ParamsMsg::SetMedian(state.median_filter));
            }
        }
        ReconChoice::Accumulator => {
            if ui
                .add(egui::Slider::new(&mut state.window_us, 1_000..=500_000).logarithmic(true).text("window µs"))
                .changed()
            {
                let _ = state.tx_params.send(ParamsMsg::SetWindowUs(state.window_us));
            }
        }
    }

    ui.add_space(4.0);
    let mut paired_only = state.paired_only.load(Ordering::Relaxed);
    if ui.checkbox(&mut paired_only, "仅显示配对帧 paired-only").changed() {
        state.paired_only.store(paired_only, Ordering::Relaxed);
    }

    ui.add_space(4.0);
    let stats = *state.stats.lock().unwrap();
    let mut auto_degrade = state.auto_degrade.load(Ordering::Relaxed);
    if ui.checkbox(&mut auto_degrade, "过载自动降级 auto-degrade on overload").changed() {
        state.auto_degrade.store(auto_degrade, Ordering::Relaxed);
    }
    if stats.render_overloaded {
        ui.colored_label(
            egui::Color32::RED,
            format!(
                "渲染过慢 RENDER TOO SLOW: {:.0}ms avg render (over budget). First try: enable the EVK4's \
                 Event Trail Filter / ERC in Device settings (sensor-side rate limiting -- the correct \
                 fix for a genuinely overloaded event rate). Software fallback: lower PD iters, disable \
                 median filter, or lower fps -- auto-degrade above already does the first two for you.",
                stats.render_ms_avg
            ),
        );
    }
    if stats.coverage_overloaded {
        ui.colored_label(
            egui::Color32::RED,
            format!(
                "配对请求不足 INSUFFICIENT PAIRED REQUESTS: only {:.0}% coverage -- the recon thread isn't \
                 being asked to render often enough (upstream frame drops or sync issues, not render \
                 speed). Auto-degrade does NOT act on this -- lowering PD iters/median can't fix a \
                 request-starvation problem. Check sync/trigger health (同步 panel above) instead.",
                stats.paired_coverage_pct
            ),
        );
    }
    if state.choice == ReconChoice::CudaManifold {
        ui.label(format!(
            "effective: iters={} median={} (render avg {:.1}ms, coverage {:.0}%, dropped {})",
            stats.effective_iters, stats.median_active, stats.render_ms_avg, stats.paired_coverage_pct, stats.events_dropped
        ));
    }
}
