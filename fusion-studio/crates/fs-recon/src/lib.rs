pub mod accumulator;
pub use accumulator::Accumulator;
pub mod cuda;
pub use cuda::{CudaManifold, Normalization};
pub mod gpu_decode;
pub use gpu_decode::{CarryState, CpuTriggerScanner, DecodedEvent, DecodedEventsDevice, DecodedTrigger, GpuEventDecoder};
pub use cudarc::driver::CudaStream;

use std::any::Any;

use fs_core::{Event, EventBatch, GrayImage};


pub trait Reconstructor: Send {
    fn name(&self) -> &'static str;
    fn push_events(&mut self, batch: &EventBatch);
    fn render_at(&mut self, t_us: i64, out: &mut GrayImage);
    fn reset(&mut self);
    fn push_events_device(&mut self, dev: &DecodedEventsDevice, stream: &std::sync::Arc<cudarc::driver::CudaStream>) {
        let events: Vec<Event> = match dev.download(stream) {
            Ok(decoded) => decoded
                .into_iter()
                .map(|e| Event { t_us: e.t_us, x: e.x, y: e.y, p: e.p as i8 })
                .collect(),
            Err(err) => {
                debug_assert!(false, "push_events_device: default download fallback failed: {err}");
                return;
            }
        };
        self.push_events(&EventBatch { events });
    }
    fn params_ui(&mut self, _ui: &mut egui::Ui) {}
    fn as_any_mut(&mut self) -> &mut dyn Any;
}
