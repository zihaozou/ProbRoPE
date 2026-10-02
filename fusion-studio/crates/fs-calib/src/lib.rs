
pub mod active;
pub mod background;
pub mod board;
pub mod engine;
pub mod export;
pub mod geom;
pub mod pool;
pub mod sim;

pub use active::{ActiveCalibration, CalibSource};
pub use background::BackgroundSession;
pub use board::{BoardConfig, BoardKind, Detection};
pub use engine::{CalibSession, CamCalib, IntrinsicsEngine, StereoCalib, StereoEngine};
pub use export::{to_native_json, to_stereo_calibration_v1};
pub use geom::{derive, post_op_document, CamMaps, GeomMode};
pub use pool::{KeyframePool, Offer, PoolEntry};
pub use sim::SimRig;
