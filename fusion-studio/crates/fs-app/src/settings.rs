
use std::sync::{Arc, Mutex};

use serde::Serialize;

pub const FLIR_EXPOSURE_RANGE_US: (f64, f64) = (11.0, 30_000_005.0);
pub const FLIR_FPS_RANGE: (f64, f64) = (1.0, 84.499_596_767_924_22);
pub const FLIR_GAIN_RANGE_DB: (f64, f64) = (0.0, 18.062_167_207_256_408);

pub const CONNECT_REVISION: u64 = 1;


#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Serialize)]
pub enum AutoMode {
    #[default]
    Off,
    Once,
    Continuous,
}

impl AutoMode {
    pub const ALL: [AutoMode; 3] = [AutoMode::Off, AutoMode::Once, AutoMode::Continuous];


    pub fn node_entry(self) -> &'static str {
        match self {
            AutoMode::Off => "Off",
            AutoMode::Once => "Once",
            AutoMode::Continuous => "Continuous",
        }
    }


    pub fn from_node_entry(s: &str) -> Option<AutoMode> {
        match s {
            "Off" => Some(AutoMode::Off),
            "Once" => Some(AutoMode::Once),
            "Continuous" => Some(AutoMode::Continuous),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub enum FpsApplyState {
    #[default]
    Idle,
    InFlight,
    Failed(String),
}

pub fn fps_apply_enabled(populated: bool, state: &FpsApplyState, edit: f64, range: (f64, f64)) -> bool {
    populated && *state != FpsApplyState::InFlight && edit.is_finite() && edit >= range.0 && edit <= range.1
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct FlirSettingsSnapshot {
    pub populated: bool,
    pub exposure_us: f64,
    pub exposure_range_us: (f64, f64),
    pub exposure_auto: AutoMode,
    pub gain_db: f64,
    pub gain_range_db: (f64, f64),
    pub gain_auto: AutoMode,

    pub fps: f64,
    pub fps_range: (f64, f64),
    pub isp_enable: bool,
    pub temperature_c: Option<f64>,
    pub fps_apply: FpsApplyState,
    pub last_error: Option<String>,
    pub revision: u64,
}


#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Serialize)]
pub enum AntiflickerMode {

    BandPass,
    #[default]
    BandStop,
}

impl AntiflickerMode {
    pub const ALL: [AntiflickerMode; 2] = [AntiflickerMode::BandPass, AntiflickerMode::BandStop];

    pub fn to_shim(self) -> i32 {
        match self {
            AntiflickerMode::BandPass => 0,
            AntiflickerMode::BandStop => 1,
        }
    }
    pub fn from_shim(v: i32) -> Option<AntiflickerMode> {
        match v {
            0 => Some(AntiflickerMode::BandPass),
            1 => Some(AntiflickerMode::BandStop),
            _ => None,
        }
    }
    pub fn label(self) -> &'static str {
        match self {
            AntiflickerMode::BandPass => "band-pass (keep band)",
            AntiflickerMode::BandStop => "band-stop (remove band)",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Serialize)]
pub enum TrailFilterType {
    #[default]
    Trail,
    StcCutTrail,
    StcKeepTrail,
}

impl TrailFilterType {
    pub const ALL: [TrailFilterType; 3] =
        [TrailFilterType::Trail, TrailFilterType::StcCutTrail, TrailFilterType::StcKeepTrail];

    pub fn to_shim(self) -> i32 {
        match self {
            TrailFilterType::Trail => 0,
            TrailFilterType::StcCutTrail => 1,
            TrailFilterType::StcKeepTrail => 2,
        }
    }
    pub fn from_shim(v: i32) -> Option<TrailFilterType> {
        match v {
            0 => Some(TrailFilterType::Trail),
            1 => Some(TrailFilterType::StcCutTrail),
            2 => Some(TrailFilterType::StcKeepTrail),
            _ => None,
        }
    }

    pub fn bit(self) -> u32 {
        1 << self.to_shim() as u32
    }
    pub fn label(self) -> &'static str {
        match self {
            TrailFilterType::Trail => "TRAIL (keep first)",
            TrailFilterType::StcCutTrail => "STC cut trail",
            TrailFilterType::StcKeepTrail => "STC keep trail",
        }
    }
}

pub const EVK_BIAS_NAMES: [&str; 5] = ["bias_diff_on", "bias_diff_off", "bias_fo", "bias_hpf", "bias_refr"];

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct BiasSnapshot {

    pub available: bool,
    pub value: i32,

    pub recommended: (i32, i32),

    pub allowed: (i32, i32),
}


#[derive(Clone, Debug, Default, Serialize)]
pub struct EvkSettingsSnapshot {
    pub populated: bool,
    pub biases_available: bool,

    pub biases: [BiasSnapshot; 5],
    pub erc_available: bool,
    pub erc_enabled: bool,
    pub erc_event_count: u32,
    pub erc_count_range: (u32, u32),

    pub erc_period_us: u32,
    pub af_available: bool,
    pub af_enabled: bool,
    pub af_band_hz: (u32, u32),
    pub af_freq_range_hz: (u32, u32),
    pub af_mode: AntiflickerMode,
    pub trail_available: bool,
    pub trail_enabled: bool,
    pub trail_type: TrailFilterType,

    pub trail_available_types: u32,
    pub trail_threshold_us: u32,
    pub trail_threshold_range_us: (u32, u32),
    pub last_error: Option<String>,

    pub revision: u64,
}

#[derive(Clone)]
pub struct SharedTuning {
    fps: Arc<Mutex<f64>>,
    exposure_us: Arc<Mutex<i64>>,
}

impl SharedTuning {
    pub fn new(fps: f64, exposure_us: i64) -> Self {
        Self { fps: Arc::new(Mutex::new(fps)), exposure_us: Arc::new(Mutex::new(exposure_us)) }
    }
    pub fn fps(&self) -> f64 {
        *self.fps.lock().unwrap()
    }
    pub fn set_fps(&self, v: f64) {
        *self.fps.lock().unwrap() = v;
    }
    pub fn exposure_us(&self) -> i64 {
        *self.exposure_us.lock().unwrap()
    }
    pub fn set_exposure_us(&self, v: i64) {
        *self.exposure_us.lock().unwrap() = v;
    }
}


pub fn clamp_f64(v: f64, range: (f64, f64)) -> f64 {
    if v.is_nan() {
        return range.0;
    }
    v.clamp(range.0, range.1)
}

pub fn clamp_i32(v: i32, range: (i32, i32)) -> i32 {
    v.clamp(range.0, range.1)
}

pub fn clamp_u32(v: u32, range: (u32, u32)) -> u32 {
    v.clamp(range.0, range.1)
}

pub fn sanitize_af_band(low_hz: u32, high_hz: u32, supported: (u32, u32)) -> (u32, u32) {
    let (lo, hi) = if low_hz <= high_hz { (low_hz, high_hz) } else { (high_hz, low_hz) };
    (clamp_u32(lo, supported), clamp_u32(hi, supported))
}


pub fn should_copy_snapshot(populated: bool, snapshot_revision: u64, seeded_revision: u64) -> bool {
    populated && snapshot_revision != seeded_revision
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_mode_node_entry_round_trip() {
        for m in AutoMode::ALL {
            assert_eq!(AutoMode::from_node_entry(m.node_entry()), Some(m));
        }
        assert_eq!(AutoMode::from_node_entry("Banana"), None);
        assert_eq!(AutoMode::from_node_entry(""), None);
    }

    #[test]
    fn antiflicker_mode_shim_round_trip() {
        for m in AntiflickerMode::ALL {
            assert_eq!(AntiflickerMode::from_shim(m.to_shim()), Some(m));
        }
        assert_eq!(AntiflickerMode::BandPass.to_shim(), 0);
        assert_eq!(AntiflickerMode::BandStop.to_shim(), 1);
        assert_eq!(AntiflickerMode::from_shim(2), None);
        assert_eq!(AntiflickerMode::from_shim(-1), None);
    }

    #[test]
    fn trail_type_shim_round_trip_and_bits() {
        for t in TrailFilterType::ALL {
            assert_eq!(TrailFilterType::from_shim(t.to_shim()), Some(t));
        }
        assert_eq!(TrailFilterType::Trail.to_shim(), 0);
        assert_eq!(TrailFilterType::StcCutTrail.to_shim(), 1);
        assert_eq!(TrailFilterType::StcKeepTrail.to_shim(), 2);
        assert_eq!(TrailFilterType::Trail.bit(), 0b001);
        assert_eq!(TrailFilterType::StcCutTrail.bit(), 0b010);
        assert_eq!(TrailFilterType::StcKeepTrail.bit(), 0b100);
        assert_eq!(TrailFilterType::from_shim(3), None);
    }

    #[test]
    fn clamp_f64_handles_nan_and_bounds() {
        assert_eq!(clamp_f64(5.0, (1.0, 10.0)), 5.0);
        assert_eq!(clamp_f64(-3.0, (1.0, 10.0)), 1.0);
        assert_eq!(clamp_f64(99.0, (1.0, 10.0)), 10.0);
        assert_eq!(clamp_f64(f64::NAN, (1.0, 10.0)), 1.0);
        assert_eq!(clamp_f64(f64::INFINITY, (1.0, 10.0)), 10.0);
        assert_eq!(clamp_f64(f64::NEG_INFINITY, (1.0, 10.0)), 1.0);
    }

    #[test]
    fn clamp_ints() {
        assert_eq!(clamp_i32(-200, (-150, 200)), -150);
        assert_eq!(clamp_i32(250, (-150, 200)), 200);
        assert_eq!(clamp_i32(0, (-150, 200)), 0);
        assert_eq!(clamp_u32(5, (10, 20)), 10);
        assert_eq!(clamp_u32(25, (10, 20)), 20);
    }

    #[test]
    fn af_band_sanitizer_orders_and_clamps() {
        let sup = (50, 520);
        assert_eq!(sanitize_af_band(100, 120, sup), (100, 120));
        assert_eq!(sanitize_af_band(120, 100, sup), (100, 120));
        assert_eq!(sanitize_af_band(10, 9999, sup), (50, 520));
        assert_eq!(sanitize_af_band(700, 600, sup), (520, 520));
        assert_eq!(sanitize_af_band(0, 0, sup), (50, 50));
    }

    #[test]
    fn fps_apply_button_gating() {
        let range = (1.0, 84.5);
        assert!(fps_apply_enabled(true, &FpsApplyState::Idle, 30.0, range));
        assert!(fps_apply_enabled(true, &FpsApplyState::Failed("x".into()), 30.0, range));
        assert!(!fps_apply_enabled(true, &FpsApplyState::InFlight, 30.0, range));
        assert!(!fps_apply_enabled(false, &FpsApplyState::Idle, 30.0, range));
        assert!(!fps_apply_enabled(true, &FpsApplyState::Idle, 0.5, range));
        assert!(!fps_apply_enabled(true, &FpsApplyState::Idle, 85.0, range));
        assert!(!fps_apply_enabled(true, &FpsApplyState::Idle, f64::NAN, range));
        assert!(fps_apply_enabled(true, &FpsApplyState::Idle, 1.0, range));
        assert!(fps_apply_enabled(true, &FpsApplyState::Idle, 84.5, range));
    }

    #[test]
    fn snapshot_copy_gating() {
        assert!(!should_copy_snapshot(false, 1, 0), "never before populated");
        assert!(should_copy_snapshot(true, 1, 0), "new revision -> copy");
        assert!(!should_copy_snapshot(true, 3, 3), "already consumed");
    }

    #[test]
    fn connect_time_populate_triggers_panel_copy() {
        let fresh_panel_seeded_revision = 0;
        assert!(should_copy_snapshot(true, CONNECT_REVISION, fresh_panel_seeded_revision));
        assert_ne!(CONNECT_REVISION, 0);
        assert!(!should_copy_snapshot(true, 0, fresh_panel_seeded_revision));
    }


    #[test]
    fn flir_snapshot_round_trips_with_readable_fields() {
        let snap = FlirSettingsSnapshot {
            populated: true,
            exposure_us: 5000.0,
            exposure_range_us: FLIR_EXPOSURE_RANGE_US,
            exposure_auto: AutoMode::Once,
            gain_db: 6.5,
            gain_range_db: FLIR_GAIN_RANGE_DB,
            gain_auto: AutoMode::Off,
            fps: 30.0,
            fps_range: FLIR_FPS_RANGE,
            isp_enable: true,
            temperature_c: Some(42.5),
            fps_apply: FpsApplyState::Idle,
            last_error: None,
            revision: 3,
        };
        let v = serde_json::to_value(&snap).unwrap();
        assert_eq!(v["exposure_us"], 5000.0);
        assert_eq!(v["gain_db"], 6.5);
        assert_eq!(v["fps"], 30.0);
        assert_eq!(v["exposure_auto"], "Once");
        assert_eq!(v["temperature_c"], 42.5);
    }

    #[test]
    fn evk_snapshot_round_trips_with_readable_bias_fields() {
        let mut biases = [BiasSnapshot::default(); 5];
        biases[0] = BiasSnapshot { available: true, value: -35, recommended: (-50, 50), allowed: (-200, 200) };
        let snap = EvkSettingsSnapshot {
            populated: true,
            biases_available: true,
            biases,
            erc_available: true,
            erc_enabled: true,
            erc_event_count: 20_000_000,
            erc_count_range: (0, 100_000_000),
            erc_period_us: 1000,
            af_available: true,
            af_enabled: false,
            af_band_hz: (90, 110),
            af_freq_range_hz: (50, 520),
            af_mode: AntiflickerMode::BandStop,
            trail_available: true,
            trail_enabled: true,
            trail_type: TrailFilterType::StcCutTrail,
            trail_available_types: 0b111,
            trail_threshold_us: 10_000,
            trail_threshold_range_us: (0, 1_000_000),
            last_error: None,
            revision: 1,
        };
        let v = serde_json::to_value(&snap).unwrap();
        assert_eq!(v["biases"][0]["value"], -35);
        assert_eq!(v["biases"][0]["available"], true);
        assert_eq!(v["erc_event_count"], 20_000_000);
        assert_eq!(v["af_mode"], "BandStop");
        assert_eq!(v["trail_type"], "StcCutTrail");
    }


    #[test]
    fn fps_apply_state_failed_serializes_as_readable_tagged_message() {
        assert_eq!(serde_json::to_value(FpsApplyState::Idle).unwrap(), serde_json::json!("Idle"));
        assert_eq!(serde_json::to_value(FpsApplyState::InFlight).unwrap(), serde_json::json!("InFlight"));
        let failed = FpsApplyState::Failed("node write refused".into());
        assert_eq!(serde_json::to_value(failed).unwrap(), serde_json::json!({"Failed": "node write refused"}));
    }
}
