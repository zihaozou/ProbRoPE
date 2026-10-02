use serde::Serialize;


#[derive(Clone)]
pub struct RgbFrame {
    pub seq: u64,
    pub t_cam_us: i64,
    pub w: u32,
    pub h: u32,
    pub data: Vec<u8>,
    pub format: PixelFormat,
}


#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PixelFormat { Bayer8, Bgr8, Gray8, Rgb8 }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Event {
    pub t_us: i64,
    pub x: u16,
    pub y: u16,

    pub p: i8,
}

#[derive(Clone, Default)]
pub struct EventBatch { pub events: Vec<Event> }

#[derive(Clone, Copy, Debug)]
pub struct TriggerEvent {
    pub t_us: i64,

    pub polarity: i8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum SyncSource { Matched, Interpolated }

#[derive(Clone)]
pub struct SyncedFrame {
    pub frame: RgbFrame,

    pub t_evk_us: i64,
    pub source: SyncSource,
}

impl SyncedFrame {

    pub fn recon_time_us(&self, exposure_us: i64) -> i64 { self.t_evk_us + exposure_us / 2 }
}


#[derive(Clone)]
pub struct GrayImage { pub w: u32, pub h: u32, pub data: Vec<u8> }

impl GrayImage {
    pub fn new(w: u32, h: u32) -> Self { Self { w, h, data: vec![0; (w * h) as usize] } }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn recon_time_is_exposure_midpoint() {
        let f = SyncedFrame {
            frame: RgbFrame { seq: 0, t_cam_us: 0, w: 1, h: 1, data: vec![0], format: PixelFormat::Gray8 },
            t_evk_us: 1000, source: SyncSource::Matched,
        };
        assert_eq!(f.recon_time_us(4000), 3000);
    }
}
