#![allow(non_camel_case_types)]
use std::ffi::{c_char, c_int, c_void, CStr, CString};

pub mod evt3_raw;
pub use evt3_raw::{Evt3Header, Evt3RawFile};

#[repr(C)]
#[derive(Clone, Copy)]
pub struct MvEventCD { pub t: i64, pub x: u16, pub y: u16, pub p: i16 }

#[repr(C)]
#[derive(Clone, Copy)]
pub struct MvEventTrigger { pub t: i64, pub p: i16, pub id: i16 }

#[derive(Debug, Clone, Copy, Default)]
pub struct FacilityAvailability {
    pub ll_biases: bool,
    pub erc: bool,
    pub antiflicker: bool,
    pub trail_filter: bool,
    pub roi: bool,
    pub digital_crop: bool,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct BiasInfo {
    pub min_recommended: i32,
    pub max_recommended: i32,
    pub min_allowed: i32,
    pub max_allowed: i32,
}

pub type CdCb = unsafe extern "C" fn(*const MvEventCD, usize, *mut c_void);
pub type TrigCb = unsafe extern "C" fn(*const MvEventTrigger, usize, *mut c_void);
pub type StatusCb = unsafe extern "C" fn(c_int, *mut c_void);
pub type RawCb = unsafe extern "C" fn(*const u8, usize, *mut c_void);

extern "C" {
    fn mv_last_error() -> *const c_char;
    fn mv_open_live() -> *mut c_void;
    fn mv_open_file(path: *const c_char, realtime: c_int) -> *mut c_void;
    fn mv_get_geometry(h: *mut c_void, w: *mut c_int, hgt: *mut c_int) -> c_int;
    fn mv_set_cd_callback(h: *mut c_void, cb: CdCb, user: *mut c_void) -> c_int;
    fn mv_set_trigger_callback(h: *mut c_void, cb: TrigCb, user: *mut c_void) -> c_int;
    fn mv_set_raw_callback(h: *mut c_void, cb: RawCb, user: *mut c_void) -> c_int;
    fn mv_set_status_callback(h: *mut c_void, cb: StatusCb, user: *mut c_void) -> c_int;
    fn mv_enable_trigger_in(h: *mut c_void) -> c_int;
    fn mv_set_bias(h: *mut c_void, name: *const c_char, value: c_int) -> c_int;
    fn mv_get_bias(h: *mut c_void, name: *const c_char, value: *mut c_int) -> c_int;
    fn mv_probe_facilities(
        h: *mut c_void,
        has_biases: *mut c_int,
        has_erc: *mut c_int,
        has_antiflicker: *mut c_int,
        has_trail_filter: *mut c_int,
        has_roi: *mut c_int,
        has_digital_crop: *mut c_int,
    ) -> c_int;
    fn mv_get_bias_info(
        h: *mut c_void,
        name: *const c_char,
        min_rec: *mut c_int,
        max_rec: *mut c_int,
        min_alw: *mut c_int,
        max_alw: *mut c_int,
    ) -> c_int;
    fn mv_erc_enable(h: *mut c_void, enable: c_int) -> c_int;
    fn mv_erc_is_enabled(h: *mut c_void, enabled: *mut c_int) -> c_int;
    fn mv_erc_set_cd_event_count(h: *mut c_void, event_count: u32) -> c_int;
    fn mv_erc_get_cd_event_count(h: *mut c_void, event_count: *mut u32) -> c_int;
    fn mv_erc_get_min_max_cd_event_count(h: *mut c_void, min: *mut u32, max: *mut u32) -> c_int;
    fn mv_erc_get_count_period(h: *mut c_void, period_us: *mut u32) -> c_int;
    fn mv_af_enable(h: *mut c_void, enable: c_int) -> c_int;
    fn mv_af_is_enabled(h: *mut c_void, enabled: *mut c_int) -> c_int;
    fn mv_af_set_frequency_band(h: *mut c_void, low_hz: u32, high_hz: u32) -> c_int;
    fn mv_af_get_frequency_band(h: *mut c_void, low_hz: *mut u32, high_hz: *mut u32) -> c_int;
    fn mv_af_get_supported_frequency_range(h: *mut c_void, min_hz: *mut u32, max_hz: *mut u32) -> c_int;
    fn mv_af_set_mode(h: *mut c_void, mode: c_int) -> c_int;
    fn mv_af_get_mode(h: *mut c_void, mode: *mut c_int) -> c_int;
    fn mv_trail_enable(h: *mut c_void, enable: c_int) -> c_int;
    fn mv_trail_is_enabled(h: *mut c_void, enabled: *mut c_int) -> c_int;
    fn mv_trail_set_type(h: *mut c_void, ty: c_int) -> c_int;
    fn mv_trail_get_type(h: *mut c_void, ty: *mut c_int) -> c_int;
    fn mv_trail_get_available_types(h: *mut c_void, type_bitmask: *mut c_int) -> c_int;
    fn mv_trail_set_threshold(h: *mut c_void, threshold_us: u32) -> c_int;
    fn mv_trail_get_threshold(h: *mut c_void, threshold_us: *mut u32) -> c_int;
    fn mv_trail_get_threshold_range(h: *mut c_void, min_us: *mut u32, max_us: *mut u32) -> c_int;
    fn mv_start(h: *mut c_void) -> c_int;
    fn mv_stop(h: *mut c_void) -> c_int;
    fn mv_start_recording(h: *mut c_void, path: *const c_char) -> c_int;
    fn mv_stop_recording(h: *mut c_void) -> c_int;
    fn mv_close(h: *mut c_void);
}

pub struct MvCamera { h: *mut c_void }
unsafe impl Send for MvCamera {}

fn last_error() -> String {
    unsafe { CStr::from_ptr(mv_last_error()).to_string_lossy().into_owned() }
}

fn check(ret: c_int, ctx: &str) -> Result<(), String> {
    if ret != 0 { Ok(()) } else { Err(format!("{ctx}: {}", last_error())) }
}

impl MvCamera {
    pub fn open_live() -> Result<Self, String> {
        let h = unsafe { mv_open_live() };
        if h.is_null() { Err(format!("open_live: {}", last_error())) } else { Ok(Self { h }) }
    }

    pub fn open_file(path: &str, realtime: bool) -> Result<Self, String> {
        let c = CString::new(path).unwrap();
        let h = unsafe { mv_open_file(c.as_ptr(), realtime as c_int) };
        if h.is_null() { Err(format!("open_file: {}", last_error())) } else { Ok(Self { h }) }
    }

    pub fn geometry(&self) -> Result<(u32, u32), String> {
        let (mut w, mut h) = (0, 0);
        check(unsafe { mv_get_geometry(self.h, &mut w, &mut h) }, "geometry")?;
        Ok((w as u32, h as u32))
    }

    pub unsafe fn set_cd_callback(&self, cb: CdCb, user: *mut c_void) -> Result<(), String> {
        check(mv_set_cd_callback(self.h, cb, user), "set_cd_callback")
    }
    pub unsafe fn set_trigger_callback(&self, cb: TrigCb, user: *mut c_void) -> Result<(), String> {
        check(mv_set_trigger_callback(self.h, cb, user), "set_trigger_callback")
    }
    pub unsafe fn set_raw_callback(&self, cb: RawCb, user: *mut c_void) -> Result<(), String> {
        check(mv_set_raw_callback(self.h, cb, user), "set_raw_callback")
    }
    pub unsafe fn set_status_callback(&self, cb: StatusCb, user: *mut c_void) -> Result<(), String> {
        check(mv_set_status_callback(self.h, cb, user), "set_status_callback")
    }

    pub fn enable_trigger_in(&self) -> Result<(), String> {
        check(unsafe { mv_enable_trigger_in(self.h) }, "enable_trigger_in")
    }
    pub fn set_bias(&self, name: &str, v: i32) -> Result<(), String> {
        let c = CString::new(name).unwrap();
        check(unsafe { mv_set_bias(self.h, c.as_ptr(), v) }, "set_bias")
    }
    pub fn get_bias(&self, name: &str) -> Result<i32, String> {
        let c = CString::new(name).unwrap();
        let mut v = 0;
        check(unsafe { mv_get_bias(self.h, c.as_ptr(), &mut v) }, "get_bias")?;
        Ok(v)
    }


    pub fn probe_facilities(&self) -> Result<FacilityAvailability, String> {
        let (mut biases, mut erc, mut af, mut trail, mut roi, mut crop) = (0, 0, 0, 0, 0, 0);
        check(
            unsafe {
                mv_probe_facilities(
                    self.h, &mut biases, &mut erc, &mut af, &mut trail, &mut roi, &mut crop,
                )
            },
            "probe_facilities",
        )?;
        Ok(FacilityAvailability {
            ll_biases: biases != 0,
            erc: erc != 0,
            antiflicker: af != 0,
            trail_filter: trail != 0,
            roi: roi != 0,
            digital_crop: crop != 0,
        })
    }


    pub fn bias_info(&self, name: &str) -> Result<BiasInfo, String> {
        let c = CString::new(name).unwrap();
        let (mut min_rec, mut max_rec, mut min_alw, mut max_alw) = (0, 0, 0, 0);
        check(
            unsafe {
                mv_get_bias_info(self.h, c.as_ptr(), &mut min_rec, &mut max_rec, &mut min_alw, &mut max_alw)
            },
            "bias_info",
        )?;
        Ok(BiasInfo { min_recommended: min_rec, max_recommended: max_rec, min_allowed: min_alw, max_allowed: max_alw })
    }


    pub fn erc_enable(&self, on: bool) -> Result<(), String> {
        check(unsafe { mv_erc_enable(self.h, on as c_int) }, "erc_enable")
    }
    pub fn erc_is_enabled(&self) -> Result<bool, String> {
        let mut v = 0;
        check(unsafe { mv_erc_is_enabled(self.h, &mut v) }, "erc_is_enabled")?;
        Ok(v != 0)
    }
    pub fn erc_set_cd_event_count(&self, count: u32) -> Result<(), String> {
        check(unsafe { mv_erc_set_cd_event_count(self.h, count) }, "erc_set_cd_event_count")
    }
    pub fn erc_cd_event_count(&self) -> Result<u32, String> {
        let mut v = 0;
        check(unsafe { mv_erc_get_cd_event_count(self.h, &mut v) }, "erc_cd_event_count")?;
        Ok(v)
    }
    pub fn erc_cd_event_count_range(&self) -> Result<(u32, u32), String> {
        let (mut lo, mut hi) = (0, 0);
        check(unsafe { mv_erc_get_min_max_cd_event_count(self.h, &mut lo, &mut hi) }, "erc_cd_event_count_range")?;
        Ok((lo, hi))
    }
    pub fn erc_count_period_us(&self) -> Result<u32, String> {
        let mut v = 0;
        check(unsafe { mv_erc_get_count_period(self.h, &mut v) }, "erc_count_period_us")?;
        Ok(v)
    }

    pub fn af_enable(&self, on: bool) -> Result<(), String> {
        check(unsafe { mv_af_enable(self.h, on as c_int) }, "af_enable")
    }
    pub fn af_is_enabled(&self) -> Result<bool, String> {
        let mut v = 0;
        check(unsafe { mv_af_is_enabled(self.h, &mut v) }, "af_is_enabled")?;
        Ok(v != 0)
    }
    pub fn af_set_frequency_band(&self, low_hz: u32, high_hz: u32) -> Result<(), String> {
        check(unsafe { mv_af_set_frequency_band(self.h, low_hz, high_hz) }, "af_set_frequency_band")
    }
    pub fn af_frequency_band(&self) -> Result<(u32, u32), String> {
        let (mut lo, mut hi) = (0, 0);
        check(unsafe { mv_af_get_frequency_band(self.h, &mut lo, &mut hi) }, "af_frequency_band")?;
        Ok((lo, hi))
    }
    pub fn af_supported_frequency_range(&self) -> Result<(u32, u32), String> {
        let (mut lo, mut hi) = (0, 0);
        check(unsafe { mv_af_get_supported_frequency_range(self.h, &mut lo, &mut hi) }, "af_supported_frequency_range")?;
        Ok((lo, hi))
    }

    pub fn af_set_mode(&self, mode: i32) -> Result<(), String> {
        check(unsafe { mv_af_set_mode(self.h, mode) }, "af_set_mode")
    }
    pub fn af_mode(&self) -> Result<i32, String> {
        let mut v = 0;
        check(unsafe { mv_af_get_mode(self.h, &mut v) }, "af_mode")?;
        Ok(v)
    }

    pub fn trail_enable(&self, on: bool) -> Result<(), String> {
        check(unsafe { mv_trail_enable(self.h, on as c_int) }, "trail_enable")
    }
    pub fn trail_is_enabled(&self) -> Result<bool, String> {
        let mut v = 0;
        check(unsafe { mv_trail_is_enabled(self.h, &mut v) }, "trail_is_enabled")?;
        Ok(v != 0)
    }

    pub fn trail_set_type(&self, ty: i32) -> Result<(), String> {
        check(unsafe { mv_trail_set_type(self.h, ty) }, "trail_set_type")
    }
    pub fn trail_type(&self) -> Result<i32, String> {
        let mut v = 0;
        check(unsafe { mv_trail_get_type(self.h, &mut v) }, "trail_type")?;
        Ok(v)
    }

    pub fn trail_available_types(&self) -> Result<u32, String> {
        let mut v: c_int = 0;
        check(unsafe { mv_trail_get_available_types(self.h, &mut v) }, "trail_available_types")?;
        Ok(v as u32)
    }
    pub fn trail_set_threshold_us(&self, us: u32) -> Result<(), String> {
        check(unsafe { mv_trail_set_threshold(self.h, us) }, "trail_set_threshold_us")
    }
    pub fn trail_threshold_us(&self) -> Result<u32, String> {
        let mut v = 0;
        check(unsafe { mv_trail_get_threshold(self.h, &mut v) }, "trail_threshold_us")?;
        Ok(v)
    }
    pub fn trail_threshold_range_us(&self) -> Result<(u32, u32), String> {
        let (mut lo, mut hi) = (0, 0);
        check(unsafe { mv_trail_get_threshold_range(self.h, &mut lo, &mut hi) }, "trail_threshold_range_us")?;
        Ok((lo, hi))
    }

    pub fn start(&self) -> Result<(), String> { check(unsafe { mv_start(self.h) }, "start") }
    pub fn stop(&self) -> Result<(), String> { check(unsafe { mv_stop(self.h) }, "stop") }
    pub fn start_recording(&self, path: &str) -> Result<(), String> {
        let c = CString::new(path).unwrap();
        check(unsafe { mv_start_recording(self.h, c.as_ptr()) }, "start_recording")
    }
    pub fn stop_recording(&self) -> Result<(), String> {
        check(unsafe { mv_stop_recording(self.h) }, "stop_recording")
    }
}

impl Drop for MvCamera {
    fn drop(&mut self) { unsafe { mv_close(self.h) } }
}
