use crate::*;
use std::ffi::CString;

pub type Result<T> = std::result::Result<T, String>;

fn ck(err: spinError, ctx: &str) -> Result<()> {
    if err == _spinError_SPINNAKER_ERR_SUCCESS {
        Ok(())
    } else {
        Err(format!("{ctx}: spinError {err}"))
    }
}

pub struct SpinSystem {
    h: spinSystem,
}
unsafe impl Send for SpinSystem {}

impl SpinSystem {
    pub fn new() -> Result<Self> {
        let mut h = std::ptr::null_mut();
        ck(unsafe { spinSystemGetInstance(&mut h) }, "SystemGetInstance")?;
        Ok(Self { h })
    }

    pub fn camera_count(&self) -> Result<usize> {
        unsafe {
            let mut list = std::ptr::null_mut();
            ck(spinCameraListCreateEmpty(&mut list), "CameraListCreateEmpty")?;
            ck(spinSystemGetCameras(self.h, list), "SystemGetCameras")?;
            let mut n: usize = 0;
            ck(spinCameraListGetSize(list, &mut n), "CameraListGetSize")?;
            spinCameraListClear(list);
            spinCameraListDestroy(list);
            Ok(n)
        }
    }

    pub fn camera(&self, index: usize) -> Result<SpinCamera> {
        unsafe {
            let mut list = std::ptr::null_mut();
            ck(spinCameraListCreateEmpty(&mut list), "CameraListCreateEmpty")?;
            ck(spinSystemGetCameras(self.h, list), "SystemGetCameras")?;
            let mut cam = std::ptr::null_mut();
            let r = ck(spinCameraListGet(list, index, &mut cam), "CameraListGet");
            spinCameraListClear(list);
            spinCameraListDestroy(list);
            r?;
            ck(spinCameraInit(cam), "CameraInit")?;
            let mut nodemap = std::ptr::null_mut();
            ck(spinCameraGetNodeMap(cam, &mut nodemap), "GetNodeMap")?;
            Ok(SpinCamera { h: cam, nodemap })
        }
    }
}

impl Drop for SpinSystem {
    fn drop(&mut self) {
        unsafe {
            spinSystemReleaseInstance(self.h);
        }
    }
}

pub struct SpinCamera {
    h: spinCamera,
    nodemap: spinNodeMapHandle,
}
unsafe impl Send for SpinCamera {}

impl SpinCamera {
    fn node(&self, name: &str) -> Result<spinNodeHandle> {
        let c = CString::new(name).unwrap();
        let mut node = std::ptr::null_mut();
        ck(
            unsafe { spinNodeMapGetNode(self.nodemap, c.as_ptr(), &mut node) },
            name,
        )?;
        Ok(node)
    }

    pub fn set_enum(&self, name: &str, entry: &str) -> Result<()> {
        unsafe {
            let node = self.node(name)?;
            let ce = CString::new(entry).unwrap();
            let mut eh = std::ptr::null_mut();
            ck(
                spinEnumerationGetEntryByName(node, ce.as_ptr(), &mut eh),
                "GetEntryByName",
            )?;
            let mut v: i64 = 0;
            ck(spinEnumerationEntryGetIntValue(eh, &mut v), "EntryGetIntValue")?;
            ck(spinEnumerationSetIntValue(node, v), name)
        }
    }
    pub fn set_float(&self, name: &str, v: f64) -> Result<()> {
        unsafe { ck(spinFloatSetValue(self.node(name)?, v), name) }
    }
    pub fn get_float(&self, name: &str) -> Result<f64> {
        let mut v = 0.0;
        ck(unsafe { spinFloatGetValue(self.node(name)?, &mut v) }, name)?;
        Ok(v)
    }
    pub fn set_bool(&self, name: &str, v: bool) -> Result<()> {
        unsafe { ck(spinBooleanSetValue(self.node(name)?, v as bool8_t), name) }
    }
    pub fn get_bool(&self, name: &str) -> Result<bool> {
        let mut v: bool8_t = 0;
        ck(unsafe { spinBooleanGetValue(self.node(name)?, &mut v) }, name)?;
        Ok(v != 0)
    }

    pub fn float_range(&self, name: &str) -> Result<(f64, f64)> {
        let node = self.node(name)?;
        let (mut lo, mut hi) = (0.0, 0.0);
        ck(unsafe { spinFloatGetMin(node, &mut lo) }, name)?;
        ck(unsafe { spinFloatGetMax(node, &mut hi) }, name)?;
        Ok((lo, hi))
    }

    pub fn get_enum(&self, name: &str) -> Result<String> {
        unsafe {
            let node = self.node(name)?;
            let mut eh = std::ptr::null_mut();
            ck(spinEnumerationGetCurrentEntry(node, &mut eh), name)?;
            let mut buf = [0u8; 256];
            let mut len = buf.len();
            ck(
                spinEnumerationEntryGetSymbolic(eh, buf.as_mut_ptr() as *mut _, &mut len),
                "EntryGetSymbolic",
            )?;
            let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
            Ok(String::from_utf8_lossy(&buf[..end]).into_owned())
        }
    }
    pub fn execute(&self, name: &str) -> Result<()> {
        unsafe { ck(spinCommandExecute(self.node(name)?), name) }
    }

    pub fn begin_acquisition(&self) -> Result<()> {
        ck(unsafe { spinCameraBeginAcquisition(self.h) }, "BeginAcquisition")
    }
    pub fn end_acquisition(&self) -> Result<()> {
        ck(unsafe { spinCameraEndAcquisition(self.h) }, "EndAcquisition")
    }

    pub fn next_image(&self, timeout_ms: u64) -> Result<CapturedImage> {
        unsafe {
            let mut img = std::ptr::null_mut();
            ck(
                spinCameraGetNextImageEx(self.h, timeout_ms, &mut img),
                "GetNextImageEx",
            )?;
            let result = (|| {
                let mut incomplete: bool8_t = 0;
                spinImageIsIncomplete(img, &mut incomplete);
                if incomplete != 0 {
                    return Err("incomplete image".into());
                }
                let mut w: usize = 0;
                let mut h: usize = 0;
                let mut ts: u64 = 0;
                ck(spinImageGetWidth(img, &mut w), "GetWidth")?;
                ck(spinImageGetHeight(img, &mut h), "GetHeight")?;
                ck(spinImageGetTimeStamp(img, &mut ts), "GetTimeStamp")?;
                let mut buffer_size: usize = 0;
                ck(spinImageGetBufferSize(img, &mut buffer_size), "GetBufferSize")?;
                let mut payload_size: usize = 0;
                ck(
                    spinImageGetValidPayloadSize(img, &mut payload_size),
                    "GetValidPayloadSize",
                )?;
                let size = payload_size.min(buffer_size);
                let mut ptr = std::ptr::null_mut();
                ck(spinImageGetData(img, &mut ptr), "GetData")?;
                let data = std::slice::from_raw_parts(ptr as *const u8, size).to_vec();
                Ok(CapturedImage {
                    w: w as u32,
                    h: h as u32,
                    t_ns: ts,
                    data,
                })
            })();
            spinImageRelease(img);
            result
        }
    }
}

impl Drop for SpinCamera {
    fn drop(&mut self) {
        unsafe {
            spinCameraDeInit(self.h);
            spinCameraRelease(self.h);
        }
    }
}

pub struct CapturedImage {
    pub w: u32,
    pub h: u32,
    pub t_ns: u64,
    pub data: Vec<u8>,
}
