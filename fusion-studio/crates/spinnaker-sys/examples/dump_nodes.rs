
#![allow(non_upper_case_globals)]

use spinnaker_sys::*;
use std::ffi::CStr;
use std::os::raw::c_char;

fn ck(err: spinError, ctx: &str) -> Result<(), String> {
    if err == _spinError_SPINNAKER_ERR_SUCCESS {
        Ok(())
    } else {
        Err(format!("{ctx}: spinError {err}"))
    }
}


fn get_str(f: impl FnOnce(*mut c_char, *mut usize) -> spinError) -> String {
    let mut buf = vec![0u8; 512];
    let mut len: usize = buf.len();
    let err = f(buf.as_mut_ptr() as *mut c_char, &mut len);
    if err != _spinError_SPINNAKER_ERR_SUCCESS {
        return "<unavailable>".to_string();
    }
    unsafe { CStr::from_ptr(buf.as_ptr() as *const c_char) }
        .to_string_lossy()
        .into_owned()
}

fn node_type_name(t: spinNodeType) -> &'static str {
    match t {
        _spinNodeType_ValueNode => "Value",
        _spinNodeType_BaseNode => "Base",
        _spinNodeType_IntegerNode => "Integer",
        _spinNodeType_BooleanNode => "Boolean",
        _spinNodeType_FloatNode => "Float",
        _spinNodeType_CommandNode => "Command",
        _spinNodeType_StringNode => "String",
        _spinNodeType_RegisterNode => "Register",
        _spinNodeType_EnumerationNode => "Enumeration",
        _spinNodeType_EnumEntryNode => "EnumEntry",
        _spinNodeType_CategoryNode => "Category",
        _spinNodeType_PortNode => "Port",
        _ => "Unknown",
    }
}


unsafe fn describe_value(node: spinNodeHandle, ty: spinNodeType) -> String {
    match ty {
        _spinNodeType_IntegerNode => {
            let mut v: i64 = 0;
            let mut lo: i64 = 0;
            let mut hi: i64 = 0;
            let ov = spinIntegerGetValue(node, &mut v);
            let ol = spinIntegerGetMin(node, &mut lo);
            let oh = spinIntegerGetMax(node, &mut hi);
            if ov == _spinError_SPINNAKER_ERR_SUCCESS {
                let range = if ol == _spinError_SPINNAKER_ERR_SUCCESS && oh == _spinError_SPINNAKER_ERR_SUCCESS {
                    format!(" range=[{lo}, {hi}]")
                } else {
                    String::new()
                };
                format!("value={v}{range}")
            } else {
                "<read failed>".to_string()
            }
        }
        _spinNodeType_FloatNode => {
            let mut v: f64 = 0.0;
            let mut lo: f64 = 0.0;
            let mut hi: f64 = 0.0;
            let ov = spinFloatGetValue(node, &mut v);
            let ol = spinFloatGetMin(node, &mut lo);
            let oh = spinFloatGetMax(node, &mut hi);
            if ov == _spinError_SPINNAKER_ERR_SUCCESS {
                let range = if ol == _spinError_SPINNAKER_ERR_SUCCESS && oh == _spinError_SPINNAKER_ERR_SUCCESS {
                    format!(" range=[{lo}, {hi}]")
                } else {
                    String::new()
                };
                format!("value={v}{range}")
            } else {
                "<read failed>".to_string()
            }
        }
        _spinNodeType_BooleanNode => {
            let mut v: bool8_t = 0;
            if spinBooleanGetValue(node, &mut v) == _spinError_SPINNAKER_ERR_SUCCESS {
                format!("value={}", v != 0)
            } else {
                "<read failed>".to_string()
            }
        }
        _spinNodeType_EnumerationNode => {
            let mut cur_name = "<none>".to_string();
            let mut entry: spinNodeHandle = std::ptr::null_mut();
            if spinEnumerationGetCurrentEntry(node, &mut entry) == _spinError_SPINNAKER_ERR_SUCCESS
                && !entry.is_null()
            {
                cur_name = get_str(|buf, len| spinEnumerationEntryGetSymbolic(entry, buf, len));
            }
            let mut num: usize = 0;
            let mut entries = Vec::new();
            if spinEnumerationGetNumEntries(node, &mut num) == _spinError_SPINNAKER_ERR_SUCCESS {
                for i in 0..num {
                    let mut eh: spinNodeHandle = std::ptr::null_mut();
                    if spinEnumerationGetEntryByIndex(node, i, &mut eh) == _spinError_SPINNAKER_ERR_SUCCESS {
                        let mut avail: bool8_t = 0;
                        spinNodeIsAvailable(eh, &mut avail);
                        let sym = get_str(|buf, len| spinEnumerationEntryGetSymbolic(eh, buf, len));
                        if avail != 0 {
                            entries.push(sym);
                        } else {
                            entries.push(format!("({sym} unavailable)"));
                        }
                    }
                }
            }
            format!("current={cur_name} entries=[{}]", entries.join(", "))
        }
        _spinNodeType_StringNode => {
            let mut buf = vec![0u8; 512];
            let mut len: usize = buf.len();
            if spinStringGetValue(node, buf.as_mut_ptr() as *mut c_char, &mut len)
                == _spinError_SPINNAKER_ERR_SUCCESS
            {
                let s = unsafe { CStr::from_ptr(buf.as_ptr() as *const c_char) }.to_string_lossy();
                format!("value=\"{s}\"")
            } else {
                "<read failed>".to_string()
            }
        }
        _ => String::new(),
    }
}

fn main() {
    unsafe {
        let mut sys: spinSystem = std::ptr::null_mut();
        if let Err(e) = ck(spinSystemGetInstance(&mut sys), "SystemGetInstance") {
            eprintln!("FATAL (unexpected -- SystemGetInstance basically never fails): {e}");
            std::process::exit(0);
        }

        let mut list: spinCameraList = std::ptr::null_mut();
        let _ = ck(spinCameraListCreateEmpty(&mut list), "CameraListCreateEmpty");
        if let Err(e) = ck(spinSystemGetCameras(sys, list), "SystemGetCameras") {
            eprintln!("camera busy or SDK error, exiting gracefully: {e}");
            spinSystemReleaseInstance(sys);
            std::process::exit(0);
        }
        let mut n: usize = 0;
        let _ = ck(spinCameraListGetSize(list, &mut n), "CameraListGetSize");
        if n == 0 {
            eprintln!("no cameras found (check cabling/USB, or another process holds it)");
            spinCameraListClear(list);
            spinCameraListDestroy(list);
            spinSystemReleaseInstance(sys);
            std::process::exit(0);
        }
        println!("found {n} camera(s), dumping camera 0's nodemap");

        let mut cam: spinCamera = std::ptr::null_mut();
        let got = ck(spinCameraListGet(list, 0, &mut cam), "CameraListGet");
        spinCameraListClear(list);
        spinCameraListDestroy(list);
        if let Err(e) = got {
            eprintln!("camera busy (held by another process?), exiting gracefully: {e}");
            spinSystemReleaseInstance(sys);
            std::process::exit(0);
        }

        if let Err(e) = ck(spinCameraInit(cam), "CameraInit") {
            eprintln!(
                "CameraInit failed -- camera is almost certainly held by another process \
                 (e.g. fs-app-live.exe or SpinView); exiting gracefully: {e}"
            );
            spinSystemReleaseInstance(sys);
            std::process::exit(0);
        }

        let mut nodemap: spinNodeMapHandle = std::ptr::null_mut();
        if let Err(e) = ck(spinCameraGetNodeMap(cam, &mut nodemap), "GetNodeMap") {
            eprintln!("GetNodeMap failed, exiting gracefully: {e}");
            let _ = spinCameraDeInit(cam);
            spinCameraRelease(cam);
            spinSystemReleaseInstance(sys);
            std::process::exit(0);
        }

        let mut num: usize = 0;
        if let Err(e) = ck(spinNodeMapGetNumNodes(nodemap, &mut num), "GetNumNodes") {
            eprintln!("GetNumNodes failed, exiting gracefully: {e}");
            let _ = spinCameraDeInit(cam);
            spinCameraRelease(cam);
            spinSystemReleaseInstance(sys);
            std::process::exit(0);
        }
        println!("total nodes in map: {num}\n");

        for i in 0..num {
            let mut node: spinNodeHandle = std::ptr::null_mut();
            if spinNodeMapGetNodeByIndex(nodemap, i, &mut node) != _spinError_SPINNAKER_ERR_SUCCESS
                || node.is_null()
            {
                continue;
            }
            let name = get_str(|buf, len| spinNodeGetName(node, buf, len));
            let mut ty: spinNodeType = _spinNodeType_UnknownNode;
            spinNodeGetType(node, &mut ty);
            let mut readable: bool8_t = 0;
            let mut writable: bool8_t = 0;
            spinNodeIsReadable(node, &mut readable);
            spinNodeIsWritable(node, &mut writable);
            let access = format!(
                "{}{}",
                if readable != 0 { "R" } else { "-" },
                if writable != 0 { "W" } else { "-" }
            );
            let detail = if readable != 0 { describe_value(node, ty) } else { String::new() };
            println!("{name}\ttype={}\taccess={access}\t{detail}", node_type_name(ty));
        }

        let _ = spinCameraDeInit(cam);
        spinCameraRelease(cam);
        spinSystemReleaseInstance(sys);
    }
}
