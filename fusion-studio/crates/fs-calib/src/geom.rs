
use opencv::calib3d;
use opencv::core::{Mat, Point2f, Size, CV_32FC1, CV_64FC1};
use opencv::prelude::*;
use serde_json::{json, Value};

use crate::active::{ActiveCalibration, CamIntrinsics};
use crate::sim::{mat3x3_from, vecn_from};

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum GeomMode {
    Off,
    Undistort,
    SharedK { target: (u32, u32) },
}


#[derive(Debug)]
pub struct CamMaps {
    pub k_new: [[f64; 3]; 3],
    pub out_size: (u32, u32),
    pub frame_maps: (Mat, Mat),
    pub event_lut: Vec<u32>,
    pub lut_dropped_cells: u32,
}


pub fn derive(active: &ActiveCalibration, mode: GeomMode) -> Result<(CamMaps, CamMaps), String> {
    validate_cam(&active.cam0, "cam0")?;
    validate_cam(&active.cam1, "cam1")?;

    match mode {
        GeomMode::Off => Err("Off 模式是恒等变换,不需要映射表 —— 调用方不该走到这里".into()),
        GeomMode::Undistort => {
            let m0 = cam_maps(&active.cam0, undistort_k(&active.cam0), active.cam0.image_size, "cam0")?;
            let m1 = cam_maps(&active.cam1, undistort_k(&active.cam1), active.cam1.image_size, "cam1")?;
            Ok((m0, m1))
        }
        GeomMode::SharedK { target } => {
            validate_shared_target(target)?;
            let k_shared = compute_shared_k(&[(active.cam0.k, active.cam0.image_size), (active.cam1.k, active.cam1.image_size)], target);
            let m0 = cam_maps(&active.cam0, k_shared, target, "cam0")?;
            let m1 = cam_maps(&active.cam1, k_shared, target, "cam1")?;
            Ok((m0, m1))
        }
    }
}

pub fn validate_shared_target(target: (u32, u32)) -> Result<(), String> {
    for (name, v) in [("宽", target.0), ("高", target.1)] {
        if v < 16 {
            return Err(format!("目标{name} {v} 太小:最小 16"));
        }
        if v > 4094 {
            return Err(format!("目标{name} {v} 超过上限 4094(events.bin 坐标只有 12 位)"));
        }
        if v % 2 != 0 {
            return Err(format!("目标{name} {v} 是奇数:HEVC 编码要求偶数尺寸"));
        }
    }
    Ok(())
}

pub fn post_op_document(active: &ActiveCalibration, mode: GeomMode, maps: (&CamMaps, &CamMaps)) -> Result<Value, String> {
    if mode == GeomMode::Off {
        return Err("Off 模式没有操作后文档 —— 原始 document 原样使用".into());
    }

    let mut doc = active.document.clone();
    for (label, m) in [("cam0", maps.0), ("cam1", maps.1)] {
        let cam = doc["cameras"][label]
            .as_object_mut()
            .ok_or_else(|| format!("document 缺少 cameras.{label} 块,无法生成操作后文档"))?;
        let dist_len = cam
            .get("dist")
            .and_then(|d| d.as_array())
            .map(|a| a.len())
            .ok_or_else(|| format!("document 的 cameras.{label}.dist 不是数组"))?;
        cam.insert("K".into(), json!(m.k_new));
        cam.insert("dist".into(), json!(vec![0.0; dist_len]));
        cam.insert("image_size".into(), json!([m.out_size.0, m.out_size.1]));
        let resize = cam.entry("resize_op").or_insert_with(|| json!({}));
        if !resize.is_object() {
            *resize = json!({});
        }
        resize["target_size"] = json!([m.out_size.0, m.out_size.1]);
    }

    if let GeomMode::SharedK { target } = mode {
        doc["shared_intrinsics"] = json!({
            "K": json!(maps.0.k_new),
            "image_size": [target.0, target.1],
        });
    }

    Ok(doc)
}


pub fn compute_shared_k(cams: &[([[f64; 3]; 3], (u32, u32))], target: (u32, u32)) -> [[f64; 3]; 3] {
    let w_t = target.0 as f64;
    let h_t = target.1 as f64;

    let mut f_new = f64::MIN;
    for (k, (w, h)) in cams {
        f_new = f_new.max(k[0][0] * w_t / *w as f64);
        f_new = f_new.max(k[1][1] * h_t / *h as f64);
    }

    [[f_new, 0.0, w_t / 2.0], [0.0, f_new, h_t / 2.0], [0.0, 0.0, 1.0]]
}


fn undistort_k(cam: &CamIntrinsics) -> [[f64; 3]; 3] {
    let (w, h) = cam.image_size;
    [[cam.k[0][0], 0.0, w as f64 / 2.0], [0.0, cam.k[1][1], h as f64 / 2.0], [0.0, 0.0, 1.0]]
}

fn validate_cam(cam: &CamIntrinsics, label: &str) -> Result<(), String> {
    let (w, h) = cam.image_size;
    if w == 0 || h == 0 {
        return Err(format!("{label} 的 image_size 是 {w}x{h}:尺寸为零,无法建表"));
    }
    if w > 4094 || h > 4094 {
        return Err(format!("{label} 的 image_size 是 {w}x{h}:超过上限 4094(events.bin 坐标只有 12 位),拒绝建表"));
    }
    let (fx, fy) = (cam.k[0][0], cam.k[1][1]);
    if !(fx.is_finite() && fy.is_finite() && fx > 0.0 && fy > 0.0) {
        return Err(format!("{label} 的焦距非法(fx={fx}, fy={fy}):标定已退化,拒绝建表"));
    }
    if cam.k.iter().flatten().any(|v| !v.is_finite()) {
        return Err(format!("{label} 的 K 含非有限值:标定已退化,拒绝建表"));
    }
    if !matches!(cam.dist.len(), 0 | 4 | 5 | 8 | 12 | 14) {
        return Err(format!("{label} 的畸变系数有 {} 个,OpenCV 只接受 0/4/5/8/12/14 个", cam.dist.len()));
    }
    if cam.dist.iter().any(|d| !d.is_finite()) {
        return Err(format!("{label} 的畸变系数含非有限值:标定已退化,拒绝建表"));
    }
    Ok(())
}

fn cam_maps(cam: &CamIntrinsics, k_new: [[f64; 3]; 3], out_size: (u32, u32), label: &str) -> Result<CamMaps, String> {
    let k_mat = mat3x3_from(&cam.k).map_err(|e| format!("{label}.K 转 Mat 失败:{e}"))?;
    let dist_mat = vecn_from(&cam.dist).map_err(|e| format!("{label}.dist 转 Mat 失败:{e}"))?;
    let k_new_mat = mat3x3_from(&k_new).map_err(|e| format!("{label} 的 K_new 转 Mat 失败:{e}"))?;
    let eye = Mat::eye(3, 3, CV_64FC1)
        .and_then(|e| e.to_mat())
        .map_err(|e| format!("构造单位旋转失败:{e}"))?;

    let size = Size::new(out_size.0 as i32, out_size.1 as i32);
    let mut map_x = Mat::default();
    let mut map_y = Mat::default();
    calib3d::init_undistort_rectify_map(&k_mat, &dist_mat, &eye, &k_new_mat, size, CV_32FC1, &mut map_x, &mut map_y)
        .map_err(|e| format!("{label} 生成 remap 映射表失败:{e}"))?;

    let (sw, sh) = cam.image_size;
    let mut src_pts = Vec::with_capacity((sw as usize) * (sh as usize));
    for y in 0..sh {
        for x in 0..sw {
            src_pts.push(Point2f::new(x as f32, y as f32));
        }
    }
    let src = Mat::from_slice(&src_pts).map_err(|e| format!("{label} 源像素批转 Mat 失败:{e}"))?;
    let mut dst = Mat::default();
    calib3d::undistort_points(&src, &mut dst, &k_mat, &dist_mat, &eye, &k_new_mat)
        .map_err(|e| format!("{label} undistortPoints 失败:{e}"))?;
    let dst_pts = dst.data_typed::<Point2f>().map_err(|e| format!("{label} 读取 undistortPoints 结果失败:{e}"))?;
    if dst_pts.len() != src_pts.len() {
        return Err(format!("{label} undistortPoints 输出 {} 点,输入 {} 点 —— OpenCV 行为异常", dst_pts.len(), src_pts.len()));
    }

    let (ow, oh) = out_size;
    let mut event_lut = Vec::with_capacity(dst_pts.len());
    let mut lut_dropped_cells = 0u32;
    for p in dst_pts {
        let (x, y) = (f64::from(p.x).round(), f64::from(p.y).round());
        if x >= 0.0 && y >= 0.0 && x < f64::from(ow) && y < f64::from(oh) {
            event_lut.push(y as u32 * ow + x as u32);
        } else {
            event_lut.push(u32::MAX);
            lut_dropped_cells += 1;
        }
    }

    Ok(CamMaps { k_new, out_size, frame_maps: (map_x, map_y), event_lut, lut_dropped_cells })
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::active::{load_stereo_calibration_v1, CalibSource};
    use serde_json::json;

    fn doc_with(cams: [(f64, f64, f64, f64, (u32, u32), Vec<f64>); 2]) -> Value {
        let cam = |label: &str, (fx, fy, cx, cy, (w, h), dist): &(f64, f64, f64, f64, (u32, u32), Vec<f64>)| {
            json!({
                "camera_label": label,
                "original_image_size": [w, h],
                "resize_op": { "target_size": [w, h] },
                "image_size": [w, h],
                "K": [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
                "dist": dist,
                "reprojection_error_px": 0.25,
                "checkerboard": { "board_size": [6, 7], "rect_size_mm": [82.0, 98.0] },
            })
        };
        json!({
            "schema": "stereo_calibration.v1",
            "cameras": { "cam0": cam("cam0", &cams[0]), "cam1": cam("cam1", &cams[1]) },
            "extrinsics": {
                "R_cam0_to_cam1": [
                    [0.9998476951563913, -0.017452406437283512, 0.0],
                    [0.017452406437283512, 0.9998476951563913, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                "T_cam0_to_cam1": [100.123456789, 0.5, -2.25],
                "stereo_reprojection_error_px": 0.4321,
            },
            "checkerboard": { "board_size": [6, 7], "rect_size_mm": [82.0, 98.0] },
            "shared_intrinsics": {
                "K": [[1000.0, 0.0, 640.0], [0.0, 1000.0, 512.0], [0.0, 0.0, 1.0]],
                "image_size": [1280, 1024],
            },
        })
    }

    fn fixture() -> ActiveCalibration {
        let doc = doc_with([
            (1000.25, 995.5, 660.75, 500.125, (1280, 1024), vec![-0.2, 0.05, 0.001, -0.002, 0.03]),
            (900.5, 905.25, 630.5, 370.75, (1280, 720), vec![-0.15, 0.03, 0.0, 0.0, 0.0]),
        ]);
        load_stereo_calibration_v1(doc, CalibSource::LiveSession).expect("fixture doc must load")
    }

    fn small_fixture(w: u32, h: u32, f: f64, cx: f64, cy: f64, k1: f64) -> ActiveCalibration {
        let cam = (f, f, cx, cy, (w, h), vec![k1, 0.0, 0.0, 0.0, 0.0]);
        load_stereo_calibration_v1(doc_with([cam.clone(), cam]), CalibSource::LiveSession).expect("small fixture must load")
    }

    #[test]
    fn undistort_k_keeps_focals_and_centers_pp() {
        let a = fixture();
        let (m0, m1) = derive(&a, GeomMode::Undistort).expect("undistort derive must succeed");

        assert_eq!(m0.k_new, [[1000.25, 0.0, 640.0], [0.0, 995.5, 512.0], [0.0, 0.0, 1.0]]);
        assert_eq!(m0.out_size, (1280, 1024));
        assert_eq!(m1.k_new, [[900.5, 0.0, 640.0], [0.0, 905.25, 360.0], [0.0, 0.0, 1.0]]);
        assert_eq!(m1.out_size, (1280, 720));

        for m in [&m0, &m1] {
            assert_eq!(m.frame_maps.0.typ(), CV_32FC1);
            assert_eq!(m.frame_maps.1.typ(), CV_32FC1);
            let want = Size::new(m.out_size.0 as i32, m.out_size.1 as i32);
            assert_eq!(m.frame_maps.0.size().unwrap(), want);
            assert_eq!(m.frame_maps.1.size().unwrap(), want);
        }
        assert_eq!(m0.event_lut.len(), 1280 * 1024);
        assert_eq!(m1.event_lut.len(), 1280 * 720);
    }

    #[test]
    fn compute_shared_k_matches_hand_computed_values() {
        let k0 = [[1000.0, 0.0, 660.0], [0.0, 995.0, 500.0], [0.0, 0.0, 1.0]];
        let k1 = [[900.0, 0.0, 630.0], [0.0, 905.0, 370.0], [0.0, 0.0, 1.0]];
        let cams = [(k0, (1280u32, 1024u32)), (k1, (1280u32, 720u32))];

        let f = 905.0 * 1024.0 / 720.0;
        assert_eq!(compute_shared_k(&cams, (1280, 1024)), [[f, 0.0, 640.0], [0.0, f, 512.0], [0.0, 0.0, 1.0]]);

        let f = 905.0 * 480.0 / 720.0;
        assert_eq!(compute_shared_k(&cams, (640, 480)), [[f, 0.0, 320.0], [0.0, f, 240.0], [0.0, 0.0, 1.0]]);
    }

    #[test]
    fn event_lut_round_trips_synthetic_distortion() {
        let (w, h, f, k1) = (320u32, 240u32, 260.0f64, -0.2f64);
        let a = small_fixture(w, h, f, w as f64 / 2.0, h as f64 / 2.0, k1);
        let (m0, _) = derive(&a, GeomMode::Undistort).expect("small undistort derive");
        assert_eq!(m0.event_lut.len(), (w * h) as usize);

        let (cx, cy) = (w as f64 / 2.0, h as f64 / 2.0);
        let mut checked = 0u32;
        for ys in (4..h - 4).step_by(13) {
            for xs in (4..w - 4).step_by(11) {
                let ad = (xs as f64 - cx) / f;
                let bd = (ys as f64 - cy) / f;
                let (mut au, mut bu) = (ad, bd);
                for _ in 0..50 {
                    let d = 1.0 + k1 * (au * au + bu * bu);
                    au = ad / d;
                    bu = bd / d;
                }
                let ideal = (f * au + cx, f * bu + cy);
                if ideal.0 < 1.0 || ideal.1 < 1.0 || ideal.0 > w as f64 - 2.0 || ideal.1 > h as f64 - 2.0 {
                    continue;
                }
                let idx = m0.event_lut[(ys * w + xs) as usize];
                assert_ne!(idx, u32::MAX, "界内像素 ({xs},{ys}) 不该被丢弃");
                let (tx, ty) = ((idx % w) as f64, (idx / w) as f64);
                assert!((tx - ideal.0).abs() <= 0.51, "({xs},{ys}) -> x {tx},真值 {:.4}", ideal.0);
                assert!((ty - ideal.1).abs() <= 0.51, "({xs},{ys}) -> y {ty},真值 {:.4}", ideal.1);
                checked += 1;
            }
        }
        assert!(checked > 100, "采样覆盖太少({checked}),测试没测到东西");
        let sentinels = m0.event_lut.iter().filter(|&&v| v == u32::MAX).count() as u32;
        assert_eq!(sentinels, m0.lut_dropped_cells);
    }

    #[test]
    fn event_lut_marks_out_of_range_as_drop() {
        let (w, h) = (128u32, 96u32);
        let a = small_fixture(w, h, 100.0, 120.0, h as f64 / 2.0, 0.0);
        let (m0, _) = derive(&a, GeomMode::Undistort).expect("skewed-pp derive");

        assert_eq!(m0.lut_dropped_cells, 56 * 96, "左侧 56 列 × 96 行应全部出界");
        let sentinels = m0.event_lut.iter().filter(|&&v| v == u32::MAX).count() as u32;
        assert_eq!(sentinels, m0.lut_dropped_cells);
        assert_eq!(m0.event_lut[0], u32::MAX, "左上角必然出界");
        assert_eq!(m0.event_lut[(10 * w + 100) as usize], 10 * w + (100 - 56));
        assert!(m0.event_lut.iter().all(|&v| v == u32::MAX || v < w * h));
    }

    #[test]
    fn post_op_document_keeps_extrinsics_bitwise() {
        let a = fixture();
        for mode in [GeomMode::Undistort, GeomMode::SharedK { target: (1280, 1024) }] {
            let (m0, m1) = derive(&a, mode).expect("derive must succeed");
            let doc = post_op_document(&a, mode, (&m0, &m1)).expect("post-op document");
            assert_eq!(doc["extrinsics"], a.document["extrinsics"], "{mode:?} 的 extrinsics 变了");
            assert_eq!(
                serde_json::to_string(&doc["extrinsics"]).unwrap(),
                serde_json::to_string(&a.document["extrinsics"]).unwrap(),
                "{mode:?} 的 extrinsics 序列化不一致"
            );
        }
    }

    #[test]
    fn post_op_document_is_loadable_v1() {
        let a = fixture();

        let (m0, m1) = derive(&a, GeomMode::Undistort).expect("undistort derive");
        let doc = post_op_document(&a, GeomMode::Undistort, (&m0, &m1)).expect("post-op undistort");
        assert_eq!(doc["cameras"]["cam0"]["resize_op"]["target_size"], json!([1280, 1024]));
        assert_eq!(doc["cameras"]["cam1"]["resize_op"]["target_size"], json!([1280, 720]));
        assert_eq!(doc["cameras"]["cam0"]["original_image_size"], a.document["cameras"]["cam0"]["original_image_size"]);
        let loaded = load_stereo_calibration_v1(doc, CalibSource::LiveSession).expect("post-op undistort doc must load");
        for (cam, m, orig) in [(&loaded.cam0, &m0, &a.cam0), (&loaded.cam1, &m1, &a.cam1)] {
            assert_eq!(cam.dist.len(), orig.dist.len(), "dist 长度必须保留");
            assert!(cam.dist.iter().all(|&d| d == 0.0), "dist 必须全零");
            assert_eq!(cam.k, m.k_new);
            assert_eq!(cam.image_size, m.out_size);
        }
        assert_eq!(loaded.r, a.r);
        assert_eq!(loaded.t, a.t);

        let target = (960u32, 640u32);
        let (s0, s1) = derive(&a, GeomMode::SharedK { target }).expect("shared-k derive");
        assert_eq!(s0.k_new, s1.k_new, "SharedK 下双相机必须同 K");
        assert_eq!(s0.out_size, target);
        assert_eq!(s1.out_size, target);
        let doc = post_op_document(&a, GeomMode::SharedK { target }, (&s0, &s1)).expect("post-op shared-k");
        assert_eq!(doc["shared_intrinsics"]["K"], json!(s0.k_new));
        assert_eq!(doc["shared_intrinsics"]["image_size"], json!([960, 640]));
        let loaded = load_stereo_calibration_v1(doc, CalibSource::LiveSession).expect("post-op shared-k doc must load");
        assert_eq!(loaded.cam0.k, s0.k_new);
        assert_eq!(loaded.cam1.k, s0.k_new);
        assert_eq!(loaded.cam0.image_size, target);
        assert_eq!(loaded.cam1.image_size, target);
        assert!(loaded.cam0.dist.iter().all(|&d| d == 0.0));
        assert!(loaded.cam1.dist.iter().all(|&d| d == 0.0));
    }

    #[test]
    fn shared_target_size_is_validated() {
        for ok in [(16, 16), (1280, 1024), (4094, 4094)] {
            validate_shared_target(ok).unwrap_or_else(|e| panic!("{ok:?} 应合法:{e}"));
        }
        for bad in [(1281, 1024), (1280, 1023), (14, 720), (1280, 14), (4096, 1024), (1280, 4096), (0, 0)] {
            validate_shared_target(bad).expect_err("非法尺寸必须拒绝");
        }
        let err = validate_shared_target((1281, 1024)).unwrap_err();
        assert!(err.contains("偶数"), "奇数错误要提偶数要求:{err}");
        let err = validate_shared_target((4096, 1024)).unwrap_err();
        assert!(err.contains("4094"), "超界错误要提上限:{err}");
        let a = fixture();
        derive(&a, GeomMode::SharedK { target: (1281, 1024) }).expect_err("derive 必须拒绝非法 target");
    }

    #[test]
    fn derive_rejects_off_and_degenerate_input() {
        let a = fixture();

        let err = derive(&a, GeomMode::Off).expect_err("Off 不该建表");
        assert!(err.contains("Off"), "{err}");
        let (m0, m1) = derive(&a, GeomMode::Undistort).expect("undistort derive");
        post_op_document(&a, GeomMode::Off, (&m0, &m1)).expect_err("Off 没有操作后文档");

        let mut bad = a.clone();
        bad.cam0.image_size = (0, 0);
        let err = derive(&bad, GeomMode::Undistort).expect_err("零尺寸必须拒绝");
        assert!(err.contains("cam0"), "{err}");

        let mut bad = a.clone();
        bad.cam1.dist = vec![0.1, 0.2, 0.3];
        let err = derive(&bad, GeomMode::Undistort).expect_err("3 个畸变系数必须拒绝");
        assert!(err.contains("cam1"), "{err}");

        let mut bad = a.clone();
        bad.cam0.k[0][0] = 0.0;
        let err = derive(&bad, GeomMode::Undistort).expect_err("零焦距必须拒绝");
        assert!(err.contains("cam0"), "{err}");
        let mut bad = a.clone();
        bad.cam1.k[1][1] = f64::NAN;
        let err = derive(&bad, GeomMode::Undistort).expect_err("NaN 焦距必须拒绝");
        assert!(err.contains("cam1"), "{err}");
    }

    #[test]
    fn oversized_camera_dims_are_rejected_before_lut_allocation() {
        let a = fixture();
        let mut bad = a.clone();
        bad.cam1.image_size = (66000, 66000);
        let err = derive(&bad, GeomMode::Undistort).expect_err("超大尺寸必须拒绝");
        assert!(err.contains("cam1"), "错误要点名相机:{err}");
        assert!(err.contains("4094"), "错误要提上限:{err}");
    }

    #[test]
    fn shared_k_lut_uses_target_stride_with_unequal_sizes() {
        let (sw, sh) = (64u32, 48u32);
        let a = small_fixture(sw, sh, 100.0, sw as f64 / 2.0, sh as f64 / 2.0, 0.0);
        let target = (128u32, 96u32);
        let (m0, m1) = derive(&a, GeomMode::SharedK { target }).expect("shared-k derive");

        assert_eq!(m0.k_new, [[200.0, 0.0, 64.0], [0.0, 200.0, 48.0], [0.0, 0.0, 1.0]]);
        assert_eq!(m0.out_size, target);
        assert_eq!(m0.lut_dropped_cells, 0, "2 倍放大下全部源像素都该在界内");
        let ow = target.0;
        for (x, y) in [(0u32, 0u32), (63, 0), (0, 47), (63, 47), (17, 31), (40, 9)] {
            assert_eq!(m0.event_lut[(y * sw + x) as usize], (2 * y) * ow + 2 * x, "源 ({x},{y}) 的目标索引必须按输出宽 {ow} 作行距");
        }
        assert_eq!(m1.event_lut, m0.event_lut);
    }
}
