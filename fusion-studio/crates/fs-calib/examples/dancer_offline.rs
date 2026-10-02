
use std::env;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use opencv::calib3d;
use opencv::core::{Mat, Point2f, Point3f, Size, Vector};
use opencv::imgcodecs;
use opencv::imgproc;
use opencv::prelude::*;
use opencv::videoio::{self, VideoCapture, CAP_ANY};
use serde_json::Value;

use fs_calib::board::{self, BoardConfig, BoardKind, Detection};
use fs_calib::engine::CalibSession;

const DEFAULT_AVI: &str = r"C:\Users\98600\Desktop\calibration_dancer_revision\rgb\rgb-08172026145538-0000.avi";
const DEFAULT_E2CALIB_DIR: &str = r"C:\Users\98600\Desktop\calibration_dancer_revision\event\e2calib";
const DEFAULT_REFERENCE_JSON: &str = r"C:\Users\98600\Desktop\calibration_dancer_revision\tmpst8b6gpy_stereo_calibration.json";


fn board_config() -> BoardConfig {
    BoardConfig {
        kind: BoardKind::Checkerboard,
        inner_cols: 7,
        inner_rows: 6,
        rect_w_mm: 82.0,
        rect_h_mm: 98.0,
    }
}

fn main() {
    let mut args: Vec<String> = env::args().skip(1).collect();
    let save_exports_dir = extract_flag_value(&mut args, "--save-exports");

    let (avi_path, png_dir, ref_json_path) = match args.len() {
        0 => (DEFAULT_AVI.to_string(), DEFAULT_E2CALIB_DIR.to_string(), DEFAULT_REFERENCE_JSON.to_string()),
        3 => (args[0].clone(), args[1].clone(), args[2].clone()),
        _ => {
            eprintln!("usage: dancer_offline [--save-exports <dir>] <avi> <e2calib_png_dir> <reference_json>  (or just [--save-exports <dir>] to use the dancer defaults)");
            std::process::exit(2);
        }
    };

    println!("=== fs-calib dancer_offline ===");
    println!("cam0 (avi):        {avi_path}");
    println!("cam1 (e2calib):    {png_dir}");
    println!("reference json:    {ref_json_path}\n");

    let mut png_paths: Vec<PathBuf> = fs::read_dir(&png_dir)
        .unwrap_or_else(|e| panic!("failed to read e2calib dir '{png_dir}': {e}"))
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|p| p.extension().map(|e| e.eq_ignore_ascii_case("png")).unwrap_or(false))
        .collect();
    png_paths.sort();
    let n1 = png_paths.len();
    println!("cam1: found {n1} e2calib PNG files");

    let mut cap = VideoCapture::from_file(&avi_path, CAP_ANY).unwrap_or_else(|e| panic!("failed to open avi '{avi_path}': {e}"));
    if !cap.is_opened().unwrap_or(false) {
        panic!("VideoCapture reports the avi did not open: {avi_path}");
    }
    let avi_reported_count = cap.get(videoio::CAP_PROP_FRAME_COUNT).unwrap_or(-1.0);
    println!("cam0: avi reports CAP_PROP_FRAME_COUNT={avi_reported_count:.0} (container metadata, not always exact)");

    let mut probe = Mat::default();
    if !cap.read(&mut probe).unwrap_or(false) || probe.empty() {
        panic!("avi has no readable frames");
    }
    let size0 = (probe.cols() as u32, probe.rows() as u32);
    let _ = cap.set(videoio::CAP_PROP_POS_FRAMES, 0.0);

    if png_paths.is_empty() {
        panic!("no PNG files found in {png_dir}");
    }
    let first_png = imgcodecs::imread(png_paths[0].to_str().expect("png path is valid utf-8"), imgcodecs::IMREAD_GRAYSCALE)
        .unwrap_or_else(|e| panic!("failed to read first e2calib PNG: {e}"));
    let size1 = (first_png.cols() as u32, first_png.rows() as u32);

    println!("cam0 frame size: {}x{} (8-bit BGR avi, converted to gray via cvtColor)", size0.0, size0.1);
    println!("cam1 frame size: {}x{} (e2calib PNG read with IMREAD_GRAYSCALE -- verified: this forces single-channel output regardless of the PNG's underlying channel count, so no separate conversion step is needed)\n", size1.0, size1.1);

    let board_cfg = board_config();
    let mut session = CalibSession::new(board_cfg.clone(), 40, size0, size1);

    let mut det0_count = 0usize;
    let mut det1_count = 0usize;
    let mut both_count = 0usize;
    let mut eq_rescue0 = 0usize;
    let mut eq_rescue1 = 0usize;
    let mut processed = 0usize;
    let mut all_pairs: Vec<(Detection, Detection)> = Vec::new();

    let mut avi_frame = Mat::default();
    let mut gray0 = Mat::default();
    let t0 = Instant::now();

    for (idx, png_path) in png_paths.iter().enumerate() {
        if !cap.read(&mut avi_frame).unwrap_or(false) || avi_frame.empty() {
            println!("NOTE: avi exhausted at frame {idx} (e2calib has {n1} PNGs) -- pairing min(n0, n1) = {idx} pairs, not {n1}.");
            break;
        }

        imgproc::cvt_color_def(&avi_frame, &mut gray0, imgproc::COLOR_BGR2GRAY).expect("BGR->gray convert (cam0)");
        let (det0, rescued0) = detect_with_fallback(&board_cfg, &gray0);
        if rescued0 {
            eq_rescue0 += 1;
        }

        let png_mat = imgcodecs::imread(png_path.to_str().expect("png path is valid utf-8"), imgcodecs::IMREAD_GRAYSCALE)
            .unwrap_or_else(|e| panic!("failed to read e2calib PNG '{}': {e}", png_path.display()));
        let (det1, rescued1) = detect_with_fallback(&board_cfg, &png_mat);
        if rescued1 {
            eq_rescue1 += 1;
        }

        if det0.is_some() {
            det0_count += 1;
        }
        if det1.is_some() {
            det1_count += 1;
        }
        if det0.is_some() && det1.is_some() {
            both_count += 1;
        }
        if let (Some(d0), Some(d1)) = (&det0, &det1) {
            all_pairs.push((d0.clone(), d1.clone()));
        }

        session.observe(det0, det1);
        processed += 1;

        if processed % 25 == 0 {
            println!(
                "progress: {processed}/{n1} pairs | cam0 det={det0_count} ({:.1}%) cam1 det={det1_count} ({:.1}%) both={both_count} ({:.1}%) | cam0 pool={} cam1 pool={} stereo pairs={}",
                pct(det0_count, processed),
                pct(det1_count, processed),
                pct(both_count, processed),
                session.cam0.pool_len(),
                session.cam1.pool_len(),
                session.stereo.pair_count(),
            );
        }
    }

    let elapsed = t0.elapsed();
    println!("\n=== detection pass complete: {processed} pairs in {:.1}s ({:.1} pairs/s) ===", elapsed.as_secs_f64(), processed as f64 / elapsed.as_secs_f64().max(1e-9));
    println!(
        "cam0 detection rate: {det0_count}/{processed} = {:.1}%  (histogram-equalize fallback rescued {eq_rescue0} additional frames)",
        pct(det0_count, processed)
    );
    println!(
        "cam1 detection rate: {det1_count}/{processed} = {:.1}%  (histogram-equalize fallback rescued {eq_rescue1} additional frames)",
        pct(det1_count, processed)
    );
    println!("both-cameras detection rate: {both_count}/{processed} = {:.1}%  (this is what feeds the stereo engine)", pct(both_count, processed));
    if pct(det0_count, processed) < 2.0 {
        println!(
            "WARNING: cam0 detection rate is near zero even with the histogram-equalize fallback ({} of {} rescued). \
             Board may be off-frame for most of the clip, or motion blur is too severe for both the SB and classic+subpixel detectors.",
            eq_rescue0, processed
        );
    }

    let ref_text = fs::read_to_string(&ref_json_path).unwrap_or_else(|e| panic!("failed to read reference json '{ref_json_path}': {e}"));
    let reference: Value = serde_json::from_str(&ref_text).expect("reference json must parse");
    let ref_stereo_rms = reference["extrinsics"]["stereo_reprojection_error_px"].as_f64().unwrap_or(f64::NAN);

    println!("\n================ FINAL REPORT ================");
    println!("cam0 pool_len={} | cam1 pool_len={} | stereo pair_count={}", session.cam0.pool_len(), session.cam1.pool_len(), session.stereo.pair_count());

    match &session.cam0.state {
        Some(c) => println!(
            "cam0: fx={:.3} fy={:.3} cx={:.3} cy={:.3} dist={:?} rms={:.4}px",
            c.k[0][0], c.k[1][1], c.k[0][2], c.k[1][2], c.dist, c.rms
        ),
        None => println!("cam0: NOT CALIBRATED"),
    }
    match &session.cam1.state {
        Some(c) => println!(
            "cam1: fx={:.3} fy={:.3} cx={:.3} cy={:.3} dist={:?} rms={:.4}px",
            c.k[0][0], c.k[1][1], c.k[0][2], c.k[1][2], c.dist, c.rms
        ),
        None => println!("cam1: NOT CALIBRATED"),
    }
    match &session.stereo.state {
        Some(s) => {
            let rvec_deg = rotation_matrix_to_rvec_deg(&s.r);
            println!(
                "stereo: r(angle-axis,deg)=[{:.4}, {:.4}, {:.4}] |angle|={:.4}deg  t={:?}mm |t|={:.3}mm  rms={:.4}px",
                rvec_deg[0],
                rvec_deg[1],
                rvec_deg[2],
                vec3_norm(rvec_deg),
                s.t,
                vec3_norm(s.t),
                s.rms
            );
        }
        None => println!("stereo: NOT CALIBRATED"),
    }

    if let Some(dir) = &save_exports_dir {
        match save_exports(&session, dir) {
            Ok(msg) => println!("--save-exports: {msg}"),
            Err(e) => println!("--save-exports: FAILED ({e})"),
        }
    }

    match fs_calib::export::to_stereo_calibration_v1(&session) {
        Ok(v) => println!("v1 export: OK (schema={:?})", v["schema"]),
        Err(e) => println!("v1 export: not available yet ({e})"),
    }

    println!(
        "\n================ DIAGNOSTIC: batch stereo calib over ALL {} both-detected pairs (bypasses the online engine's 30-pair ring buffer) ================",
        all_pairs.len()
    );
    if let (Some(cam0), Some(cam1)) = (&session.cam0.state, &session.cam1.state) {
        match batch_stereo_calibrate(&all_pairs, cam0, cam1, &board_cfg, size0) {
            Some((r_batch, t_batch, rms_batch)) => {
                let rvec_deg = rotation_matrix_to_rvec_deg(&r_batch);
                println!(
                    "batch stereo (n={}): r(angle-axis,deg)=[{:.4}, {:.4}, {:.4}] |angle|={:.4}deg  t={:?}mm |t|={:.3}mm  rms={:.4}px",
                    all_pairs.len(),
                    rvec_deg[0],
                    rvec_deg[1],
                    rvec_deg[2],
                    vec3_norm(rvec_deg),
                    t_batch,
                    vec3_norm(t_batch),
                    rms_batch
                );
                let r_ref = mat3_from_ref(&reference["extrinsics"]["R_cam0_to_cam1"]);
                let t_ref = vec3_from_ref(&reference["extrinsics"]["T_cam0_to_cam1"]);
                let rot_err = rotation_angle_deg(&r_batch, &r_ref);
                let dir_err = direction_angle_deg(t_batch, t_ref);
                println!("batch stereo vs reference: rotation_err={rot_err:.4}deg translation_dir_err={dir_err:.4}deg (compare to the ONLINE session's numbers in DELTAS below)");
            }
            None => println!("batch stereo calibration failed (insufficient or mismatched pairs)"),
        }
    } else {
        println!("skipped: cam0/cam1 intrinsics not available");
    }

    println!("\nreference stereo_reprojection_error_px (context only, not a pass/fail criterion): {ref_stereo_rms:.4}px");

    println!("\n================ DELTAS vs REFERENCE (online session -- the official acceptance numbers) ================");
    let mut all_pass = true;
    let intrinsics_threshold_pct = 5.0;
    let rotation_threshold_deg = 2.0;
    let translation_dir_threshold_deg = 5.0;

    for (cam_label, engine_state) in [("cam0", &session.cam0.state), ("cam1", &session.cam1.state)] {
        let Some(calib) = engine_state else {
            println!("[FAIL] {cam_label}: cannot compute deltas -- camera never calibrated");
            all_pass = false;
            continue;
        };
        let (fx_ref, fy_ref, cx_ref, cy_ref) = k_from_ref(&reference["cameras"][cam_label]["K"]);
        let deltas = [
            ("fx", calib.k[0][0], fx_ref),
            ("fy", calib.k[1][1], fy_ref),
            ("cx", calib.k[0][2], cx_ref),
            ("cy", calib.k[1][2], cy_ref),
        ];
        for (name, est, refv) in deltas {
            let delta_pct = pct_delta(est, refv);
            let pass = delta_pct <= intrinsics_threshold_pct;
            all_pass &= pass;
            println!(
                "[{}] {cam_label}.{name}: est={est:.3} ref={refv:.3} delta={delta_pct:.2}% (threshold {intrinsics_threshold_pct}%)",
                if pass { "PASS" } else { "FAIL" }
            );
        }
    }

    match &session.stereo.state {
        Some(stereo) => {
            let r_ref = mat3_from_ref(&reference["extrinsics"]["R_cam0_to_cam1"]);
            let t_ref = vec3_from_ref(&reference["extrinsics"]["T_cam0_to_cam1"]);

            let rot_err_deg = rotation_angle_deg(&stereo.r, &r_ref);
            let dir_err_deg = direction_angle_deg(stereo.t, t_ref);
            let t_norm_ratio = vec3_norm(stereo.t) / vec3_norm(t_ref);

            let rot_pass = rot_err_deg <= rotation_threshold_deg;
            all_pass &= rot_pass;
            println!(
                "[{}] stereo rotation angle(R_est, R_ref) = {rot_err_deg:.4}deg (threshold {rotation_threshold_deg}deg)",
                if rot_pass { "PASS" } else { "FAIL" }
            );

            let dir_pass = dir_err_deg <= translation_dir_threshold_deg;
            all_pass &= dir_pass;
            println!(
                "[{}] stereo translation direction angle(T_est, T_ref) = {dir_err_deg:.4}deg (threshold {translation_dir_threshold_deg}deg)",
                if dir_pass { "PASS" } else { "FAIL" }
            );

            println!("[INFO] |T_est|={:.3}mm |T_ref|={:.3}mm ratio={t_norm_ratio:.4} (not a pass/fail criterion -- different physical rig baseline is expected/irrelevant to direction check)", vec3_norm(stereo.t), vec3_norm(t_ref));
        }
        None => {
            println!("[FAIL] stereo: cannot compute deltas -- stereo never calibrated");
            all_pass = false;
        }
    }

    println!("\n================ OVERALL: {} ================", if all_pass { "PASS" } else { "FAIL" });
    std::process::exit(if all_pass { 0 } else { 1 });
}

fn extract_flag_value(args: &mut Vec<String>, flag: &str) -> Option<String> {
    let pos = args.iter().position(|a| a == flag)?;
    args.remove(pos);
    if pos >= args.len() {
        eprintln!("{flag} requires a directory argument");
        std::process::exit(2);
    }
    Some(args.remove(pos))
}


fn save_exports(session: &CalibSession, dir: &str) -> Result<String, String> {
    let path = std::path::Path::new(dir);
    fs::create_dir_all(path).map_err(|e| format!("failed to create '{dir}': {e}"))?;

    let native = fs_calib::export::to_native_json(session);
    let native_path = path.join("calib.json");
    fs::write(&native_path, serde_json::to_string_pretty(&native).expect("json serialize"))
        .map_err(|e| format!("failed to write {}: {e}", native_path.display()))?;

    let v1 = fs_calib::export::to_stereo_calibration_v1(session)?;
    let v1_path = path.join("stereo_calibration.v1.json");
    fs::write(&v1_path, serde_json::to_string_pretty(&v1).expect("json serialize"))
        .map_err(|e| format!("failed to write {}: {e}", v1_path.display()))?;

    Ok(format!("wrote {} and {}", native_path.display(), v1_path.display()))
}

fn pct(n: usize, total: usize) -> f64 {
    if total == 0 {
        0.0
    } else {
        100.0 * n as f64 / total as f64
    }
}

fn pct_delta(est: f64, refv: f64) -> f64 {
    (est - refv).abs() / refv.abs() * 100.0
}

fn detect_with_fallback(cfg: &BoardConfig, mat: &Mat) -> (Option<Detection>, bool) {
    let (w, h) = (mat.cols() as u32, mat.rows() as u32);
    let buf = mat.data_bytes().expect("gray Mat must be continuous 8-bit single-channel").to_vec();
    if let Some(d) = board::detect(cfg, &buf, w, h) {
        return (Some(d), false);
    }

    let mut eq = Mat::default();
    if imgproc::equalize_hist(mat, &mut eq).is_ok() {
        if let Ok(eq_buf) = eq.data_bytes() {
            if let Some(d) = board::detect(cfg, eq_buf, w, h) {
                return (Some(d), true);
            }
        }
    }
    (None, false)
}

fn rotation_matrix_to_rvec_deg(r: &[[f64; 3]; 3]) -> [f64; 3] {
    let mut mat = Mat::zeros(3, 3, opencv::core::CV_64FC1).expect("alloc 3x3").to_mat().expect("to_mat");
    for i in 0..3 {
        for j in 0..3 {
            *mat.at_2d_mut::<f64>(i as i32, j as i32).expect("at_2d_mut") = r[i][j];
        }
    }
    let mut rvec = Mat::default();
    calib3d::rodrigues_def(&mat, &mut rvec).expect("Rodrigues(R) -> rvec");
    let n = (rvec.rows() * rvec.cols()) as usize;
    let mut out = [0.0; 3];
    for (i, o) in out.iter_mut().enumerate().take(n.min(3)) {
        *o = (*rvec.at::<f64>(i as i32).expect("rvec component")).to_degrees();
    }
    out
}


fn rotation_angle_deg(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> f64 {
    let mut trace = 0.0;
    for i in 0..3 {
        for k in 0..3 {
            trace += a[k][i] * b[k][i];
        }
    }
    let cos_angle = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0);
    cos_angle.acos().to_degrees()
}

fn vec3_norm(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn direction_angle_deg(a: [f64; 3], b: [f64; 3]) -> f64 {
    let dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let cos_dir = (dot / (vec3_norm(a) * vec3_norm(b))).clamp(-1.0, 1.0);
    cos_dir.acos().to_degrees()
}

fn k_from_ref(k: &Value) -> (f64, f64, f64, f64) {
    let fx = k[0][0].as_f64().expect("K[0][0]");
    let fy = k[1][1].as_f64().expect("K[1][1]");
    let cx = k[0][2].as_f64().expect("K[0][2]");
    let cy = k[1][2].as_f64().expect("K[1][2]");
    (fx, fy, cx, cy)
}

fn mat3_from_ref(v: &Value) -> [[f64; 3]; 3] {
    let mut m = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            m[i][j] = v[i][j].as_f64().expect("R entry");
        }
    }
    m
}

fn vec3_from_ref(v: &Value) -> [f64; 3] {
    let mut out = [0.0; 3];
    for (i, o) in out.iter_mut().enumerate() {
        *o = v[i].as_f64().expect("T entry");
    }
    out
}

fn batch_stereo_calibrate(pairs: &[(Detection, Detection)], cam0: &fs_calib::CamCalib, cam1: &fs_calib::CamCalib, board_cfg: &BoardConfig, size0: (u32, u32)) -> Option<([[f64; 3]; 3], [f64; 3], f64)> {
    let obj_template: Vector<Point3f> = Vector::from_iter(board_cfg.object_points());
    let mut obj_pts: Vector<Vector<Point3f>> = Vector::new();
    let mut img_pts0: Vector<Vector<Point2f>> = Vector::new();
    let mut img_pts1: Vector<Vector<Point2f>> = Vector::new();
    for (d0, d1) in pairs {
        if d0.corners.len() != obj_template.len() || d1.corners.len() != obj_template.len() {
            continue;
        }
        obj_pts.push(obj_template.clone());
        img_pts0.push(Vector::from_iter(d0.corners.iter().map(|&(x, y)| Point2f::new(x, y))));
        img_pts1.push(Vector::from_iter(d1.corners.iter().map(|&(x, y)| Point2f::new(x, y))));
    }
    if obj_pts.is_empty() {
        return None;
    }

    let mut k1 = mat3x3_to_mat(&cam0.k);
    let mut d1 = vecn_to_mat(&cam0.dist);
    let mut k2 = mat3x3_to_mat(&cam1.k);
    let mut d2 = vecn_to_mat(&cam1.dist);
    let image_size = Size::new(size0.0 as i32, size0.1 as i32);
    let mut r = Mat::default();
    let mut t = Mat::default();
    let mut e = Mat::default();
    let mut f = Mat::default();
    let mut rvecs: Vector<Mat> = Vector::new();
    let mut tvecs: Vector<Mat> = Vector::new();
    let mut per_view_errors = Mat::default();

    let rms = calib3d::stereo_calibrate_extended_def(
        &obj_pts, &img_pts0, &img_pts1, &mut k1, &mut d1, &mut k2, &mut d2, image_size, &mut r, &mut t, &mut e, &mut f, &mut rvecs, &mut tvecs, &mut per_view_errors,
    )
    .ok()?;

    let r_out = mat_to_mat3x3(&r);
    let t_out = mat_to_vecn(&t);
    if t_out.len() < 3 {
        return None;
    }
    Some((r_out, [t_out[0], t_out[1], t_out[2]], rms))
}

fn mat3x3_to_mat(m: &[[f64; 3]; 3]) -> Mat {
    let mut mat = Mat::zeros(3, 3, opencv::core::CV_64FC1).expect("alloc 3x3").to_mat().expect("to_mat");
    for r in 0..3 {
        for c in 0..3 {
            *mat.at_2d_mut::<f64>(r as i32, c as i32).expect("at_2d_mut") = m[r][c];
        }
    }
    mat
}

fn vecn_to_mat(v: &[f64]) -> Mat {
    let mut mat = Mat::zeros(v.len() as i32, 1, opencv::core::CV_64FC1).expect("alloc vec").to_mat().expect("to_mat");
    for (i, val) in v.iter().enumerate() {
        *mat.at_2d_mut::<f64>(i as i32, 0).expect("at_2d_mut") = *val;
    }
    mat
}

fn mat_to_mat3x3(mat: &Mat) -> [[f64; 3]; 3] {
    let mut m = [[0.0; 3]; 3];
    for (r, row) in m.iter_mut().enumerate() {
        for (c, cell) in row.iter_mut().enumerate() {
            *cell = *mat.at_2d::<f64>(r as i32, c as i32).expect("at_2d");
        }
    }
    m
}

fn mat_to_vecn(mat: &Mat) -> Vec<f64> {
    let n = (mat.rows() * mat.cols()) as usize;
    (0..n).map(|i| *mat.at::<f64>(i as i32).expect("at")).collect()
}
