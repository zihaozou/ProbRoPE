use serde_json::{json, Value};

use crate::board::{BoardConfig, BoardKind};
use crate::engine::{CalibSession, CamCalib, IntrinsicsEngine, StereoEngine};



pub fn to_native_json(session: &CalibSession) -> Value {
    json!({
        "schema": "fusion-studio-calib.v1",
        "board": board_json(&session.cfg),
        "cam0": intrinsics_engine_json(&session.cam0),
        "cam1": intrinsics_engine_json(&session.cam1),
        "stereo": stereo_engine_json(&session.stereo),
    })
}

fn board_json(cfg: &BoardConfig) -> Value {
    let (kind, marker_mm, dict) = match &cfg.kind {
        BoardKind::Checkerboard => ("checkerboard", None, None),
        BoardKind::Charuco { marker_mm, dict } => ("charuco", Some(*marker_mm), Some(dict.clone())),
    };
    json!({
        "kind": kind,
        "marker_mm": marker_mm,
        "dict": dict,
        "inner_cols": cfg.inner_cols,
        "inner_rows": cfg.inner_rows,
        "rect_w_mm": cfg.rect_w_mm,
        "rect_h_mm": cfg.rect_h_mm,
    })
}

fn intrinsics_engine_json(engine: &IntrinsicsEngine) -> Value {
    let (w, h) = engine.image_size();
    match &engine.state {
        Some(c) => json!({
            "calibrated": true,
            "image_size": [w, h],
            "k": mat3(&c.k),
            "dist": c.dist,
            "rms": c.rms,
            "per_view_rms": c.per_view_rms,
            "pool_len": engine.pool_len(),
            "converged": engine.converged(),
        }),
        None => json!({
            "calibrated": false,
            "image_size": [w, h],
            "pool_len": engine.pool_len(),
            "converged": false,
        }),
    }
}

fn stereo_engine_json(engine: &StereoEngine) -> Value {
    match &engine.state {
        Some(s) => json!({
            "calibrated": true,
            "r": mat3(&s.r),
            "t": s.t,
            "rms": s.rms,
            "pair_count": engine.pair_count(),
        }),
        None => json!({
            "calibrated": false,
            "pair_count": engine.pair_count(),
        }),
    }
}



pub fn to_stereo_calibration_v1(session: &CalibSession) -> Result<Value, String> {
    let cam0 = session.cam0.state.as_ref().ok_or("cam0 has not calibrated yet")?;
    let cam1 = session.cam1.state.as_ref().ok_or("cam1 has not calibrated yet")?;
    let stereo = session.stereo.state.as_ref().ok_or("stereo has not calibrated yet")?;

    let size0 = session.cam0.image_size();
    let size1 = session.cam1.image_size();

    Ok(json!({
        "schema": "stereo_calibration.v1",
        "cameras": {
            "cam0": camera_block_json("cam0", cam0, size0, &session.cfg),
            "cam1": camera_block_json("cam1", cam1, size1, &session.cfg),
        },
        "extrinsics": extrinsics_json(stereo),
        "checkerboard": checkerboard_json(&session.cfg),
        "shared_intrinsics": shared_intrinsics_json(cam0, size0, cam1, size1),
    }))
}


fn camera_block_json(label: &str, calib: &CamCalib, size: (u32, u32), cfg: &BoardConfig) -> Value {
    let (w, h) = size;
    json!({
        "camera_label": label,
        "original_image_size": [w, h],
        "resize_op": { "target_size": [w, h] },
        "image_size": [w, h],
        "K": mat3(&calib.k),
        "dist": calib.dist,
        "reprojection_error_px": calib.rms,
        "checkerboard": checkerboard_json(cfg),
    })
}


fn extrinsics_json(stereo: &crate::engine::StereoCalib) -> Value {
    json!({
        "R_cam0_to_cam1": mat3(&stereo.r),
        "T_cam0_to_cam1": stereo.t,
        "stereo_reprojection_error_px": stereo.rms,
    })
}


fn checkerboard_json(cfg: &BoardConfig) -> Value {
    json!({
        "board_size": [cfg.inner_rows, cfg.inner_cols],
        "rect_size_mm": [cfg.rect_w_mm, cfg.rect_h_mm],
    })
}


fn shared_intrinsics_json(cam0: &CamCalib, size0: (u32, u32), cam1: &CamCalib, size1: (u32, u32)) -> Value {
    let target = (size0.0.max(size1.0), size0.1.max(size1.1));
    let k_shared = crate::geom::compute_shared_k(&[(cam0.k, size0), (cam1.k, size1)], target);
    json!({
        "K": mat3(&k_shared),
        "image_size": [target.0, target.1],
    })
}

fn mat3(m: &[[f64; 3]; 3]) -> Value {
    json!([m[0], m[1], m[2]])
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::BoardKind;
    use crate::sim::SimRig;
    use std::collections::BTreeSet;

    const INNER_COLS: u32 = 7;
    const INNER_ROWS: u32 = 6;
    const RECT_W_MM: f64 = 82.0;
    const RECT_H_MM: f64 = 98.0;


    const REFERENCE_JSON: &str = include_str!("../testdata/dancer_stereo_calibration_reference.json");

    fn test_board() -> BoardConfig {
        BoardConfig {
            kind: BoardKind::Checkerboard,
            inner_cols: INNER_COLS,
            inner_rows: INNER_ROWS,
            rect_w_mm: RECT_W_MM,
            rect_h_mm: RECT_H_MM,
        }
    }

    fn calibrated_session() -> CalibSession {
        let board = test_board();
        let rig = SimRig::realistic();
        let views = rig.gen_views(&board, 30, 42);
        assert_eq!(views.len(), 30, "sim rig should produce the requested number of valid views");

        let mut session = CalibSession::new(board, 40, rig.size0, rig.size1);
        for (d0, d1) in views {
            session.observe(Some(d0), Some(d1));
        }
        assert!(session.cam0.state.is_some());
        assert!(session.cam1.state.is_some());
        assert!(session.stereo.state.is_some());
        session
    }

    fn key_tree(v: &Value, path: &str, out: &mut Vec<(String, BTreeSet<String>)>) {
        match v {
            Value::Object(map) => {
                let keys: BTreeSet<String> = map.keys().cloned().collect();
                out.push((path.to_string(), keys));
                for (k, child) in map {
                    key_tree(child, &format!("{path}/{k}"), out);
                }
            }
            Value::Array(items) => {
                for item in items {
                    key_tree(item, &format!("{path}[]"), out);
                }
            }
            _ => {}
        }
    }

    #[test]
    fn native_export_smoke() {
        let session = calibrated_session();
        let v = to_native_json(&session);
        assert_eq!(v["schema"], "fusion-studio-calib.v1");
        assert_eq!(v["cam0"]["calibrated"], true);
        assert_eq!(v["cam1"]["calibrated"], true);
        assert_eq!(v["stereo"]["calibrated"], true);
        assert!(v["cam0"]["k"].is_array());
        assert!(v["stereo"]["pair_count"].as_u64().unwrap() > 0);
    }

    #[test]
    fn native_export_uncalibrated_session_does_not_panic() {
        let board = test_board();
        let session = CalibSession::new(board, 40, (640, 480), (640, 480));
        let v = to_native_json(&session);
        assert_eq!(v["cam0"]["calibrated"], false);
        assert_eq!(v["stereo"]["calibrated"], false);
    }

    #[test]
    fn v1_export_errs_when_not_fully_calibrated() {
        let board = test_board();
        let session = CalibSession::new(board, 40, (640, 480), (640, 480));
        let err = to_stereo_calibration_v1(&session).expect_err("empty session must not export");
        assert!(err.contains("cam0"), "error should mention which piece is missing: {err}");
    }

    #[test]
    fn v1_export_matches_reference_schema_shape() {
        let session = calibrated_session();
        let exported = to_stereo_calibration_v1(&session).expect("fully-calibrated session should export");

        let reference: Value = serde_json::from_str(REFERENCE_JSON).expect("reference fixture must be valid JSON");

        let mut exported_tree = Vec::new();
        key_tree(&exported, "", &mut exported_tree);
        let mut reference_tree = Vec::new();
        key_tree(&reference, "", &mut reference_tree);

        assert_eq!(
            exported_tree, reference_tree,
            "exported document's key tree must match the reference stereo_calibration.v1 file's shape exactly \
             (values ignored, only object key sets at each path compared)"
        );

        assert_eq!(exported["schema"], "stereo_calibration.v1");
        assert_eq!(exported["schema"], reference["schema"]);

        for cam in ["cam0", "cam1"] {
            let k = exported["cameras"][cam]["K"].as_array().expect("K must be an array");
            assert_eq!(k.len(), 3, "K must have 3 rows");
            for row in k {
                assert_eq!(row.as_array().expect("K row must be an array").len(), 3, "K row must have 3 entries");
            }
        }

        assert_eq!(
            exported["checkerboard"]["board_size"],
            json!([INNER_ROWS, INNER_COLS]),
            "board_size must be [rows, cols], matching the reference's [6, 7] ordering"
        );
        assert_eq!(reference["checkerboard"]["board_size"], json!([6, 7]));

        let t = exported["extrinsics"]["T_cam0_to_cam1"].as_array().expect("T must be an array");
        assert_eq!(t.len(), 3, "T_cam0_to_cam1 must be a 3-vector");

        let shared_k = exported["shared_intrinsics"]["K"].as_array().expect("shared K must be an array");
        assert_eq!(shared_k.len(), 3);
        assert_eq!(exported["shared_intrinsics"]["image_size"].as_array().expect("shared image_size must be an array").len(), 2);
    }
}
