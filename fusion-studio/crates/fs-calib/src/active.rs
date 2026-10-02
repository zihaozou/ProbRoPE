use serde_json::Value;


#[derive(Clone, Debug, PartialEq)]
pub struct CamIntrinsics {
    pub k: [[f64; 3]; 3],
    pub dist: Vec<f64>,
    pub image_size: (u32, u32),
    pub rms: f64,
}


#[derive(Clone, Debug, PartialEq)]
pub enum CalibSource {
    Loaded(String),
    LiveSession,
}

#[derive(Clone, Debug)]
pub struct ActiveCalibration {
    pub cam0: CamIntrinsics,
    pub cam1: CamIntrinsics,

    pub r: [[f64; 3]; 3],
    pub t: [f64; 3],
    pub stereo_rms: f64,
    pub source: CalibSource,

    pub document: Value,
}


pub fn load_stereo_calibration_v1(document: Value, source: CalibSource) -> Result<ActiveCalibration, String> {
    let schema = document["schema"].as_str().unwrap_or("");
    if schema != "stereo_calibration.v1" {
        return Err(format!("不认识的 schema:{schema:?},需要 stereo_calibration.v1"));
    }

    let cam = |label: &str| -> Result<CamIntrinsics, String> {
        let b = &document["cameras"][label];
        if b.is_null() {
            return Err(format!("缺少 cameras.{label}"));
        }
        Ok(CamIntrinsics {
            k: mat3(&b["K"]).ok_or_else(|| format!("{label}.K 不是 3x3 数值矩阵"))?,
            dist: b["dist"]
                .as_array()
                .ok_or_else(|| format!("{label}.dist 缺失"))?
                .iter()
                .map(|v| v.as_f64().ok_or_else(|| format!("{label}.dist 含非数值")))
                .collect::<Result<_, _>>()?,
            image_size: size2(&b["image_size"]).ok_or_else(|| format!("{label}.image_size 缺失或非法"))?,
            rms: b["reprojection_error_px"].as_f64().ok_or_else(|| format!("{label}.reprojection_error_px 缺失"))?,
        })
    };

    let ex = &document["extrinsics"];
    Ok(ActiveCalibration {
        cam0: cam("cam0")?,
        cam1: cam("cam1")?,
        r: mat3(&ex["R_cam0_to_cam1"]).ok_or("extrinsics.R_cam0_to_cam1 缺失或非法")?,
        t: vec3(&ex["T_cam0_to_cam1"]).ok_or("extrinsics.T_cam0_to_cam1 缺失或非法")?,
        stereo_rms: ex["stereo_reprojection_error_px"].as_f64().ok_or("extrinsics.stereo_reprojection_error_px 缺失")?,
        source,
        document,
    })
}

impl ActiveCalibration {
    pub fn validate_against(&self, cam0: (u32, u32), cam1: (u32, u32)) -> Result<(), String> {
        for (label, have, want) in [("cam0", self.cam0.image_size, cam0), ("cam1", self.cam1.image_size, cam1)] {
            if have != want {
                return Err(format!("{label} 尺寸不符:标定文件 {}x{},当前源 {}x{}", have.0, have.1, want.0, want.1));
            }
        }
        Ok(())
    }
}


fn mat3(v: &Value) -> Option<[[f64; 3]; 3]> {
    let rows = v.as_array()?;
    if rows.len() != 3 {
        return None;
    }
    let mut m = [[0.0; 3]; 3];
    for (i, row) in rows.iter().enumerate() {
        let row = row.as_array()?;
        if row.len() != 3 {
            return None;
        }
        for (j, x) in row.iter().enumerate() {
            m[i][j] = x.as_f64()?;
        }
    }
    Some(m)
}

fn vec3(v: &Value) -> Option<[f64; 3]> {
    let a = v.as_array()?;
    if a.len() != 3 {
        return None;
    }
    Some([a[0].as_f64()?, a[1].as_f64()?, a[2].as_f64()?])
}

fn size2(v: &Value) -> Option<(u32, u32)> {
    let a = v.as_array()?;
    if a.len() != 2 {
        return None;
    }
    let w = u32::try_from(a[0].as_u64()?).ok()?;
    let h = u32::try_from(a[1].as_u64()?).ok()?;
    Some((w, h))
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::{BoardConfig, BoardKind};
    use crate::engine::CalibSession;
    use crate::export::to_stereo_calibration_v1;
    use crate::sim::SimRig;
    use serde_json::json;

    fn test_board() -> BoardConfig {
        BoardConfig {
            kind: BoardKind::Checkerboard,
            inner_cols: 7,
            inner_rows: 6,
            rect_w_mm: 82.0,
            rect_h_mm: 98.0,
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

    fn minimal_valid_doc() -> Value {
        let cam = |w: u32, h: u32| {
            json!({
                "K": [[1000.0, 0.0, w as f64 / 2.0], [0.0, 1000.0, h as f64 / 2.0], [0.0, 0.0, 1.0]],
                "dist": [-0.1, 0.05, 0.0, 0.0, 0.0],
                "image_size": [w, h],
                "reprojection_error_px": 0.25,
            })
        };
        json!({
            "schema": "stereo_calibration.v1",
            "cameras": { "cam0": cam(1280, 1024), "cam1": cam(1280, 720) },
            "extrinsics": {
                "R_cam0_to_cam1": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                "T_cam0_to_cam1": [100.0, 0.0, 0.0],
                "stereo_reprojection_error_px": 0.4,
            },
        })
    }

    #[test]
    fn load_round_trips_a_real_session_export() {
        let session = calibrated_session();
        let doc = to_stereo_calibration_v1(&session).expect("fully-calibrated session should export");

        let a = load_stereo_calibration_v1(doc.clone(), CalibSource::LiveSession).expect("real export must load");

        assert_eq!(a.document, doc, "document must be kept identical to the input");
        assert_eq!(a.source, CalibSource::LiveSession);

        let cam0 = session.cam0.state.as_ref().unwrap();
        let cam1 = session.cam1.state.as_ref().unwrap();
        assert_eq!(a.cam0.k, cam0.k);
        assert_eq!(a.cam0.dist, cam0.dist);
        assert_eq!(a.cam0.rms, cam0.rms);
        assert_eq!(a.cam0.image_size, session.cam0.image_size());
        assert_eq!(a.cam1.k, cam1.k);
        assert_eq!(a.cam1.dist, cam1.dist);
        assert_eq!(a.cam1.image_size, session.cam1.image_size());

        let stereo = session.stereo.state.as_ref().unwrap();
        assert_eq!(a.r, stereo.r);
        assert_eq!(a.t, stereo.t);
        assert_eq!(a.stereo_rms, stereo.rms);
    }


    #[test]
    fn parsing_from_text_keeps_float_literals_bitwise() {
        for lit in [
            "1078.9618310638023",
            "0.9955872699573983",
            "0.09534844607881249",
            "933.0590096796435",
            "-0.40389234483174036",
        ] {
            let v: Value = serde_json::from_str(lit).expect("literal parses");
            assert_eq!(
                serde_json::to_string(&v).unwrap(),
                lit,
                "f64 文本必须往返逐位不变(float_roundtrip feature 被移除了?)"
            );
        }
    }

    #[test]
    fn load_rejects_wrong_schema_and_missing_fields() {
        let err = load_stereo_calibration_v1(json!({"schema": "nonsense.v9"}), CalibSource::LiveSession)
            .expect_err("wrong schema must be rejected");
        assert!(err.contains("schema"), "error should name the schema field: {err}");

        let err = load_stereo_calibration_v1(json!({"schema": "stereo_calibration.v1"}), CalibSource::LiveSession)
            .expect_err("document without cameras must be rejected");
        assert!(err.contains("cameras"), "error should name the missing block: {err}");

        load_stereo_calibration_v1(minimal_valid_doc(), CalibSource::LiveSession).expect("minimal doc must load");

        let mut doc = minimal_valid_doc();
        doc["cameras"]["cam0"]["image_size"] = json!([-1280, 1024]);
        let err = load_stereo_calibration_v1(doc, CalibSource::LiveSession).expect_err("negative image_size");
        assert!(err.contains("image_size"), "{err}");

        let mut doc = minimal_valid_doc();
        doc["cameras"]["cam0"]["K"] = json!([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]);
        let err = load_stereo_calibration_v1(doc, CalibSource::LiveSession).expect_err("2-wide K row");
        assert!(err.contains("K"), "{err}");

        let mut doc = minimal_valid_doc();
        doc["cameras"]["cam1"]["K"] = json!("not a matrix");
        let err = load_stereo_calibration_v1(doc, CalibSource::LiveSession).expect_err("non-array K");
        assert!(err.contains("K"), "{err}");

        let mut doc = minimal_valid_doc();
        doc["cameras"]["cam0"]["dist"] = json!([0.1, "oops"]);
        let err = load_stereo_calibration_v1(doc, CalibSource::LiveSession).expect_err("non-numeric dist entry");
        assert!(err.contains("dist"), "{err}");

        let mut doc = minimal_valid_doc();
        doc["extrinsics"]["T_cam0_to_cam1"] = json!([1.0, 2.0]);
        let err = load_stereo_calibration_v1(doc, CalibSource::LiveSession).expect_err("2-vector T");
        assert!(err.contains("T_cam0_to_cam1"), "{err}");

        let mut doc = minimal_valid_doc();
        doc.as_object_mut().unwrap().remove("extrinsics");
        let err = load_stereo_calibration_v1(doc, CalibSource::LiveSession).expect_err("missing extrinsics");
        assert!(err.contains("R_cam0_to_cam1"), "{err}");
    }

    #[test]
    fn validate_against_checks_both_sizes() {
        let a = load_stereo_calibration_v1(minimal_valid_doc(), CalibSource::LiveSession).expect("minimal doc must load");

        a.validate_against((1280, 1024), (1280, 720)).expect("matching sizes must validate");

        let err = a.validate_against((640, 480), (1280, 720)).expect_err("mismatched cam0 must be rejected");
        assert!(err.contains("cam0"), "error should name the camera: {err}");
        assert!(err.contains("1280x1024"), "error should carry the calibration's size: {err}");
        assert!(err.contains("640x480"), "error should carry the source's size: {err}");

        let err = a.validate_against((1280, 1024), (640, 480)).expect_err("mismatched cam1 must be rejected");
        assert!(err.contains("cam1"), "error should name the camera: {err}");
    }
}
