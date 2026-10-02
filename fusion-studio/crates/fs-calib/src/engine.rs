
use std::collections::VecDeque;

use opencv::calib3d;
use opencv::core::{Mat, Point2f, Point3f, Size, TermCriteria, Vector, TermCriteria_COUNT, TermCriteria_EPS};
use opencv::prelude::*;

use crate::board::{BoardConfig, Detection};
use crate::pool::{KeyframePool, Offer};
use crate::sim::{mat3x3_from, mat3x3_to, vecn_from, vecn_to};

const HISTORY_CAP: usize = 64;
const CONVERGENCE_MIN_HISTORY: usize = 5;
const CONVERGENCE_REL_DELTA: f64 = 1e-3;
const OUTLIER_MIN_PX: f64 = 1.5;
const OUTLIER_MEDIAN_MULT: f64 = 2.0;

const MIN_VIEWS_INTRINSICS: usize = 4;

const MIN_VIEWS_STEREO: usize = 4;

const STEREO_PAIR_CAP_DEFAULT: usize = 30;


#[derive(Clone, Debug)]
pub struct CamCalib {
    pub k: [[f64; 3]; 3],
    pub dist: Vec<f64>,

    pub rms: f64,

    pub per_view_rms: Vec<f64>,
}

pub struct IntrinsicsEngine {
    pool: KeyframePool,
    cfg: BoardConfig,
    image_size: (u32, u32),
    pub state: Option<CamCalib>,

    history: VecDeque<[f64; 5]>,
}

impl IntrinsicsEngine {
    pub fn new(cfg: BoardConfig, cap: usize, image_w: u32, image_h: u32) -> Self {
        IntrinsicsEngine {
            pool: KeyframePool::new(cap, image_w, image_h),
            cfg,
            image_size: (image_w, image_h),
            state: None,
            history: VecDeque::with_capacity(HISTORY_CAP),
        }
    }

    pub fn offer(&mut self, det: Detection) -> bool {
        if !self.accept(det) {
            return false;
        }
        self.recalibrate();
        true
    }


    pub(crate) fn accept(&mut self, det: Detection) -> bool {
        !matches!(self.pool.offer(det), Offer::Rejected)
    }

    pub(crate) fn snapshot_views(&self) -> Vec<(u64, Detection)> {
        self.pool.entries().iter().map(|e| (e.id(), e.detection.clone())).collect()
    }


    pub(crate) fn apply_outcome(&mut self, outcome: IntrinsicsOutcome) {
        for id in &outcome.evicted_ids {
            self.pool.remove_by_id(*id);
        }
        match outcome.state {
            Some(calib) => {
                self.push_history(&calib);
                self.state = Some(calib);
            }
            None => self.state = None,
        }
    }


    fn recalibrate(&mut self) {
        let views = self.snapshot_views();
        let outcome = intrinsics_pass(&self.cfg, &views, self.image_size, self.state.as_ref());
        self.apply_outcome(outcome);
        if let Some(calib) = &self.state {
            debug_assert_eq!(
                calib.per_view_rms.len(),
                self.pool.len(),
                "CamCalib.per_view_rms must have one entry per current pool entry after a synchronous pass"
            );
        }
    }

    fn push_history(&mut self, calib: &CamCalib) {
        if self.history.len() >= HISTORY_CAP {
            self.history.pop_front();
        }
        self.history.push_back([calib.k[0][0], calib.k[1][1], calib.k[0][2], calib.k[1][2], calib.rms]);
    }


    pub fn converged(&self) -> bool {
        if self.history.len() < CONVERGENCE_MIN_HISTORY {
            return false;
        }
        let recent: Vec<&[f64; 5]> = self.history.iter().rev().take(CONVERGENCE_MIN_HISTORY).collect();
        for param_idx in 0..4 {
            let vals: Vec<f64> = recent.iter().map(|h| h[param_idx]).collect();
            let lo = vals.iter().cloned().fold(f64::INFINITY, f64::min);
            let hi = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let denom = hi.abs().max(1e-9);
            if (hi - lo) / denom >= CONVERGENCE_REL_DELTA {
                return false;
            }
        }
        true
    }

    pub fn reset(&mut self) {
        self.pool.clear();
        self.state = None;
        self.history.clear();
    }

    pub fn coverage_mask(&self) -> [[bool; 6]; 6] {
        self.pool.coverage_mask()
    }

    pub fn pool_len(&self) -> usize {
        self.pool.len()
    }

    pub fn history(&self) -> impl Iterator<Item = &[f64; 5]> {
        self.history.iter()
    }


    pub fn image_size(&self) -> (u32, u32) {
        self.image_size
    }
}


#[derive(Clone, Debug)]
pub struct StereoCalib {
    pub r: [[f64; 3]; 3],
    pub t: [f64; 3],
    pub rms: f64,
}


pub struct StereoEngine {

    pairs: Vec<(u64, Detection, Detection)>,
    next_id: u64,
    cap: usize,
    pub state: Option<StereoCalib>,
    per_pair_rms: Vec<f64>,
}

impl StereoEngine {
    pub fn new(cap: usize) -> Self {
        StereoEngine {
            pairs: Vec::with_capacity(cap.max(1)),
            next_id: 0,
            cap: cap.max(1),
            state: None,
            per_pair_rms: Vec::new(),
        }
    }

    pub fn offer(
        &mut self,
        det0: Detection,
        det1: Detection,
        cam0: &CamCalib,
        cam1: &CamCalib,
        cfg: &BoardConfig,
        size0: (u32, u32),
        _size1: (u32, u32),
    ) -> bool {
        self.accept_pair(det0, det1);
        let outcome = stereo_pass(&self.pairs, cam0, cam1, cfg, size0);
        let ok = self.apply_outcome(outcome);
        if self.state.is_some() {
            debug_assert_eq!(self.per_pair_rms.len(), self.pairs.len(), "per-pair rms must have one entry per current pair after a synchronous pass");
        }
        ok
    }


    pub(crate) fn accept_pair(&mut self, det0: Detection, det1: Detection) {
        let id = self.next_id;
        self.next_id += 1;
        self.pairs.push((id, det0, det1));
        if self.pairs.len() > self.cap {
            self.pairs.remove(0);
        }
    }

    pub(crate) fn snapshot_pairs(&self) -> Vec<(u64, Detection, Detection)> {
        self.pairs.clone()
    }


    pub(crate) fn apply_outcome(&mut self, outcome: StereoOutcome) -> bool {
        for id in &outcome.evicted_ids {
            if let Some(idx) = self.pairs.iter().position(|(pid, _, _)| pid == id) {
                self.pairs.remove(idx);
            }
        }
        match outcome.state {
            Some(calib) => {
                self.per_pair_rms = outcome.per_pair_rms;
                self.state = Some(calib);
                true
            }
            None => {
                self.state = None;
                self.per_pair_rms.clear();
                false
            }
        }
    }

    pub fn reset(&mut self) {
        self.pairs.clear();
        self.state = None;
        self.per_pair_rms.clear();
    }

    pub fn pair_count(&self) -> usize {
        self.pairs.len()
    }
}

pub struct CalibSession {
    pub cam0: IntrinsicsEngine,
    pub cam1: IntrinsicsEngine,
    pub stereo: StereoEngine,
    pub cfg: BoardConfig,
}

impl CalibSession {
    pub fn new(cfg: BoardConfig, cap: usize, size0: (u32, u32), size1: (u32, u32)) -> Self {
        CalibSession {
            cam0: IntrinsicsEngine::new(cfg.clone(), cap, size0.0, size0.1),
            cam1: IntrinsicsEngine::new(cfg.clone(), cap, size1.0, size1.1),
            stereo: StereoEngine::new(STEREO_PAIR_CAP_DEFAULT),
            cfg,
        }
    }


    pub fn observe(&mut self, det0: Option<Detection>, det1: Option<Detection>) {
        let stereo_input = match (&det0, &det1) {
            (Some(d0), Some(d1)) => Some((d0.clone(), d1.clone())),
            _ => None,
        };

        if let Some(d0) = det0 {
            self.cam0.offer(d0);
        }
        if let Some(d1) = det1 {
            self.cam1.offer(d1);
        }

        if let Some((d0, d1)) = stereo_input {
            if let (Some(cam0), Some(cam1)) = (self.cam0.state.clone(), self.cam1.state.clone()) {
                self.stereo.offer(d0, d1, &cam0, &cam1, &self.cfg, self.cam0.image_size, self.cam1.image_size);
            }
        }
    }

    pub fn reset_stereo(&mut self) {
        self.stereo.reset();
    }


    pub fn reset_camera(&mut self, idx: usize) {
        match idx {
            0 => self.cam0.reset(),
            1 => self.cam1.reset(),
            _ => return,
        }
        self.stereo.reset();
    }

    pub fn reset_all(&mut self) {
        self.cam0.reset();
        self.cam1.reset();
        self.stereo.reset();
    }
}


pub(crate) struct IntrinsicsOutcome {

    pub(crate) state: Option<CamCalib>,
    pub(crate) evicted_ids: Vec<u64>,
}


pub(crate) fn intrinsics_pass(cfg: &BoardConfig, views: &[(u64, Detection)], image_size: (u32, u32), warm: Option<&CamCalib>) -> IntrinsicsOutcome {
    let size = Size::new(image_size.0 as i32, image_size.1 as i32);
    let Some((calib, used_ids)) = run_intrinsics_calibration(cfg, views, size, warm) else {
        return IntrinsicsOutcome { state: None, evicted_ids: Vec::new() };
    };

    if let Some(threshold) = outlier_threshold(&calib.per_view_rms) {
        debug_assert_eq!(calib.per_view_rms.len(), used_ids.len(), "per_view_rms must pair 1:1 with used_ids");
        let evicted: Vec<u64> = calib
            .per_view_rms
            .iter()
            .zip(&used_ids)
            .filter(|&(&e, _)| e > threshold)
            .map(|(_, &id)| id)
            .collect();
        if !evicted.is_empty() {
            let survivors = views.len().saturating_sub(evicted.len());
            if survivors >= MIN_VIEWS_INTRINSICS {
                let surviving: Vec<(u64, Detection)> = views.iter().filter(|(id, _)| !evicted.contains(id)).cloned().collect();
                return match run_intrinsics_calibration(cfg, &surviving, size, Some(&calib)) {
                    Some((recalib, _)) => IntrinsicsOutcome { state: Some(recalib), evicted_ids: evicted },
                    None => IntrinsicsOutcome { state: None, evicted_ids: evicted },
                };
            }
        }
    }
    IntrinsicsOutcome { state: Some(calib), evicted_ids: Vec::new() }
}


pub(crate) struct StereoOutcome {
    pub(crate) state: Option<StereoCalib>,
    pub(crate) per_pair_rms: Vec<f64>,
    pub(crate) evicted_ids: Vec<u64>,
}

pub(crate) fn stereo_pass(pairs: &[(u64, Detection, Detection)], cam0: &CamCalib, cam1: &CamCalib, cfg: &BoardConfig, size0: (u32, u32)) -> StereoOutcome {
    let Some((calib, errors, used_ids)) = run_stereo_calibration(pairs, cam0, cam1, cfg, size0) else {
        return StereoOutcome { state: None, per_pair_rms: Vec::new(), evicted_ids: Vec::new() };
    };

    if let Some(threshold) = outlier_threshold(&errors) {
        debug_assert_eq!(errors.len(), used_ids.len(), "per-pair rms must pair 1:1 with used_ids");
        let evicted: Vec<u64> = errors.iter().zip(&used_ids).filter(|&(&e, _)| e > threshold).map(|(_, &id)| id).collect();
        if !evicted.is_empty() {
            let survivors = pairs.len().saturating_sub(evicted.len());
            if survivors >= MIN_VIEWS_STEREO {
                let surviving: Vec<(u64, Detection, Detection)> = pairs.iter().filter(|(id, _, _)| !evicted.contains(id)).cloned().collect();
                return match run_stereo_calibration(&surviving, cam0, cam1, cfg, size0) {
                    Some((recalib, reerrors, _)) => StereoOutcome { state: Some(recalib), per_pair_rms: reerrors, evicted_ids: evicted },
                    None => StereoOutcome { state: None, per_pair_rms: Vec::new(), evicted_ids: evicted },
                };
            }
        }
    }
    StereoOutcome { state: Some(calib), per_pair_rms: errors, evicted_ids: Vec::new() }
}

fn object_points_for(cfg: &BoardConfig, det: &Detection) -> Vec<Point3f> {
    let full = cfg.object_points();
    match &det.ids {
        Some(ids) => ids.iter().map(|&id| full[id as usize]).collect(),
        None => full,
    }
}


fn run_intrinsics_calibration(cfg: &BoardConfig, views: &[(u64, Detection)], image_size: Size, warm: Option<&CamCalib>) -> Option<(CamCalib, Vec<u64>)> {
    if views.is_empty() {
        return None;
    }
    let warm = if std::env::var_os("FS_CALIB_WARM").is_some() { warm } else { None };
    let mut obj_pts: Vector<Vector<Point3f>> = Vector::new();
    let mut img_pts: Vector<Vector<Point2f>> = Vector::new();
    let mut used_ids: Vec<u64> = Vec::with_capacity(views.len());
    for (id, det) in views {
        let op = object_points_for(cfg, det);
        if op.len() != det.corners.len() {
            continue;
        }
        obj_pts.push(Vector::from_iter(op));
        img_pts.push(Vector::from_iter(det.corners.iter().map(|&(x, y)| Point2f::new(x, y))));
        used_ids.push(*id);
    }
    if obj_pts.is_empty() {
        return None;
    }

    let (mut camera_matrix, mut dist_coeffs, flags) = match warm {
        Some(prev) => (mat3x3_from(&prev.k).ok()?, vecn_from(&prev.dist).ok()?, calib3d::CALIB_USE_INTRINSIC_GUESS),
        None => (Mat::default(), Mat::default(), 0),
    };
    let mut rvecs: Vector<Mat> = Vector::new();
    let mut tvecs: Vector<Mat> = Vector::new();
    let mut std_dev_intrinsics = Mat::default();
    let mut std_dev_extrinsics = Mat::default();
    let mut per_view_errors = Mat::default();
    let criteria = TermCriteria::new(TermCriteria_COUNT + TermCriteria_EPS, 30, f64::EPSILON).ok()?;

    let rms = calib3d::calibrate_camera_extended(
        &obj_pts,
        &img_pts,
        image_size,
        &mut camera_matrix,
        &mut dist_coeffs,
        &mut rvecs,
        &mut tvecs,
        &mut std_dev_intrinsics,
        &mut std_dev_extrinsics,
        &mut per_view_errors,
        flags,
        criteria,
    )
    .ok()?;

    let k = mat3x3_to(&camera_matrix).ok()?;
    let dist = vecn_to(&dist_coeffs).ok()?;
    let per_view_rms = vecn_to(&per_view_errors).ok()?;

    Some((CamCalib { k, dist, rms, per_view_rms }, used_ids))
}

fn run_stereo_calibration(
    pairs: &[(u64, Detection, Detection)],
    cam0: &CamCalib,
    cam1: &CamCalib,
    cfg: &BoardConfig,
    size0: (u32, u32),
) -> Option<(StereoCalib, Vec<f64>, Vec<u64>)> {
    if pairs.is_empty() {
        return None;
    }
    let mut obj_pts: Vector<Vector<Point3f>> = Vector::new();
    let mut img_pts0: Vector<Vector<Point2f>> = Vector::new();
    let mut img_pts1: Vector<Vector<Point2f>> = Vector::new();
    let mut used_ids: Vec<u64> = Vec::with_capacity(pairs.len());
    for (id, d0, d1) in pairs {
        let op0 = object_points_for(cfg, d0);
        let op1 = object_points_for(cfg, d1);
        if op0.len() != d0.corners.len() || op1.len() != d1.corners.len() || op0.len() != op1.len() {
            continue;
        }
        obj_pts.push(Vector::from_iter(op0));
        img_pts0.push(Vector::from_iter(d0.corners.iter().map(|&(x, y)| Point2f::new(x, y))));
        img_pts1.push(Vector::from_iter(d1.corners.iter().map(|&(x, y)| Point2f::new(x, y))));
        used_ids.push(*id);
    }
    if obj_pts.is_empty() {
        return None;
    }

    let mut k1 = mat3x3_from(&cam0.k).ok()?;
    let mut d1 = vecn_from(&cam0.dist).ok()?;
    let mut k2 = mat3x3_from(&cam1.k).ok()?;
    let mut d2 = vecn_from(&cam1.dist).ok()?;
    let image_size = Size::new(size0.0 as i32, size0.1 as i32);
    let mut r = Mat::default();
    let mut t = Mat::default();
    let mut e = Mat::default();
    let mut f = Mat::default();
    let mut rvecs: Vector<Mat> = Vector::new();
    let mut tvecs: Vector<Mat> = Vector::new();
    let mut per_view_errors = Mat::default();

    let rms = calib3d::stereo_calibrate_extended_def(
        &obj_pts, &img_pts0, &img_pts1, &mut k1, &mut d1, &mut k2, &mut d2, image_size, &mut r, &mut t, &mut e, &mut f, &mut rvecs, &mut tvecs,
        &mut per_view_errors,
    )
    .ok()?;

    let r_out = mat3x3_to(&r).ok()?;
    let t_out = vecn_to(&t).ok()?;
    if t_out.len() < 3 {
        return None;
    }
    let t_arr = [t_out[0], t_out[1], t_out[2]];

    let errors = per_pair_rms_from(&per_view_errors).ok()?;

    Some((StereoCalib { r: r_out, t: t_arr, rms }, errors, used_ids))
}

fn per_pair_rms_from(m: &Mat) -> opencv::Result<Vec<f64>> {
    let rows = m.rows();
    let cols = m.cols();
    let mut out = Vec::with_capacity(rows.max(0) as usize);
    for r in 0..rows {
        if cols >= 2 {
            let e0 = *m.at_2d::<f64>(r, 0)?;
            let e1 = *m.at_2d::<f64>(r, 1)?;
            out.push(((e0 * e0 + e1 * e1) / 2.0).sqrt());
        } else {
            out.push(*m.at_2d::<f64>(r, 0)?);
        }
    }
    Ok(out)
}

fn outlier_threshold(errors: &[f64]) -> Option<f64> {
    if errors.is_empty() {
        return None;
    }
    let median = median_of(errors);
    Some((OUTLIER_MEDIAN_MULT * median).max(OUTLIER_MIN_PX))
}

fn median_of(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let n = sorted.len();
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::BoardKind;
    use crate::sim::SimRig;

    const INNER_COLS: u32 = 7;
    const INNER_ROWS: u32 = 6;
    const RECT_W_MM: f64 = 82.0;
    const RECT_H_MM: f64 = 98.0;

    fn test_board() -> BoardConfig {
        BoardConfig {
            kind: BoardKind::Checkerboard,
            inner_cols: INNER_COLS,
            inner_rows: INNER_ROWS,
            rect_w_mm: RECT_W_MM,
            rect_h_mm: RECT_H_MM,
        }
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

    #[test]
    fn intrinsics_recover_ground_truth() {
        let board = test_board();
        let rig = SimRig::realistic();
        let views = rig.gen_views(&board, 30, 1);
        assert_eq!(views.len(), 30, "sim rig should produce the requested number of valid views");

        let mut engine = IntrinsicsEngine::new(board.clone(), 40, rig.size0.0, rig.size0.1);
        for (d0, _d1) in views {
            engine.offer(d0);
        }
        let pool_len = engine.pool_len();
        let calib = engine.state.expect("engine should have calibrated");

        let fx_truth = rig.k0[0][0];
        let fy_truth = rig.k0[1][1];
        let cx_truth = rig.k0[0][2];
        let cy_truth = rig.k0[1][2];

        println!(
            "[intrinsics_recover_ground_truth] pool_len={} rms={:.4} fx={:.3} (truth {:.3}) fy={:.3} (truth {:.3}) cx={:.3} (truth {:.3}) cy={:.3} (truth {:.3}) k1={:.5} (truth {:.5}) k2={:.5} (truth {:.5})",
            pool_len,
            calib.rms,
            calib.k[0][0], fx_truth,
            calib.k[1][1], fy_truth,
            calib.k[0][2], cx_truth,
            calib.k[1][2], cy_truth,
            calib.dist[0], rig.dist0[0],
            calib.dist[1], rig.dist0[1],
        );

        assert!((calib.k[0][0] - fx_truth).abs() / fx_truth < 0.01, "fx off: {} vs {}", calib.k[0][0], fx_truth);
        assert!((calib.k[1][1] - fy_truth).abs() / fy_truth < 0.01, "fy off: {} vs {}", calib.k[1][1], fy_truth);
        assert!((calib.k[0][2] - cx_truth).abs() / cx_truth < 0.01, "cx off: {} vs {}", calib.k[0][2], cx_truth);
        assert!((calib.k[1][2] - cy_truth).abs() / cy_truth < 0.01, "cy off: {} vs {}", calib.k[1][2], cy_truth);

        assert!((calib.dist[0] - rig.dist0[0]).abs() < 0.02, "k1 off: {} vs {}", calib.dist[0], rig.dist0[0]);
        assert!((calib.dist[1] - rig.dist0[1]).abs() < 0.02, "k2 off: {} vs {}", calib.dist[1], rig.dist0[1]);
    }

    #[test]
    fn stereo_recovers_extrinsics() {
        let board = test_board();
        let rig = SimRig::realistic();
        let views = rig.gen_views(&board, 30, 2);
        assert_eq!(views.len(), 30);

        let mut session = CalibSession::new(board.clone(), 40, rig.size0, rig.size1);
        for (d0, d1) in views {
            session.observe(Some(d0), Some(d1));
        }

        let stereo = session.stereo.state.clone().expect("stereo should have calibrated");

        let angle_deg = rotation_angle_deg(&rig.r, &stereo.r);
        let dot = rig.t[0] * stereo.t[0] + rig.t[1] * stereo.t[1] + rig.t[2] * stereo.t[2];
        let norm_true = vec3_norm(rig.t);
        let norm_est = vec3_norm(stereo.t);
        let cos_dir = (dot / (norm_true * norm_est)).clamp(-1.0, 1.0);
        let dir_angle_deg = cos_dir.acos().to_degrees();
        let baseline_err_pct = (norm_est - norm_true).abs() / norm_true * 100.0;

        println!(
            "[stereo_recovers_extrinsics] rms={:.4} rotation_err={:.4}deg t_dir_err={:.4}deg |t|_est={:.3}mm |t|_truth={:.3}mm baseline_err={:.3}%",
            stereo.rms, angle_deg, dir_angle_deg, norm_est, norm_true, baseline_err_pct
        );

        assert!(angle_deg < 0.2, "rotation angle error too large: {angle_deg} deg");
        assert!(dir_angle_deg < 0.5, "translation direction error too large: {dir_angle_deg} deg");
        assert!((norm_est - norm_true).abs() / norm_true < 0.01, "baseline length off: {norm_est} vs {norm_true}");
    }

    #[test]
    fn outliers_rejected() {
        let board = test_board();
        let rig = SimRig::realistic();
        let mut views = rig.gen_views(&board, 30, 3);
        assert_eq!(views.len(), 30);

        let corrupt_idx = [4usize, 14, 24];
        let mut state: u64 = 0xC0FFEE ^ 3;
        let next = |state: &mut u64| -> f64 {
            *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (*state >> 11) as f64 / (1u64 << 53) as f64
        };
        for &idx in &corrupt_idx {
            let (d0, d1) = &mut views[idx];
            for corner in d0.corners.iter_mut().chain(d1.corners.iter_mut()) {
                let nx = next(&mut state) * 10.0 - 5.0;
                let ny = next(&mut state) * 10.0 - 5.0;
                corner.0 += nx as f32;
                corner.1 += ny as f32;
            }
        }

        let mut session = CalibSession::new(board.clone(), 40, rig.size0, rig.size1);
        for (d0, d1) in views {
            session.observe(Some(d0), Some(d1));
        }

        let cam0_pool_len = session.cam0.pool_len();
        let cam1_pool_len = session.cam1.pool_len();
        let cam0_calib = session.cam0.state.expect("cam0 should have calibrated");
        let cam1_calib = session.cam1.state.expect("cam1 should have calibrated");

        println!(
            "[outliers_rejected] cam0: pool_len={} rms={:.4} | cam1: pool_len={} rms={:.4} (30 offered, 3 corrupted w/ ~5px noise)",
            cam0_pool_len, cam0_calib.rms, cam1_pool_len, cam1_calib.rms
        );

        assert!(cam0_calib.rms < 0.3, "cam0 rms too high, outliers not purged: {}", cam0_calib.rms);
        assert!(cam1_calib.rms < 0.3, "cam1 rms too high, outliers not purged: {}", cam1_calib.rms);
    }

    #[test]
    fn reset_semantics() {
        let board = test_board();
        let rig = SimRig::realistic();

        let mut session = CalibSession::new(board.clone(), 40, rig.size0, rig.size1);
        for (d0, d1) in rig.gen_views(&board, 30, 4) {
            session.observe(Some(d0), Some(d1));
        }
        assert!(session.cam0.state.is_some());
        assert!(session.cam1.state.is_some());
        assert!(session.stereo.state.is_some());

        session.reset_stereo();
        assert!(session.stereo.state.is_none(), "reset_stereo should clear stereo");
        assert!(session.cam0.state.is_some(), "reset_stereo must not touch cam0 intrinsics");
        assert!(session.cam1.state.is_some(), "reset_stereo must not touch cam1 intrinsics");

        for (d0, d1) in rig.gen_views(&board, 30, 4) {
            session.observe(Some(d0), Some(d1));
        }
        assert!(session.stereo.state.is_some(), "stereo should re-calibrate after reset_stereo");

        session.reset_camera(0);
        assert!(session.cam0.state.is_none(), "reset_camera(0) should clear cam0");
        assert!(session.stereo.state.is_none(), "reset_camera(0) should cascade to stereo");
        assert!(session.cam1.state.is_some(), "reset_camera(0) must not touch cam1");

        for (d0, d1) in rig.gen_views(&board, 30, 4) {
            session.observe(Some(d0), Some(d1));
        }
        assert!(session.cam0.state.is_some());
        assert!(session.stereo.state.is_some());

        session.reset_all();
        assert!(session.cam0.state.is_none());
        assert!(session.cam1.state.is_none());
        assert!(session.stereo.state.is_none());
    }

    #[test]
    fn convergence_flag() {
        let board = test_board();
        let rig = SimRig::realistic();
        let views = rig.gen_views(&board, 30, 5);
        assert!(views.len() >= 10, "need enough views to reach convergence");

        let mut session = CalibSession::new(board.clone(), 40, rig.size0, rig.size1);
        for (d0, d1) in views {
            session.observe(Some(d0), Some(d1));
        }

        println!(
            "[convergence_flag] cam0: pool_len={} converged={} | cam1: pool_len={} converged={}",
            session.cam0.pool_len(),
            session.cam0.converged(),
            session.cam1.pool_len(),
            session.cam1.converged()
        );

        assert!(session.cam0.converged(), "cam0 should have converged after {} pool entries", session.cam0.pool_len());
        assert!(session.cam1.converged(), "cam1 should have converged after {} pool entries", session.cam1.pool_len());
    }

    #[test]
    fn near_minimum_pool_len_invariant() {
        let board = test_board();
        let rig = SimRig::realistic();
        let mut views = rig.gen_views(&board, 5, 6);
        assert_eq!(views.len(), 5, "need exactly 5 views for this near-minimum-pool scenario");

        let corrupt_idx = [1usize, 3];
        let mut state: u64 = 0xBADC0DE ^ 6;
        let next = |state: &mut u64| -> f64 {
            *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (*state >> 11) as f64 / (1u64 << 53) as f64
        };
        for &idx in &corrupt_idx {
            for corner in views[idx].0.corners.iter_mut() {
                let nx = next(&mut state) * 10.0 - 5.0;
                let ny = next(&mut state) * 10.0 - 5.0;
                corner.0 += nx as f32;
                corner.1 += ny as f32;
            }
        }

        let mut engine = IntrinsicsEngine::new(board.clone(), 40, rig.size0.0, rig.size0.1);
        for (d0, _d1) in views {
            engine.offer(d0);
            match &engine.state {
                Some(calib) => assert_eq!(
                    calib.per_view_rms.len(),
                    engine.pool_len(),
                    "Some(state) must have per_view_rms lined up 1:1 with the current pool"
                ),
                None => assert!(engine.pool_len() <= 40, "pool must stay within cap even when state is None"),
            }
        }
    }
}
