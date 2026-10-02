
use opencv::calib3d;
use opencv::core::{Mat, Point2f, Point3f, Vector};
use opencv::prelude::*;

use crate::board::{BoardConfig, Detection};

pub struct SimRig {
    pub k0: [[f64; 3]; 3],
    pub dist0: Vec<f64>,
    pub k1: [[f64; 3]; 3],
    pub dist1: Vec<f64>,

    pub r: [[f64; 3]; 3],
    pub t: [f64; 3],
    pub size0: (u32, u32),
    pub size1: (u32, u32),
}

impl SimRig {

    pub fn realistic() -> Self {
        let size0 = (1280u32, 1024u32);
        let size1 = (1280u32, 720u32);
        let k0 = [
            [1079.0, 0.0, size0.0 as f64 / 2.0],
            [0.0, 1079.0, size0.1 as f64 / 2.0],
            [0.0, 0.0, 1.0],
        ];
        let k1 = [
            [982.0, 0.0, size1.0 as f64 / 2.0],
            [0.0, 982.0, size1.1 as f64 / 2.0],
            [0.0, 0.0, 1.0],
        ];
        let dist0 = vec![-0.1, 0.05, 0.0, 0.0, 0.0];
        let dist1 = vec![-0.1, 0.05, 0.0, 0.0, 0.0];

        let theta = 3.0f64.to_radians();
        let (s, c) = theta.sin_cos();
        let r = [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]];
        let t = [100.0, 0.0, 0.0];

        SimRig { k0, dist0, k1, dist1, r, t, size0, size1 }
    }


    pub fn gen_views(&self, board: &BoardConfig, n: usize, seed: u64) -> Vec<(Detection, Detection)> {
        let obj_pts_cv: Vector<Point3f> = Vector::from_iter(board.object_points());

        let span_x = (board.inner_cols.max(1) - 1) as f64 * board.rect_w_mm;
        let span_y = (board.inner_rows.max(1) - 1) as f64 * board.rect_h_mm;
        let center_obj = [span_x / 2.0, span_y / 2.0, 0.0];

        let mut rng = Lcg::new(seed);
        let mut out = Vec::with_capacity(n);
        let max_attempts = (n * 60).max(400);

        for _ in 0..max_attempts {
            if out.len() >= n {
                break;
            }
            let depth = rng.range(600.0, 2600.0);
            let off_x = rng.range(-300.0, 300.0);
            let off_y = rng.range(-220.0, 220.0);
            let tilt_x = rng.range(-25.0, 25.0).to_radians();
            let tilt_y = rng.range(-25.0, 25.0).to_radians();
            let tilt_z = rng.range(-20.0, 20.0).to_radians();

            let r0 = mat_mul(&mat_mul(&rot_z(tilt_z), &rot_y(tilt_y)), &rot_x(tilt_x));
            let t0_center = [off_x, off_y, depth];
            let t0 = vec_sub(&t0_center, &mat_vec(&r0, &center_obj));

            let Some((corners0, area0)) = self.project_checked(&obj_pts_cv, &r0, &t0, &self.k0, &self.dist0, self.size0) else {
                continue;
            };

            let r1 = mat_mul(&self.r, &r0);
            let t1 = vec_add(&mat_vec(&self.r, &t0), &self.t);
            let Some((corners1, area1)) = self.project_checked(&obj_pts_cv, &r1, &t1, &self.k1, &self.dist1, self.size1) else {
                continue;
            };

            out.push((
                Detection { corners: corners0, ids: None, board_area_frac: area0 },
                Detection { corners: corners1, ids: None, board_area_frac: area1 },
            ));
        }
        out
    }


    fn project_checked(
        &self,
        obj_pts: &Vector<Point3f>,
        r: &[[f64; 3]; 3],
        t: &[f64; 3],
        k: &[[f64; 3]; 3],
        dist: &[f64],
        size: (u32, u32),
    ) -> Option<(Vec<(f32, f32)>, f32)> {
        const MARGIN: f32 = 6.0;
        const MIN_AREA_FRAC: f32 = 0.02;

        let corners = project(obj_pts, r, t, k, dist).ok()?;
        let (w, h) = (size.0 as f32, size.1 as f32);
        if corners.iter().any(|&(x, y)| x < MARGIN || y < MARGIN || x > w - MARGIN || y > h - MARGIN) {
            return None;
        }

        let (mut min_x, mut min_y, mut max_x, mut max_y) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
        for &(x, y) in &corners {
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
        }
        let area_frac = ((max_x - min_x) * (max_y - min_y) / (w * h)).max(0.0);
        if area_frac < MIN_AREA_FRAC {
            return None;
        }
        Some((corners, area_frac))
    }
}


fn project(obj_pts: &Vector<Point3f>, r: &[[f64; 3]; 3], t: &[f64; 3], k: &[[f64; 3]; 3], dist: &[f64]) -> opencv::Result<Vec<(f32, f32)>> {
    let r_mat = mat3x3_from(r)?;
    let mut rvec = Mat::default();
    calib3d::rodrigues_def(&r_mat, &mut rvec)?;
    let tvec = vecn_from(t)?;
    let k_mat = mat3x3_from(k)?;
    let dist_mat = vecn_from(dist)?;
    let mut img_pts: Vector<Point2f> = Vector::new();
    calib3d::project_points_def(obj_pts, &rvec, &tvec, &k_mat, &dist_mat, &mut img_pts)?;
    Ok(img_pts.as_slice().iter().map(|p| (p.x, p.y)).collect())
}


fn rot_x(a: f64) -> [[f64; 3]; 3] {
    let (s, c) = a.sin_cos();
    [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]
}

fn rot_y(a: f64) -> [[f64; 3]; 3] {
    let (s, c) = a.sin_cos();
    [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]
}

fn rot_z(a: f64) -> [[f64; 3]; 3] {
    let (s, c) = a.sin_cos();
    [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
}

fn mat_mul(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    out
}

fn mat_vec(a: &[[f64; 3]; 3], v: &[f64; 3]) -> [f64; 3] {
    [
        a[0][0] * v[0] + a[0][1] * v[1] + a[0][2] * v[2],
        a[1][0] * v[0] + a[1][1] * v[1] + a[1][2] * v[2],
        a[2][0] * v[0] + a[2][1] * v[1] + a[2][2] * v[2],
    ]
}

fn vec_add(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn vec_sub(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}


struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed ^ 0x9E3779B97F4A7C15)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0
    }


    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + self.next_f64() * (hi - lo)
    }
}


pub(crate) fn mat3x3_from(m: &[[f64; 3]; 3]) -> opencv::Result<Mat> {
    let mut mat = Mat::zeros(3, 3, opencv::core::CV_64FC1)?.to_mat()?;
    for r in 0..3 {
        for c in 0..3 {
            *mat.at_2d_mut::<f64>(r as i32, c as i32)? = m[r][c];
        }
    }
    Ok(mat)
}

pub(crate) fn mat3x3_to(mat: &Mat) -> opencv::Result<[[f64; 3]; 3]> {
    let mut m = [[0.0; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            m[r][c] = *mat.at_2d::<f64>(r as i32, c as i32)?;
        }
    }
    Ok(m)
}

pub(crate) fn vecn_from(v: &[f64]) -> opencv::Result<Mat> {
    let mut mat = Mat::zeros(v.len() as i32, 1, opencv::core::CV_64FC1)?.to_mat()?;
    for (i, val) in v.iter().enumerate() {
        *mat.at_2d_mut::<f64>(i as i32, 0)? = *val;
    }
    Ok(mat)
}

pub(crate) fn vecn_to(mat: &Mat) -> opencv::Result<Vec<f64>> {
    let n = (mat.rows() * mat.cols()) as usize;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(*mat.at::<f64>(i as i32)?);
    }
    Ok(out)
}
