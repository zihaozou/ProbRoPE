use opencv::calib3d;
use opencv::core::{Mat, Point2f, Point3f, Size, TermCriteria, TermCriteria_EPS, TermCriteria_MAX_ITER, ToInputArray, Vector};
use opencv::imgproc;
use opencv::objdetect::{self, PredefinedDictionaryType};
use opencv::prelude::*;
use serde::{Deserialize, Serialize};


#[derive(Clone, Serialize, Deserialize)]
pub enum BoardKind {
    Checkerboard,
    Charuco { marker_mm: f64, dict: String },
}

#[derive(Clone, Serialize, Deserialize)]
pub struct BoardConfig {
    pub kind: BoardKind,

    pub inner_cols: u32,

    pub inner_rows: u32,

    pub rect_w_mm: f64,

    pub rect_h_mm: f64,
}

impl BoardConfig {
    pub fn object_points(&self) -> Vec<Point3f> {
        let cols = self.inner_cols as i64;
        let rows = self.inner_rows as i64;
        let mut pts = Vec::with_capacity((rows * cols).max(0) as usize);
        for c_from_right in 0..cols {
            for r_from_top in 0..rows {
                let x = (cols - 1 - c_from_right) as f64 * self.rect_w_mm;
                let y = (rows - 1 - r_from_top) as f64 * self.rect_h_mm;
                pts.push(Point3f::new(x as f32, y as f32, 0.0));
            }
        }
        pts
    }
}


#[derive(Clone, Debug)]
pub struct Detection {
    pub corners: Vec<(f32, f32)>,

    pub ids: Option<Vec<i32>>,
    pub board_area_frac: f32,
}


const CHARUCO_MIN_FRACTION: f64 = 0.6;

pub fn detect(cfg: &BoardConfig, gray: &[u8], w: u32, h: u32) -> Option<Detection> {
    detect_opts(cfg, gray, w, h, DetectOpts { speed: DetectSpeed::Full, skip_prefilter: true }).detection
}


#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DetectSpeed {
    Full,
    Pyramid,
}

impl Default for DetectSpeed {
    fn default() -> Self {
        DetectSpeed::Pyramid
    }
}

#[derive(Clone, Copy, Debug)]
pub struct DetectOpts {
    pub speed: DetectSpeed,

    pub skip_prefilter: bool,
}

impl Default for DetectOpts {
    fn default() -> Self {
        DetectOpts {
            speed: DetectSpeed::default(),
            skip_prefilter: false,
        }
    }
}


#[derive(Clone, Debug, Default)]
pub struct DetectOutcome {
    pub detection: Option<Detection>,
    pub prefiltered: bool,
}

pub fn detect_opts(cfg: &BoardConfig, gray: &[u8], w: u32, h: u32, opts: DetectOpts) -> DetectOutcome {
    if gray.len() < (w as usize) * (h as usize) {
        return DetectOutcome::default();
    }
    let Ok(mat) = Mat::new_rows_cols_with_data(h as i32, w as i32, gray) else {
        return DetectOutcome::default();
    };
    match &cfg.kind {
        BoardKind::Checkerboard => detect_checkerboard_opts(cfg, &mat, w, h, opts),
        BoardKind::Charuco { marker_mm, dict } => DetectOutcome {
            detection: detect_charuco(cfg, &mat, w, h, *marker_mm, dict),
            prefiltered: false,
        },
    }
}


const FAST_CHECK_FLAGS: i32 = calib3d::CALIB_CB_ADAPTIVE_THRESH | calib3d::CALIB_CB_FAST_CHECK;


const PYRAMID_SCALE: f64 = 0.5;


const SUBPIX_WINDOW: i32 = 11;

fn detect_checkerboard_opts(cfg: &BoardConfig, mat: &impl ToInputArray, w: u32, h: u32, opts: DetectOpts) -> DetectOutcome {
    let pattern_size = Size::new(cfg.inner_cols as i32, cfg.inner_rows as i32);
    let n_expected = (cfg.inner_cols as usize) * (cfg.inner_rows as usize);
    if n_expected == 0 {
        return DetectOutcome::default();
    }

    if !opts.skip_prefilter {
        let mut fc_corners: Vector<Point2f> = Vector::new();
        let fc_found = calib3d::find_chessboard_corners(mat, pattern_size, &mut fc_corners, FAST_CHECK_FLAGS).unwrap_or(true);
        if !fc_found {
            return DetectOutcome {
                detection: None,
                prefiltered: true,
            };
        }
    }

    let detection = match opts.speed {
        DetectSpeed::Full => detect_checkerboard_full(cfg, mat, pattern_size, n_expected, w, h),
        DetectSpeed::Pyramid => detect_checkerboard_pyramid(cfg, mat, pattern_size, n_expected, w, h),
    };
    DetectOutcome {
        detection,
        prefiltered: false,
    }
}

fn detect_checkerboard_full(cfg: &BoardConfig, mat: &impl ToInputArray, pattern_size: Size, n_expected: usize, w: u32, h: u32) -> Option<Detection> {
    let mut corners_mat: Vector<Point2f> = Vector::new();

    let found_sb = calib3d::find_chessboard_corners_sb(mat, pattern_size, &mut corners_mat, 0).unwrap_or(false);

    if !found_sb {
        corners_mat.clear();
        let ok = calib3d::find_chessboard_corners_def(mat, pattern_size, &mut corners_mat).unwrap_or(false);
        if !ok {
            return None;
        }
        let criteria = TermCriteria {
            typ: TermCriteria_MAX_ITER + TermCriteria_EPS,
            max_count: 30,
            epsilon: 1e-6,
        };
        imgproc::corner_sub_pix(mat, &mut corners_mat, Size::new(SUBPIX_WINDOW, SUBPIX_WINDOW), Size::new(-1, -1), criteria).ok()?;
    }

    if corners_mat.len() != n_expected {
        return None;
    }

    let flat: Vec<(f32, f32)> = corners_mat.as_slice().iter().map(|p| (p.x, p.y)).collect();
    let canon = canonicalize_checkerboard(&flat, cfg.inner_rows as usize, cfg.inner_cols as usize);
    let area_frac = (convex_hull_area(&canon) / (w as f64 * h as f64)) as f32;

    Some(Detection {
        corners: canon,
        ids: None,
        board_area_frac: area_frac,
    })
}


fn detect_checkerboard_pyramid(cfg: &BoardConfig, mat: &impl ToInputArray, pattern_size: Size, n_expected: usize, w: u32, h: u32) -> Option<Detection> {
    let small_w = ((w as f64) * PYRAMID_SCALE).round().max(1.0) as i32;
    let small_h = ((h as f64) * PYRAMID_SCALE).round().max(1.0) as i32;
    let mut small = Mat::default();
    imgproc::resize(mat, &mut small, Size::new(small_w, small_h), 0.0, 0.0, imgproc::INTER_AREA).ok()?;

    let mut corners_mat: Vector<Point2f> = Vector::new();
    let found_sb = calib3d::find_chessboard_corners_sb(&small, pattern_size, &mut corners_mat, 0).unwrap_or(false);
    if !found_sb || corners_mat.len() != n_expected {
        return None;
    }

    let scale_x = small_w as f32 / w as f32;
    let scale_y = small_h as f32 / h as f32;
    let mut refined: Vector<Point2f> = corners_mat.iter().map(|p| Point2f::new(p.x / scale_x, p.y / scale_y)).collect();

    let criteria = TermCriteria {
        typ: TermCriteria_MAX_ITER + TermCriteria_EPS,
        max_count: 30,
        epsilon: 1e-6,
    };
    imgproc::corner_sub_pix(mat, &mut refined, Size::new(SUBPIX_WINDOW, SUBPIX_WINDOW), Size::new(-1, -1), criteria).ok()?;

    if refined.len() != n_expected {
        return None;
    }

    let flat: Vec<(f32, f32)> = refined.as_slice().iter().map(|p| (p.x, p.y)).collect();
    let canon = canonicalize_checkerboard(&flat, cfg.inner_rows as usize, cfg.inner_cols as usize);
    let area_frac = (convex_hull_area(&canon) / (w as f64 * h as f64)) as f32;

    Some(Detection {
        corners: canon,
        ids: None,
        board_area_frac: area_frac,
    })
}

fn detect_charuco(cfg: &BoardConfig, mat: &impl ToInputArray, w: u32, h: u32, marker_mm: f64, dict_name: &str) -> Option<Detection> {
    let dict_id = aruco_dict_id(dict_name)?;
    let dictionary = objdetect::get_predefined_dictionary(dict_id).ok()?;

    let squares = Size::new(cfg.inner_cols as i32 + 1, cfg.inner_rows as i32 + 1);
    let board = objdetect::CharucoBoard::new_def(squares, cfg.rect_w_mm as f32, marker_mm as f32, &dictionary).ok()?;
    let detector = objdetect::CharucoDetector::new_def(&board).ok()?;

    let mut charuco_corners: Vector<Point2f> = Vector::new();
    let mut charuco_ids: Vector<i32> = Vector::new();
    detector.detect_board_def(mat, &mut charuco_corners, &mut charuco_ids).ok()?;

    let n_total = (cfg.inner_cols as usize) * (cfg.inner_rows as usize);
    if n_total == 0 || charuco_ids.is_empty() {
        return None;
    }
    let fraction = charuco_ids.len() as f64 / n_total as f64;
    if fraction < CHARUCO_MIN_FRACTION {
        return None;
    }

    let corners: Vec<(f32, f32)> = charuco_corners.as_slice().iter().map(|p| (p.x, p.y)).collect();
    let ids: Vec<i32> = charuco_ids.as_slice().to_vec();
    let area_frac = (convex_hull_area(&corners) / (w as f64 * h as f64)) as f32;

    Some(Detection {
        corners,
        ids: Some(ids),
        board_area_frac: area_frac,
    })
}

fn aruco_dict_id(name: &str) -> Option<PredefinedDictionaryType> {
    use PredefinedDictionaryType::*;
    let id = match name {
        "DICT_4X4_50" => DICT_4X4_50,
        "DICT_4X4_100" => DICT_4X4_100,
        "DICT_4X4_250" => DICT_4X4_250,
        "DICT_4X4_1000" => DICT_4X4_1000,
        "DICT_5X5_50" => DICT_5X5_50,
        "DICT_5X5_100" => DICT_5X5_100,
        "DICT_5X5_250" => DICT_5X5_250,
        "DICT_5X5_1000" => DICT_5X5_1000,
        "DICT_6X6_50" => DICT_6X6_50,
        "DICT_6X6_100" => DICT_6X6_100,
        "DICT_6X6_250" => DICT_6X6_250,
        "DICT_6X6_1000" => DICT_6X6_1000,
        "DICT_7X7_50" => DICT_7X7_50,
        "DICT_7X7_100" => DICT_7X7_100,
        "DICT_7X7_250" => DICT_7X7_250,
        "DICT_7X7_1000" => DICT_7X7_1000,
        "DICT_ARUCO_ORIGINAL" => DICT_ARUCO_ORIGINAL,
        "DICT_APRILTAG_16h5" => DICT_APRILTAG_16h5,
        "DICT_APRILTAG_25h9" => DICT_APRILTAG_25h9,
        "DICT_APRILTAG_36h10" => DICT_APRILTAG_36h10,
        "DICT_APRILTAG_36h11" => DICT_APRILTAG_36h11,
        "DICT_ARUCO_MIP_36h12" => DICT_ARUCO_MIP_36h12,
        _ => return None,
    };
    Some(id)
}



#[derive(Clone)]
struct Grid {
    data: Vec<(f32, f32)>,
    rows: usize,
    cols: usize,
}

impl Grid {
    fn from_row_major(flat: &[(f32, f32)], rows: usize, cols: usize) -> Self {
        Grid {
            data: flat.to_vec(),
            rows,
            cols,
        }
    }

    fn get(&self, r: usize, c: usize) -> (f32, f32) {
        self.data[r * self.cols + c]
    }

    fn flipud(&self) -> Self {
        let mut data = vec![(0.0, 0.0); self.rows * self.cols];
        for r in 0..self.rows {
            for c in 0..self.cols {
                data[r * self.cols + c] = self.get(self.rows - 1 - r, c);
            }
        }
        Grid { data, rows: self.rows, cols: self.cols }
    }

    fn fliplr(&self) -> Self {
        let mut data = vec![(0.0, 0.0); self.rows * self.cols];
        for r in 0..self.rows {
            for c in 0..self.cols {
                data[r * self.cols + c] = self.get(r, self.cols - 1 - c);
            }
        }
        Grid { data, rows: self.rows, cols: self.cols }
    }

    fn rot180(&self) -> Self {
        self.flipud().fliplr()
    }

    fn transpose(&self) -> Self {
        let mut data = vec![(0.0, 0.0); self.rows * self.cols];
        for r in 0..self.cols {
            for c in 0..self.rows {
                data[r * self.rows + c] = self.get(c, r);
            }
        }
        Grid { data, rows: self.cols, cols: self.rows }
    }
}

fn all_orientations(g: &Grid) -> Vec<Grid> {
    let t = g.transpose();
    vec![g.clone(), g.flipud(), g.fliplr(), g.rot180(), t.clone(), t.flipud(), t.fliplr(), t.rot180()]
}

fn canonicalize_checkerboard(flat: &[(f32, f32)], rows: usize, cols: usize) -> Vec<(f32, f32)> {
    debug_assert_eq!(flat.len(), rows * cols);

    let mut candidates: Vec<Grid> = Vec::new();
    for (r0, c0) in [(rows, cols), (cols, rows)] {
        let grid_init = Grid::from_row_major(flat, r0, c0);
        for oriented in all_orientations(&grid_init) {
            if oriented.rows == rows && oriented.cols == cols {
                candidates.push(oriented);
            }
        }
    }

    let mut best: Option<Grid> = None;
    for cand in &candidates {
        let bl = cand.get(0, 0);
        let br = cand.get(0, cols - 1);
        let tl = cand.get(rows - 1, 0);
        let tr = cand.get(rows - 1, cols - 1);
        let scores = [bl.0 - bl.1, br.0 - br.1, tl.0 - tl.1, tr.0 - tr.1];
        let min_score = scores.iter().cloned().fold(f32::INFINITY, f32::min);
        if scores[0] != min_score {
            continue;
        }
        if br.0 <= bl.0 {
            continue;
        }
        if tl.1 >= bl.1 {
            continue;
        }

        let mut ok = true;
        'row_check: for r in 0..rows {
            for c in 0..cols.saturating_sub(1) {
                if cand.get(r, c + 1).0 <= cand.get(r, c).0 {
                    ok = false;
                    break 'row_check;
                }
            }
        }
        if ok {
            'col_check: for c in 0..cols {
                for r in 0..rows.saturating_sub(1) {
                    if cand.get(r + 1, c).1 >= cand.get(r, c).1 {
                        ok = false;
                        break 'col_check;
                    }
                }
            }
        }
        if !ok {
            continue;
        }

        best = Some(cand.clone());
        break;
    }

    let best = best.unwrap_or_else(|| Grid::from_row_major(flat, rows, cols));

    let mut out = vec![(0.0f32, 0.0f32); rows * cols];
    for c_from_right in 0..cols {
        for r_from_top in 0..rows {
            let r = rows - 1 - r_from_top;
            let c = cols - 1 - c_from_right;
            let k = c_from_right * rows + r_from_top;
            out[k] = best.get(r, c);
        }
    }
    out
}



fn convex_hull_area(points: &[(f32, f32)]) -> f64 {
    let mut pts: Vec<(f64, f64)> = points.iter().map(|&(x, y)| (x as f64, y as f64)).collect();
    pts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    pts.dedup();
    if pts.len() < 3 {
        return 0.0;
    }

    fn cross(o: (f64, f64), a: (f64, f64), b: (f64, f64)) -> f64 {
        (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0)
    }

    let n = pts.len();
    let mut hull: Vec<(f64, f64)> = Vec::with_capacity(2 * n);
    for &p in &pts {
        while hull.len() >= 2 && cross(hull[hull.len() - 2], hull[hull.len() - 1], p) <= 0.0 {
            hull.pop();
        }
        hull.push(p);
    }
    let lower_len = hull.len() + 1;
    for &p in pts.iter().rev() {
        while hull.len() >= lower_len && cross(hull[hull.len() - 2], hull[hull.len() - 1], p) <= 0.0 {
            hull.pop();
        }
        hull.push(p);
    }
    hull.pop();

    let mut area = 0.0;
    for i in 0..hull.len() {
        let (x1, y1) = hull[i];
        let (x2, y2) = hull[(i + 1) % hull.len()];
        area += x1 * y2 - x2 * y1;
    }
    (area / 2.0).abs()
}


#[cfg(test)]
mod tests {
    use super::*;

    const INNER_COLS: u32 = 7;
    const INNER_ROWS: u32 = 6;
    const RECT_W_MM: f64 = 82.0;
    const RECT_H_MM: f64 = 98.0;

    fn test_cfg() -> BoardConfig {
        BoardConfig {
            kind: BoardKind::Checkerboard,
            inner_cols: INNER_COLS,
            inner_rows: INNER_ROWS,
            rect_w_mm: RECT_W_MM,
            rect_h_mm: RECT_H_MM,
        }
    }


    fn synth_checkerboard() -> (Vec<u8>, u32, u32, u32, u32, u32) {
        let cell_w = 41u32;
        let cell_h = 49u32;
        let margin = 60u32;
        let squares_x = INNER_COLS + 1;
        let squares_y = INNER_ROWS + 1;
        let w = squares_x * cell_w + 2 * margin;
        let h = squares_y * cell_h + 2 * margin;
        let mut buf = vec![255u8; (w * h) as usize];
        for sy in 0..squares_y {
            for sx in 0..squares_x {
                if (sx + sy) % 2 == 0 {
                    let x0 = margin + sx * cell_w;
                    let y0 = margin + sy * cell_h;
                    for y in y0..y0 + cell_h {
                        for x in x0..x0 + cell_w {
                            buf[(y * w + x) as usize] = 0;
                        }
                    }
                }
            }
        }
        (buf, w, h, cell_w, cell_h, margin)
    }


    fn apply_homography(m: &Mat, pt: (f64, f64)) -> opencv::Result<(f64, f64)> {
        let mut v = [0f64; 3];
        for i in 0..3 {
            let row = [*m.at_2d::<f64>(i, 0)?, *m.at_2d::<f64>(i, 1)?, *m.at_2d::<f64>(i, 2)?];
            v[i as usize] = row[0] * pt.0 + row[1] * pt.1 + row[2];
        }
        Ok((v[0] / v[2], v[1] / v[2]))
    }


    struct WarpedTestBoard {
        data: Vec<u8>,
        w: u32,
        h: u32,
        homography: Mat,
        cell_w: u32,
        cell_h: u32,
        margin: u32,
    }

    fn build_warped_test_board() -> WarpedTestBoard {
        let (buf, src_w, src_h, cell_w, cell_h, margin) = synth_checkerboard();
        let src_mat = Mat::new_rows_cols_with_data(src_h as i32, src_w as i32, &buf).expect("build source mat");

        let pad = 150i32;
        let dst_w = src_w as i32 + 2 * pad;
        let dst_h = src_h as i32 + 2 * pad;
        let src_pts = Vector::<Point2f>::from_iter([
            Point2f::new(0.0, 0.0),
            Point2f::new(src_w as f32, 0.0),
            Point2f::new(src_w as f32, src_h as f32),
            Point2f::new(0.0, src_h as f32),
        ]);
        let dst_pts = Vector::<Point2f>::from_iter([
            Point2f::new(pad as f32 + 20.0, pad as f32 + 10.0),
            Point2f::new((pad + src_w as i32) as f32 - 15.0, pad as f32 + 5.0),
            Point2f::new((pad + src_w as i32) as f32 - 5.0, (pad + src_h as i32) as f32 - 20.0),
            Point2f::new(pad as f32 + 10.0, (pad + src_h as i32) as f32 - 10.0),
        ]);
        let src_pts_mat = Mat::from_slice(src_pts.as_slice()).expect("src pts mat");
        let dst_pts_mat = Mat::from_slice(dst_pts.as_slice()).expect("dst pts mat");
        let m = imgproc::get_perspective_transform_def(&src_pts_mat, &dst_pts_mat).expect("homography");

        let mut warped = Mat::default();
        imgproc::warp_perspective_def(&src_mat, &mut warped, &m, Size::new(dst_w, dst_h)).expect("warp");

        let w = warped.cols() as u32;
        let h = warped.rows() as u32;
        let data: Vec<u8> = warped.data_bytes().expect("warped bytes").to_vec();

        WarpedTestBoard {
            data,
            w,
            h,
            homography: m,
            cell_w,
            cell_h,
            margin,
        }
    }

    #[test]
    fn synthetic_checkerboard_detected() {
        let board = build_warped_test_board();

        let x_tr = (board.margin + INNER_COLS * board.cell_w) as f64;
        let y_tr = (board.margin + board.cell_h) as f64;
        let (exp_x, exp_y) = apply_homography(&board.homography, (x_tr, y_tr)).expect("apply homography");

        let cfg = test_cfg();
        let det = detect(&cfg, &board.data, board.w, board.h).expect("board should be detected");
        assert_eq!(det.corners.len(), (INNER_COLS * INNER_ROWS) as usize);

        let (fx, fy) = det.corners[0];
        let dist = ((fx as f64 - exp_x).powi(2) + (fy as f64 - exp_y).powi(2)).sqrt();
        assert!(
            dist < 3.0,
            "first corner {:?} too far from expected ({exp_x}, {exp_y}): {dist}px",
            det.corners[0]
        );
    }


    #[test]
    fn pyramid_matches_full_within_half_pixel() {
        let board = build_warped_test_board();
        let cfg = test_cfg();

        let full = detect_opts(
            &cfg,
            &board.data,
            board.w,
            board.h,
            DetectOpts {
                speed: DetectSpeed::Full,
                skip_prefilter: true,
            },
        )
        .detection
        .expect("full mode should detect the synthetic board");
        let pyramid = detect_opts(
            &cfg,
            &board.data,
            board.w,
            board.h,
            DetectOpts {
                speed: DetectSpeed::Pyramid,
                skip_prefilter: true,
            },
        )
        .detection
        .expect("pyramid mode should detect the synthetic board");

        assert_eq!(full.corners.len(), pyramid.corners.len());
        for (i, (a, b)) in full.corners.iter().zip(pyramid.corners.iter()).enumerate() {
            let dist = (((a.0 - b.0) as f64).powi(2) + ((a.1 - b.1) as f64).powi(2)).sqrt();
            assert!(dist < 0.5, "corner {i}: full={a:?} pyramid={b:?} dist={dist}px (want < 0.5px)");
        }
    }

    #[test]
    fn prefilter_rejects_blank_image() {
        let (w, h) = (640u32, 480u32);
        let buf = vec![128u8; (w * h) as usize];
        let cfg = test_cfg();
        let out = detect_opts(&cfg, &buf, w, h, DetectOpts::default());
        assert!(out.detection.is_none());
        assert!(out.prefiltered, "FAST_CHECK should reject a flat/blank image outright");
    }

    #[test]
    fn blank_image_none() {
        let (w, h) = (640u32, 480u32);
        let buf = vec![128u8; (w * h) as usize];
        let cfg = test_cfg();
        assert!(detect(&cfg, &buf, w, h).is_none());
    }

    #[test]
    fn object_points_non_square() {
        let cfg = test_cfg();
        let pts = cfg.object_points();
        assert_eq!(pts.len(), (INNER_COLS * INNER_ROWS) as usize);

        assert!(pts.iter().any(|p| p.x == 0.0 && p.y == 0.0));
        assert!(pts.iter().any(|p| (p.x - RECT_W_MM as f32).abs() < 1e-4 && p.y == 0.0));
        assert!(pts.iter().any(|p| p.x == 0.0 && (p.y - RECT_H_MM as f32).abs() < 1e-4));
    }
}
