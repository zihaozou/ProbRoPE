use crate::board::Detection;


pub struct PoolEntry {

    id: u64,
    pub detection: Detection,
    coverage: [[bool; 6]; 6],

    center: (f32, f32),
    area: f32,
    skew: f32,
}

impl PoolEntry {

    pub fn id(&self) -> u64 {
        self.id
    }
}


#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Offer {

    Accepted,

    Replaced(usize),

    Rejected,
}

pub struct KeyframePool {
    cap: usize,
    image_w: u32,
    image_h: u32,
    entries: Vec<PoolEntry>,
    next_id: u64,
}

impl KeyframePool {
    pub fn new(cap: usize, image_w: u32, image_h: u32) -> Self {
        KeyframePool {
            cap,
            image_w,
            image_h,
            entries: Vec::with_capacity(cap),
            next_id: 0,
        }
    }


    pub fn offer(&mut self, det: Detection) -> Offer {
        let coverage = coverage_of(&det.corners, self.image_w, self.image_h);
        let center = center_of(&det.corners, self.image_w, self.image_h);
        let skew = skew_of(&det.corners);
        let area = det.board_area_frac;
        let candidate_bin = pose_bin_id(center, area, skew);

        let union = self.coverage_mask();
        let new_cells = count_new_cells(&coverage, &union);
        let existing_bin_count = self
            .entries
            .iter()
            .filter(|e| pose_bin_id(e.center, e.area, e.skew) == candidate_bin)
            .count();

        if existing_bin_count > 0 && new_cells == 0 {
            return Offer::Rejected;
        }

        let id = self.next_id;
        let entry = PoolEntry {
            id,
            detection: det,
            coverage,
            center,
            area,
            skew,
        };

        if self.entries.len() < self.cap {
            self.entries.push(entry);
            self.next_id += 1;
            return Offer::Accepted;
        }

        if self.entries.is_empty() {
            return Offer::Rejected;
        }
        let novel = new_cells > 0 || existing_bin_count == 0;
        if !novel {
            return Offer::Rejected;
        }

        let mut worst_idx = 0usize;
        let mut worst_val = usize::MAX;
        for i in 0..self.entries.len() {
            let v = marginal_value(&self.entries, i);
            if v < worst_val {
                worst_val = v;
                worst_idx = i;
            }
        }
        self.entries[worst_idx] = entry;
        self.next_id += 1;
        Offer::Replaced(worst_idx)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn entries(&self) -> &[PoolEntry] {
        &self.entries
    }


    pub fn remove(&mut self, idx: usize) {
        self.entries.remove(idx);
    }


    pub fn remove_by_id(&mut self, id: u64) -> bool {
        match self.entries.iter().position(|e| e.id == id) {
            Some(idx) => {
                self.entries.remove(idx);
                true
            }
            None => false,
        }
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }

    pub fn coverage_mask(&self) -> [[bool; 6]; 6] {
        let mut union = [[false; 6]; 6];
        for e in &self.entries {
            for r in 0..6 {
                for c in 0..6 {
                    union[r][c] |= e.coverage[r][c];
                }
            }
        }
        union
    }
}



fn coverage_of(corners: &[(f32, f32)], image_w: u32, image_h: u32) -> [[bool; 6]; 6] {
    let mut cov = [[false; 6]; 6];
    let w = image_w.max(1) as f32;
    let h = image_h.max(1) as f32;
    for &(x, y) in corners {
        let col = ((x / w) * 6.0).floor().clamp(0.0, 5.0) as usize;
        let row = ((y / h) * 6.0).floor().clamp(0.0, 5.0) as usize;
        cov[row][col] = true;
    }
    cov
}


fn center_of(corners: &[(f32, f32)], image_w: u32, image_h: u32) -> (f32, f32) {
    if corners.is_empty() {
        return (0.0, 0.0);
    }
    let (sx, sy) = corners.iter().fold((0.0f32, 0.0f32), |acc, &(x, y)| (acc.0 + x, acc.1 + y));
    let n = corners.len() as f32;
    let w = image_w.max(1) as f32;
    let h = image_h.max(1) as f32;
    ((sx / n) / w, (sy / n) / h)
}


fn skew_of(corners: &[(f32, f32)]) -> f32 {
    if corners.len() < 3 {
        return 1.0;
    }
    let mut p_min_sum = corners[0];
    let mut p_max_sum = corners[0];
    let mut p_min_diff = corners[0];
    let mut p_max_diff = corners[0];
    for &p in corners {
        if p.0 + p.1 < p_min_sum.0 + p_min_sum.1 {
            p_min_sum = p;
        }
        if p.0 + p.1 > p_max_sum.0 + p_max_sum.1 {
            p_max_sum = p;
        }
        if p.0 - p.1 < p_min_diff.0 - p_min_diff.1 {
            p_min_diff = p;
        }
        if p.0 - p.1 > p_max_diff.0 - p_max_diff.1 {
            p_max_diff = p;
        }
    }
    let dist = |a: (f32, f32), b: (f32, f32)| ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt();
    let diag1 = dist(p_min_sum, p_max_sum);
    let diag2 = dist(p_min_diff, p_max_diff);
    let (lo, hi) = if diag1 < diag2 { (diag1, diag2) } else { (diag2, diag1) };
    if hi <= f32::EPSILON {
        1.0
    } else {
        lo / hi
    }
}


const AREA_BIN_EDGES: [f32; 2] = [0.12, 0.30];


const SKEW_FRONTAL_THRESHOLD: f32 = 0.85;


fn pose_bin_id(center: (f32, f32), area: f32, skew: f32) -> usize {
    let cx_bin = ((center.0 * 4.0).floor() as isize).clamp(0, 3) as usize;
    let cy_bin = ((center.1 * 4.0).floor() as isize).clamp(0, 3) as usize;
    let area_bin = if area < AREA_BIN_EDGES[0] {
        0
    } else if area < AREA_BIN_EDGES[1] {
        1
    } else {
        2
    };
    let skew_bin = if skew >= SKEW_FRONTAL_THRESHOLD { 0 } else { 1 };
    ((cx_bin * 4 + cy_bin) * 3 + area_bin) * 2 + skew_bin
}


fn count_new_cells(candidate: &[[bool; 6]; 6], union: &[[bool; 6]; 6]) -> usize {
    let mut n = 0;
    for r in 0..6 {
        for c in 0..6 {
            if candidate[r][c] && !union[r][c] {
                n += 1;
            }
        }
    }
    n
}

fn marginal_value(entries: &[PoolEntry], idx: usize) -> usize {
    let mut unique_cells = 0usize;
    for r in 0..6 {
        for c in 0..6 {
            if !entries[idx].coverage[r][c] {
                continue;
            }
            let covered_elsewhere = entries.iter().enumerate().any(|(j, e)| j != idx && e.coverage[r][c]);
            if !covered_elsewhere {
                unique_cells += 1;
            }
        }
    }
    let bin = pose_bin_id(entries[idx].center, entries[idx].area, entries[idx].skew);
    let bin_count = entries.iter().filter(|e| pose_bin_id(e.center, e.area, e.skew) == bin).count();
    let sole_bonus = if bin_count == 1 { 1 } else { 0 };
    unique_cells + sole_bonus
}

#[cfg(test)]
mod tests {
    use super::*;


    fn synth_detection(image_w: u32, image_h: u32, center: (f32, f32), half_w: f32, half_h: f32) -> Detection {
        let rows = 6usize;
        let cols = 7usize;
        let mut corners = Vec::with_capacity(rows * cols);
        for r in 0..rows {
            for c in 0..cols {
                let fx = c as f32 / (cols - 1) as f32;
                let fy = r as f32 / (rows - 1) as f32;
                let x = center.0 + (fx - 0.5) * 2.0 * half_w;
                let y = center.1 + (fy - 0.5) * 2.0 * half_h;
                corners.push((x, y));
            }
        }
        let area = ((2.0 * half_w) * (2.0 * half_h) / (image_w as f32 * image_h as f32)).min(1.0);
        Detection {
            corners,
            ids: None,
            board_area_frac: area,
        }
    }

    #[test]
    fn duplicate_rejected() {
        let mut pool = KeyframePool::new(10, 640, 480);
        let det1 = synth_detection(640, 480, (320.0, 240.0), 100.0, 80.0);
        let det2 = synth_detection(640, 480, (320.0, 240.0), 100.0, 80.0);
        assert_eq!(pool.offer(det1), Offer::Accepted);
        assert_eq!(pool.offer(det2), Offer::Rejected);
        assert_eq!(pool.len(), 1);
    }

    #[test]
    fn spread_fills_and_covers() {
        let (w, h) = (640u32, 480u32);
        let mut pool = KeyframePool::new(20, w, h);

        for i in 0..25 {
            let pos = i % 16;
            let bx = pos % 4;
            let by = pos / 4;
            let cx = w as f32 * (bx as f32 + 0.5) / 4.0;
            let cy = h as f32 * (by as f32 + 0.5) / 4.0;
            let variant = i / 16;
            let (half_w, half_h) = if variant == 0 { (60.0, 45.0) } else { (140.0, 110.0) };
            pool.offer(synth_detection(w, h, (cx, cy), half_w, half_h));
        }

        assert_eq!(pool.len(), 20);

        let mask = pool.coverage_mask();
        let covered: usize = mask.iter().flatten().filter(|&&b| b).count();
        assert!(covered >= 12, "expected >=12/36 cells covered, got {covered}");
    }

    #[test]
    fn cap_respected_and_replacement() {
        let (w, h) = (640u32, 480u32);
        let mut pool = KeyframePool::new(5, w, h);

        let cluster_center = (300.0, 240.0);
        let offsets = [(0.0, 0.0), (110.0, 0.0), (0.0, 90.0), (-110.0, 0.0), (0.0, -90.0)];
        for (dx, dy) in offsets {
            let center = (cluster_center.0 + dx, cluster_center.1 + dy);
            let offer = pool.offer(synth_detection(w, h, center, 40.0, 35.0));
            assert_ne!(offer, Offer::Rejected, "clustered fill should not reject: {offer:?}");
        }
        assert_eq!(pool.len(), 5);

        let mask_before = pool.coverage_mask();
        let corner_row = 0usize;
        let corner_col = 0usize;
        assert!(!mask_before[corner_row][corner_col], "test setup: corner cell should start uncovered");

        let corner_det = synth_detection(w, h, (50.0, 50.0), 30.0, 25.0);
        let offer = pool.offer(corner_det);
        assert!(matches!(offer, Offer::Replaced(_)), "expected Replaced, got {offer:?}");
        assert_eq!(pool.len(), 5);

        let mask_after = pool.coverage_mask();
        assert!(mask_after[corner_row][corner_col], "coverage mask should include the new corner region");
    }

    #[test]
    fn remove_and_clear() {
        let mut pool = KeyframePool::new(5, 640, 480);
        pool.offer(synth_detection(640, 480, (100.0, 100.0), 40.0, 30.0));
        pool.offer(synth_detection(640, 480, (500.0, 400.0), 40.0, 30.0));
        assert_eq!(pool.len(), 2);
        assert!(!pool.is_empty());

        pool.remove(0);
        assert_eq!(pool.len(), 1);
        assert_eq!(pool.entries().len(), 1);

        pool.clear();
        assert_eq!(pool.len(), 0);
        assert!(pool.is_empty());
        assert_eq!(pool.entries().len(), 0);
    }
}
