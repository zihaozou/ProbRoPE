use serde::Serialize;

pub struct ClockModel {
    x0: i64,

    theta: [f64; 2],
    p: [[f64; 2]; 2],
    lambda: f64,
    seeded: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct ClockParams {
    pub x0: i64,
    pub drift: f64,
    pub offset_us: f64,
}

impl ClockModel {
    pub fn new() -> Self {
        Self { x0: 0, theta: [0.0, 0.0], p: [[1e-2, 0.0], [0.0, 1e8]], lambda: 0.999, seeded: false }
    }


    pub fn seed(&mut self, t_flir_us: i64, t_evk_us: i64) {
        self.x0 = t_flir_us;
        self.theta = [0.0, (t_evk_us - t_flir_us) as f64];
        self.p = [[1e-2, 0.0], [0.0, 1e8]];
        self.seeded = true;
    }

    pub fn is_seeded(&self) -> bool { self.seeded }

    pub fn update(&mut self, t_flir_us: i64, t_evk_us: i64) {
        let x = (t_flir_us - self.x0) as f64;
        let y = (t_evk_us - t_flir_us) as f64;
        let phi = [x, 1.0];
        let pphi = [
            self.p[0][0] * phi[0] + self.p[0][1] * phi[1],
            self.p[1][0] * phi[0] + self.p[1][1] * phi[1],
        ];
        let denom = self.lambda + phi[0] * pphi[0] + phi[1] * pphi[1];
        let k = [pphi[0] / denom, pphi[1] / denom];
        let err = y - (phi[0] * self.theta[0] + phi[1] * self.theta[1]);
        self.theta[0] += k[0] * err;
        self.theta[1] += k[1] * err;
        for r in 0..2 {
            for c in 0..2 {
                self.p[r][c] = (self.p[r][c] - k[r] * pphi[c]) / self.lambda;
            }
        }
    }

    pub fn predict(&self, t_flir_us: i64) -> i64 {
        let x = (t_flir_us - self.x0) as f64;
        t_flir_us + (self.theta[0] * x + self.theta[1]).round() as i64
    }

    pub fn drift_ppm(&self) -> f64 { self.theta[0] * 1e6 }
    pub fn offset_us(&self) -> f64 { self.theta[1] }


    pub fn model(&self) -> ClockParams {
        ClockParams { x0: self.x0, drift: self.theta[0], offset_us: self.theta[1] }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converges_on_offset_and_drift() {
        let mut c = ClockModel::new();
        c.seed(0, 5000);
        for i in 1..200i64 {
            let t_flir = i * 33_333;
            let t_evk = (t_flir as f64 * 1.0005 + 5000.0) as i64;
            c.update(t_flir, t_evk);
        }
        let t_flir = 200 * 33_333;
        let truth = (t_flir as f64 * 1.0005 + 5000.0) as i64;
        assert!((c.predict(t_flir) - truth).abs() < 50, "pred={} truth={}", c.predict(t_flir), truth);
        assert!((c.drift_ppm() - 500.0).abs() < 50.0, "drift_ppm={}", c.drift_ppm());
    }

    #[test]
    fn seed_alone_predicts_constant_offset() {
        let mut c = ClockModel::new();
        c.seed(1_000_000, 1_003_000);
        assert_eq!(c.predict(2_000_000), 2_003_000);
    }

    #[test]
    fn model_exposes_full_parameter_set() {
        let mut c = ClockModel::new();
        c.seed(1_000_000, 1_003_000);
        for i in 1..50i64 {
            let t_flir = 1_000_000 + i * 33_333;
            let t_evk = (t_flir as f64 * 1.0003 + 3000.0) as i64;
            c.update(t_flir, t_evk);
        }
        let params = c.model();
        assert_eq!(params.x0, 1_000_000, "x0 is the seed's t_flir, fixed for the life of this fit");
        assert_eq!(params.drift * 1e6, c.drift_ppm());
        assert_eq!(params.offset_us, c.offset_us());
        let t_flir = 1_000_000 + 50 * 33_333;
        let x = (t_flir - params.x0) as f64;
        let reconstructed = t_flir + (params.drift * x + params.offset_us).round() as i64;
        assert_eq!(reconstructed, c.predict(t_flir));
    }
}
