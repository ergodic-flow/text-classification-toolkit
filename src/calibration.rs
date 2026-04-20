use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone)]
pub struct PlattCalibrator {
    params: Vec<(f64, f64)>,
    n_classes: usize,
}

impl PlattCalibrator {
    pub fn new(n_classes: usize) -> Self {
        Self {
            params: vec![(1.0, 0.0); n_classes],
            n_classes,
        }
    }

    pub fn fit_binary(&mut self, scores: &[f64], labels: &[usize]) {
        let targets: Vec<f64> = labels
            .iter()
            .map(|&l| if l == 1 { 1.0 } else { 0.0 })
            .collect();
        self.params[0] = fit_platt_sigmoid(scores, &targets);
    }

    pub fn fit_multiclass(&mut self, scores: &[Vec<f64>], labels: &[usize]) {
        for k in 0..self.n_classes {
            let class_scores: Vec<f64> = scores.iter().map(|s| s[k]).collect();
            let targets: Vec<f64> = labels
                .iter()
                .map(|&l| if l == k { 1.0 } else { 0.0 })
                .collect();
            self.params[k] = fit_platt_sigmoid(&class_scores, &targets);
        }
    }

    pub fn fit_multilabel(&mut self, scores: &[Vec<f64>], labels: &[Vec<usize>]) {
        for k in 0..self.n_classes {
            let class_scores: Vec<f64> = scores.iter().map(|s| s[k]).collect();
            let targets: Vec<f64> = labels
                .iter()
                .map(|ls| if ls.contains(&k) { 1.0 } else { 0.0 })
                .collect();
            self.params[k] = fit_platt_sigmoid(&class_scores, &targets);
        }
    }

    pub fn transform(&self, raw_scores: &[f64]) -> Vec<f64> {
        let mut calibrated: Vec<f64> = self
            .params
            .iter()
            .zip(raw_scores.iter())
            .map(|(&(a, b), &s)| sigmoid(a * s + b))
            .collect();

        if self.n_classes > 1 {
            let sum: f64 = calibrated.iter().sum();
            if sum > 0.0 {
                for p in &mut calibrated {
                    *p /= sum;
                }
            }
        }

        calibrated
    }

    pub fn n_classes(&self) -> usize {
        self.n_classes
    }
}

fn sigmoid(z: f64) -> f64 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let ez = z.exp();
        ez / (1.0 + ez)
    }
}

fn fit_platt_sigmoid(scores: &[f64], targets: &[f64]) -> (f64, f64) {
    let n = scores.len();
    if n == 0 {
        return (1.0, 0.0);
    }

    let n_pos = targets.iter().filter(|&&t| t == 1.0).count() as f64;
    let n_neg = n as f64 - n_pos;

    if n_pos == 0.0 || n_neg == 0.0 {
        return (1.0, 0.0);
    }

    let adjusted_targets: Vec<f64> = targets
        .iter()
        .map(|&t| {
            if t == 1.0 {
                (n_pos + 1.0) / (n_pos + 2.0)
            } else {
                1.0 / (n_neg + 2.0)
            }
        })
        .collect();

    let mut a = 0.0f64;
    let mut b = ((n_pos + 1.0) / (n_neg + 1.0)).ln();

    let max_iter = 100;
    let min_step = 1e-10;

    for _ in 0..max_iter {
        let mut grad_a = 0.0;
        let mut grad_b = 0.0;
        let mut h11 = 0.0;
        let mut h22 = 0.0;
        let mut h12 = 0.0;

        for i in 0..n {
            let p = sigmoid(a * scores[i] + b);
            let d = adjusted_targets[i] - p;
            let s = scores[i];

            grad_a += d * s;
            grad_b += d;

            let pp = p * (1.0 - p);
            h11 += s * s * pp;
            h22 += pp;
            h12 += s * pp;
        }

        h11 += 1e-12;
        h22 += 1e-12;

        let det = h11 * h22 - h12 * h12;
        if det.abs() < 1e-20 {
            break;
        }

        let da = (h22 * grad_a - h12 * grad_b) / det;
        let db = (h11 * grad_b - h12 * grad_a) / det;

        a += da;
        b += db;

        if da.abs() < min_step && db.abs() < min_step {
            break;
        }
    }

    (a, b)
}
