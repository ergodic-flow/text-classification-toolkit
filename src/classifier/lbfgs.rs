use serde::{Deserialize, Serialize};

use super::Objective;

#[derive(Serialize, Deserialize)]
pub struct LbfgsClassifier {
    pub weights: Vec<f64>,
    pub biases: Vec<f64>,
    pub objective: Objective,
    pub n_features: usize,
    pub n_classes: usize,
}

fn sigmoid(z: f64) -> f64 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let ez = z.exp();
        ez / (1.0 + ez)
    }
}

fn dot(u: &[f64], v: &[f64]) -> f64 {
    u.iter().zip(v.iter()).map(|(&a, &b)| a * b).sum()
}

impl LbfgsClassifier {
    pub fn new(n_features: usize, objective: Objective) -> Self {
        let n_classes = match objective {
            Objective::Binary => 1,
            Objective::Multinomial { n_classes } => n_classes,
        };

        Self {
            weights: vec![0.0; n_features * n_classes],
            biases: vec![0.0; n_classes],
            objective,
            n_features,
            n_classes,
        }
    }

    /// Evaluates the total loss and gradient over the entire dataset using precomputed weights.
    /// Scratch buffers are supplied by the caller because line search evaluates this repeatedly.
    fn evaluate(
        &self,
        x: &[f64],
        features: &[Vec<(usize, f64)>],
        labels: &[usize],
        weights: &[f64],
        sum_weights: f64,
        l2_reg: f64,
        grad: &mut [f64],
        z_scores: &mut [f64],
        exps: &mut [f64],
    ) -> f64 {
        let n_classes = self.n_classes;
        let w_len = self.weights.len();
        let mut total_loss = 0.0;

        debug_assert_eq!(grad.len(), x.len());
        debug_assert_eq!(z_scores.len(), n_classes);
        debug_assert_eq!(exps.len(), n_classes);
        grad.fill(0.0);

        let w = &x[0..w_len];
        let b = &x[w_len..];
        let (g_w, g_b) = grad.split_at_mut(w_len);

        for (i, feat) in features.iter().enumerate() {
            let true_class = labels[i];
            let w_i = weights[i];

            if w_i == 0.0 {
                continue;
            }

            z_scores.copy_from_slice(b);

            for &(feat_idx, val) in feat {
                let w_offset = feat_idx * n_classes;
                if w_offset + n_classes <= w_len {
                    for k in 0..n_classes {
                        z_scores[k] += w[w_offset + k] * val;
                    }
                }
            }

            match self.objective {
                Objective::Binary => {
                    let z = z_scores[0];
                    let y = true_class as f64;

                    let loss_i = if y == 1.0 {
                        if z > 0.0 {
                            (-z).exp().ln_1p()
                        } else {
                            -z + z.exp().ln_1p()
                        }
                    } else {
                        if z > 0.0 {
                            z + (-z).exp().ln_1p()
                        } else {
                            z.exp().ln_1p()
                        }
                    };
                    total_loss += loss_i * w_i;

                    let p = sigmoid(z);
                    let err = (p - y) * w_i;

                    g_b[0] += err;
                    for &(feat_idx, val) in feat {
                        let w_offset = feat_idx * n_classes;
                        if w_offset + n_classes <= w_len {
                            g_w[w_offset] += err * val;
                        }
                    }
                }
                Objective::Multinomial { .. } => {
                    let max_z = z_scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    let mut sum_exp = 0.0;

                    for k in 0..n_classes {
                        let e = (z_scores[k] - max_z).exp();
                        exps[k] = e;
                        sum_exp += e;
                    }

                    let loss_i = -(z_scores[true_class] - max_z) + sum_exp.ln();
                    total_loss += loss_i * w_i;

                    for k in 0..n_classes {
                        let p_k = exps[k] / sum_exp;
                        let err = (p_k - if k == true_class { 1.0 } else { 0.0 }) * w_i;

                        g_b[k] += err;
                        for &(feat_idx, val) in feat {
                            let w_offset = feat_idx * n_classes;
                            if w_offset + n_classes <= w_len {
                                g_w[w_offset + k] += err * val;
                            }
                        }
                    }
                }
            }
        }

        total_loss /= sum_weights;
        for g_val in grad.iter_mut() {
            *g_val /= sum_weights;
        }

        if l2_reg > 0.0 {
            let mut l2_loss = 0.0;
            for i in 0..w_len {
                l2_loss += w[i] * w[i];
                grad[i] += l2_reg * w[i];
            }
            total_loss += 0.5 * l2_reg * l2_loss;
        }

        total_loss
    }

    pub fn train(
        &mut self,
        features: &[Vec<(usize, f64)>],
        labels: &[usize],
        sample_weights: Option<&[f64]>,
        class_weights: Option<&[f64]>,
        l2_reg: f64,
        m: usize,
        max_iter: usize,
        tol: f64,
    ) {
        let n_samples = features.len();
        if n_samples == 0 {
            return;
        }

        let mut effective_weights = vec![1.0; n_samples];
        let mut sum_weights = 0.0;

        for i in 0..n_samples {
            let sw = sample_weights.map(|w| w[i]).unwrap_or(1.0);
            let cw = class_weights.map(|w| w[labels[i]]).unwrap_or(1.0);
            let w = sw * cw;
            effective_weights[i] = w;
            sum_weights += w;
        }

        if sum_weights == 0.0 {
            sum_weights = 1.0;
        }

        let mut x = self.weights.clone();
        x.extend_from_slice(&self.biases);

        let mut g = vec![0.0; x.len()];
        let mut g_new = vec![0.0; x.len()];
        let mut z_scores = vec![0.0; self.n_classes];
        let mut exps = vec![0.0; self.n_classes];

        let mut f = self.evaluate(
            &x,
            features,
            labels,
            &effective_weights,
            sum_weights,
            l2_reg,
            &mut g,
            &mut z_scores,
            &mut exps,
        );

        let mut s_hist: Vec<Vec<f64>> = Vec::with_capacity(m);
        let mut y_hist: Vec<Vec<f64>> = Vec::with_capacity(m);

        let pb = indicatif::ProgressBar::new(max_iter as u64);
        pb.set_style(
            indicatif::ProgressStyle::with_template(
                "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} iter (loss: {msg})",
            )
            .unwrap()
            .progress_chars("#>-"),
        );
        pb.set_message(format!("{:.6}", f));

        for _iter in 0..max_iter {
            if g.iter()
                .copied()
                .fold(0.0f64, |acc, val| acc.max(val.abs()))
                < tol
            {
                break;
            }

            let mut q = g.clone();
            let k = s_hist.len();
            let mut alphas = vec![0.0; k];

            for i in (0..k).rev() {
                let rho = 1.0 / dot(&y_hist[i], &s_hist[i]);
                alphas[i] = rho * dot(&s_hist[i], &q);
                for j in 0..q.len() {
                    q[j] -= alphas[i] * y_hist[i][j];
                }
            }

            if k > 0 {
                let gamma =
                    dot(&s_hist[k - 1], &y_hist[k - 1]) / dot(&y_hist[k - 1], &y_hist[k - 1]);
                for j in 0..q.len() {
                    q[j] *= gamma;
                }
            }

            for i in 0..k {
                let rho = 1.0 / dot(&y_hist[i], &s_hist[i]);
                let beta = rho * dot(&y_hist[i], &q);
                for j in 0..q.len() {
                    q[j] += (alphas[i] - beta) * s_hist[i][j];
                }
            }

            let mut p = vec![0.0; q.len()];
            for j in 0..p.len() {
                p[j] = -q[j];
            }

            let mut g0_dot_p = dot(&g, &p);
            if g0_dot_p >= 0.0 {
                s_hist.clear();
                y_hist.clear();
                for j in 0..p.len() {
                    p[j] = -g[j];
                }
                g0_dot_p = dot(&g, &p);
            }

            let mut alpha = 1.0;
            let mut alpha_min = 0.0;
            let mut alpha_max = f64::INFINITY;
            let c1 = 1e-4;
            let c2 = 0.9;

            let mut x_new = vec![0.0; x.len()];
            let mut f_new = 0.0;

            for _ in 0..20 {
                for j in 0..x.len() {
                    x_new[j] = x[j] + alpha * p[j];
                }

                f_new = self.evaluate(
                    &x_new,
                    features,
                    labels,
                    &effective_weights,
                    sum_weights,
                    l2_reg,
                    &mut g_new,
                    &mut z_scores,
                    &mut exps,
                );

                if f_new > f + c1 * alpha * g0_dot_p {
                    alpha_max = alpha;
                    alpha = 0.5 * (alpha_min + alpha_max);
                } else {
                    let g_new_dot_p = dot(&g_new, &p);
                    if g_new_dot_p < c2 * g0_dot_p {
                        alpha_min = alpha;
                        if alpha_max == f64::INFINITY {
                            alpha *= 2.0;
                        } else {
                            alpha = 0.5 * (alpha_min + alpha_max);
                        }
                    } else {
                        break;
                    }
                }
            }

            let mut s = vec![0.0; x.len()];
            let mut y = vec![0.0; x.len()];
            for j in 0..x.len() {
                s[j] = x_new[j] - x[j];
                y[j] = g_new[j] - g[j];
            }

            if dot(&s, &y) > 1e-10 {
                s_hist.push(s);
                y_hist.push(y);
                if s_hist.len() > m {
                    s_hist.remove(0);
                    y_hist.remove(0);
                }
            }

            x = x_new;
            f = f_new;
            std::mem::swap(&mut g, &mut g_new);
            pb.set_message(format!("{:.6}", f));
            pb.inc(1);
        }

        pb.finish();

        let w_len = self.weights.len();
        self.weights.copy_from_slice(&x[0..w_len]);
        self.biases.copy_from_slice(&x[w_len..]);
    }

    pub fn predict_proba(&self, features: &[(usize, f64)]) -> Vec<f64> {
        let mut z_scores = self.biases.clone();

        for &(feat_idx, val) in features {
            let w_offset = feat_idx * self.n_classes;
            if w_offset + self.n_classes <= self.weights.len() {
                let w_slice = &self.weights[w_offset..w_offset + self.n_classes];
                for k in 0..self.n_classes {
                    z_scores[k] += w_slice[k] * val;
                }
            }
        }

        match self.objective {
            Objective::Binary => {
                vec![sigmoid(z_scores[0])]
            }
            Objective::Multinomial { .. } => {
                let max_z = z_scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let mut sum = 0.0;
                let mut probs = vec![0.0; self.n_classes];

                for k in 0..self.n_classes {
                    let e = (z_scores[k] - max_z).exp();
                    probs[k] = e;
                    sum += e;
                }

                for k in 0..self.n_classes {
                    probs[k] /= sum;
                }

                probs
            }
        }
    }

    pub fn predict(&self, features: &[(usize, f64)]) -> usize {
        let probs = self.predict_proba(features);

        match self.objective {
            Objective::Binary => {
                if probs[0] > 0.5 {
                    1
                } else {
                    0
                }
            }
            Objective::Multinomial { .. } => probs
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0),
        }
    }
}
