use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};

use super::Objective;

#[derive(Serialize, Deserialize)]
pub struct SgdClassifier {
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

impl SgdClassifier {
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

    pub fn train(
        &mut self,
        features: &[Vec<(usize, f64)>],
        labels: &[usize],
        sample_weights: Option<&[f64]>,
        class_weights: Option<&[f64]>,
        l2_reg: f64, // <-- Added L2 penalty parameter
        lr: f64,
        epochs: usize,
        batch_size: usize,
    ) {
        if features.is_empty() || batch_size == 0 {
            return;
        }

        let n = features.len();

        if let Some(sw) = sample_weights {
            assert_eq!(
                sw.len(),
                n,
                "sample_weights length must match features length"
            );
        }

        if let Some(cw) = class_weights {
            let expected_len = match self.objective {
                Objective::Binary => 2,
                Objective::Multinomial { n_classes } => n_classes,
            };
            assert_eq!(
                cw.len(),
                expected_len,
                "class_weights length must match the number of possible classes"
            );
        }

        let mut indices: Vec<usize> = (0..n).collect();
        let mut rng = rand::rng();
        let n_classes = self.n_classes;

        let mut batch_errors = vec![0.0; batch_size * n_classes];
        let mut grad_b = vec![0.0; n_classes];
        let mut z_scores = vec![0.0; n_classes];

        let pb = indicatif::ProgressBar::new(epochs as u64);
        pb.set_style(
            indicatif::ProgressStyle::with_template(
                "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} epochs",
            )
            .unwrap()
            .progress_chars("#>-"),
        );

        for epoch in 0..epochs {
            indices.shuffle(&mut rng);

            for chunk in indices.chunks(batch_size) {
                let lr_scaled = lr / chunk.len() as f64;
                grad_b.fill(0.0);

                // --- Apply L2 Regularization (Weight Decay) ---
                // We do this once per batch. We ensure the decay factor doesn't flip signs
                // if a user accidentally passes a massive learning rate or l2_reg.
                if l2_reg > 0.0 {
                    let decay = (1.0 - lr * l2_reg).max(0.0);
                    for w in &mut self.weights {
                        *w *= decay;
                    }
                }

                for (idx_in_chunk, &i) in chunk.iter().enumerate() {
                    let feat = &features[i];
                    let true_class = labels[i];

                    // Calculate effective weight for this observation
                    let weight = match (sample_weights, class_weights) {
                        (Some(sw), Some(cw)) => sw[i] * cw[true_class],
                        (Some(sw), None) => sw[i],
                        (None, Some(cw)) => cw[true_class],
                        (None, None) => 1.0,
                    };

                    z_scores.copy_from_slice(&self.biases);

                    for &(feat_idx, val) in feat {
                        let w_offset = feat_idx * n_classes;
                        if w_offset + n_classes <= self.weights.len() {
                            let w_slice = &self.weights[w_offset..w_offset + n_classes];
                            for k in 0..n_classes {
                                z_scores[k] += w_slice[k] * val;
                            }
                        }
                    }

                    let err_offset = idx_in_chunk * n_classes;
                    let err_slice = &mut batch_errors[err_offset..err_offset + n_classes];

                    match self.objective {
                        Objective::Binary => {
                            let err = (sigmoid(z_scores[0]) - (true_class as f64)) * weight;
                            err_slice[0] = err;
                            grad_b[0] += err;
                        }
                        Objective::Multinomial { .. } => {
                            let max_z = z_scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                            let mut sum = 0.0;

                            for k in 0..n_classes {
                                let e = (z_scores[k] - max_z).exp();
                                err_slice[k] = e;
                                sum += e;
                            }

                            for k in 0..n_classes {
                                err_slice[k] /= sum;
                            }
                            err_slice[true_class] -= 1.0;

                            for k in 0..n_classes {
                                err_slice[k] *= weight;
                                grad_b[k] += err_slice[k];
                            }
                        }
                    }
                }

                for k in 0..n_classes {
                    self.biases[k] -= lr_scaled * grad_b[k];
                }

                for (idx_in_chunk, &i) in chunk.iter().enumerate() {
                    let err_offset = idx_in_chunk * n_classes;
                    let errs = &batch_errors[err_offset..err_offset + n_classes];

                    for &(feat_idx, val) in &features[i] {
                        let w_offset = feat_idx * n_classes;

                        if w_offset + n_classes <= self.weights.len() {
                            let w_slice = &mut self.weights[w_offset..w_offset + n_classes];
                            for k in 0..n_classes {
                                w_slice[k] -= lr_scaled * errs[k] * val;
                            }
                        }
                    }
                }
            }
            pb.inc(1);
        }
        pb.finish();
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
