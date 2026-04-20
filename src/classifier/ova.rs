use serde::{Deserialize, Serialize};

use super::{BinarySolver, LbfgsClassifier, Objective, SgdClassifier};

#[derive(Serialize, Deserialize)]
pub struct OvaClassifier<C> {
    pub classifiers: Vec<C>,
    pub n_features: usize,
    pub n_classes: usize,
}

impl<C: BinarySolver> OvaClassifier<C> {
    pub fn new(n_features: usize, n_classes: usize) -> Self {
        let classifiers = (0..n_classes)
            .map(|_| C::new(n_features, Objective::Binary))
            .collect();
        Self {
            classifiers,
            n_features,
            n_classes,
        }
    }

    pub fn predict_proba(&self, features: &[(usize, f64)]) -> Vec<f64> {
        self.classifiers
            .iter()
            .map(|clf| clf.predict_proba(features)[0])
            .collect()
    }

    pub fn predict(&self, features: &[(usize, f64)]) -> usize {
        self.predict_proba(features)
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0)
    }

    pub fn predict_labels(&self, features: &[(usize, f64)], threshold: f64) -> Vec<usize> {
        self.predict_proba(features)
            .iter()
            .enumerate()
            .filter(|&(_, &prob)| prob >= threshold)
            .map(|(idx, _)| idx)
            .collect()
    }
}

impl OvaClassifier<SgdClassifier> {
    pub fn train(
        &mut self,
        features: &[Vec<(usize, f64)>],
        labels: &[Vec<usize>],
        class_weights: Option<&[f64]>,
        balanced: bool,
        sample_weights: Option<&[f64]>,
        l2_reg: f64,
        lr: f64,
        epochs: usize,
        batch_size: usize,
    ) {
        let mut binary_labels = vec![0usize; features.len()];
        for (k, clf) in self.classifiers.iter_mut().enumerate() {
            for (i, sample_labels) in labels.iter().enumerate() {
                binary_labels[i] = if sample_labels.contains(&k) { 1 } else { 0 };
            }

            let binary_cw = if balanced {
                let n = binary_labels.len() as f64;
                let count_0 = binary_labels.iter().filter(|&&l| l == 0).count() as f64;
                let count_1 = n - count_0;
                let w0 = if count_0 > 0.0 {
                    n / (2.0 * count_0)
                } else {
                    0.0
                };
                let w1 = if count_1 > 0.0 {
                    n / (2.0 * count_1)
                } else {
                    0.0
                };
                Some(vec![w0, w1])
            } else {
                class_weights.map(|cw| vec![1.0, cw[k]])
            };

            clf.train(
                features,
                &binary_labels,
                sample_weights,
                binary_cw.as_deref(),
                l2_reg,
                lr,
                epochs,
                batch_size,
            );
        }
    }
}

impl OvaClassifier<LbfgsClassifier> {
    pub fn train(
        &mut self,
        features: &[Vec<(usize, f64)>],
        labels: &[Vec<usize>],
        class_weights: Option<&[f64]>,
        balanced: bool,
        sample_weights: Option<&[f64]>,
        l2_reg: f64,
        m: usize,
        max_iter: usize,
        tol: f64,
    ) {
        let mut binary_labels = vec![0usize; features.len()];
        for (k, clf) in self.classifiers.iter_mut().enumerate() {
            for (i, sample_labels) in labels.iter().enumerate() {
                binary_labels[i] = if sample_labels.contains(&k) { 1 } else { 0 };
            }

            let binary_cw = if balanced {
                let n = binary_labels.len() as f64;
                let count_0 = binary_labels.iter().filter(|&&l| l == 0).count() as f64;
                let count_1 = n - count_0;
                let w0 = if count_0 > 0.0 {
                    n / (2.0 * count_0)
                } else {
                    0.0
                };
                let w1 = if count_1 > 0.0 {
                    n / (2.0 * count_1)
                } else {
                    0.0
                };
                Some(vec![w0, w1])
            } else {
                class_weights.map(|cw| vec![1.0, cw[k]])
            };

            clf.train(
                features,
                &binary_labels,
                sample_weights,
                binary_cw.as_deref(),
                l2_reg,
                m,
                max_iter,
                tol,
            );
        }
    }
}
