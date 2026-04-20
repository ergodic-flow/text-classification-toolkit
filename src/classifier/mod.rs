mod lbfgs;
mod ova;
mod sgd;

use serde::{Deserialize, Serialize};

pub use lbfgs::LbfgsClassifier;
pub use ova::OvaClassifier;
pub use sgd::SgdClassifier;

#[derive(Serialize, Deserialize, Clone)]
pub enum Objective {
    Binary,
    Multinomial { n_classes: usize },
}

pub trait BinarySolver: Sized {
    fn new(n_features: usize, objective: Objective) -> Self;
    fn predict_proba(&self, features: &[(usize, f64)]) -> Vec<f64>;
}

impl BinarySolver for SgdClassifier {
    fn new(n_features: usize, objective: Objective) -> Self {
        SgdClassifier::new(n_features, objective)
    }
    fn predict_proba(&self, features: &[(usize, f64)]) -> Vec<f64> {
        SgdClassifier::predict_proba(self, features)
    }
}

impl BinarySolver for LbfgsClassifier {
    fn new(n_features: usize, objective: Objective) -> Self {
        LbfgsClassifier::new(n_features, objective)
    }
    fn predict_proba(&self, features: &[(usize, f64)]) -> Vec<f64> {
        LbfgsClassifier::predict_proba(self, features)
    }
}

#[derive(Serialize, Deserialize)]
pub enum Classifier {
    Sgd(SgdClassifier),
    Lbfgs(LbfgsClassifier),
    OvaSgd(OvaClassifier<SgdClassifier>),
    OvaLbfgs(OvaClassifier<LbfgsClassifier>),
}

impl Classifier {
    pub fn predict(&self, features: &[(usize, f64)]) -> usize {
        match self {
            Classifier::Sgd(c) => c.predict(features),
            Classifier::Lbfgs(c) => c.predict(features),
            Classifier::OvaSgd(c) => c.predict(features),
            Classifier::OvaLbfgs(c) => c.predict(features),
        }
    }

    pub fn predict_proba(&self, features: &[(usize, f64)]) -> Vec<f64> {
        match self {
            Classifier::Sgd(c) => c.predict_proba(features),
            Classifier::Lbfgs(c) => c.predict_proba(features),
            Classifier::OvaSgd(c) => c.predict_proba(features),
            Classifier::OvaLbfgs(c) => c.predict_proba(features),
        }
    }

    pub fn n_classes(&self) -> usize {
        match self {
            Classifier::Sgd(c) => c.n_classes,
            Classifier::Lbfgs(c) => c.n_classes,
            Classifier::OvaSgd(c) => c.n_classes,
            Classifier::OvaLbfgs(c) => c.n_classes,
        }
    }

    pub fn is_binary(&self) -> bool {
        matches!(self, Classifier::Sgd(c) if matches!(c.objective, Objective::Binary))
            || matches!(self, Classifier::Lbfgs(c) if matches!(c.objective, Objective::Binary))
    }

    pub fn predict_labels(&self, features: &[(usize, f64)]) -> Vec<usize> {
        match self {
            Classifier::Sgd(c) => vec![c.predict(features)],
            Classifier::Lbfgs(c) => vec![c.predict(features)],
            Classifier::OvaSgd(c) => c.predict_labels(features, 0.5),
            Classifier::OvaLbfgs(c) => c.predict_labels(features, 0.5),
        }
    }

    pub fn feature_contributions(&self, features: &[(usize, f64)]) -> Vec<(usize, f64)> {
        match self {
            Classifier::Sgd(c) => {
                let class = if c.n_classes == 1 {
                    0
                } else {
                    c.predict(features)
                };
                features
                    .iter()
                    .map(|&(feat_idx, val)| {
                        let w_offset = feat_idx * c.n_classes + class;
                        let contrib = if w_offset < c.weights.len() {
                            c.weights[w_offset] * val
                        } else {
                            0.0
                        };
                        (feat_idx, contrib)
                    })
                    .collect()
            }
            Classifier::Lbfgs(c) => {
                let class = if c.n_classes == 1 {
                    0
                } else {
                    c.predict(features)
                };
                features
                    .iter()
                    .map(|&(feat_idx, val)| {
                        let w_offset = feat_idx * c.n_classes + class;
                        let contrib = if w_offset < c.weights.len() {
                            c.weights[w_offset] * val
                        } else {
                            0.0
                        };
                        (feat_idx, contrib)
                    })
                    .collect()
            }
            Classifier::OvaSgd(c) => {
                let pred = c.predict(features);
                let clf = &c.classifiers[pred];
                features
                    .iter()
                    .map(|&(feat_idx, val)| {
                        let contrib = if feat_idx < clf.weights.len() {
                            clf.weights[feat_idx] * val
                        } else {
                            0.0
                        };
                        (feat_idx, contrib)
                    })
                    .collect()
            }
            Classifier::OvaLbfgs(c) => {
                let pred = c.predict(features);
                let clf = &c.classifiers[pred];
                features
                    .iter()
                    .map(|&(feat_idx, val)| {
                        let contrib = if feat_idx < clf.weights.len() {
                            clf.weights[feat_idx] * val
                        } else {
                            0.0
                        };
                        (feat_idx, contrib)
                    })
                    .collect()
            }
        }
    }
}
