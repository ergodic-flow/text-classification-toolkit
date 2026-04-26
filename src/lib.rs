pub mod calibration;
pub mod classifier;
pub mod tfidf;

use std::cmp::Ordering;
use std::fs;

use classifier::Classifier;
use pyo3::prelude::*;
use tfidf::TfIdf;

#[derive(serde::Serialize, serde::Deserialize)]
pub struct Model {
    pub tfidf: TfIdf,
    pub classifier: Classifier,
    pub calibrator: Option<calibration::PlattCalibrator>,
}

impl Model {
    pub fn predict(&self, text: &str) -> usize {
        let features = self.tfidf.transform(text);
        self.predict_features(&features)
    }

    pub fn predict_features(&self, features: &[(usize, f64)]) -> usize {
        self.predict_features_with_proba(features).0
    }

    pub fn predict_features_with_proba(&self, features: &[(usize, f64)]) -> (usize, Vec<f64>) {
        let probs = self.predict_features_proba(features);
        let pred = predict_from_proba(&probs, self.classifier.is_binary());
        (pred, probs)
    }

    pub fn predict_proba(&self, text: &str) -> Vec<f64> {
        let features = self.tfidf.transform(text);
        self.predict_features_proba(&features)
    }

    pub fn predict_features_proba(&self, features: &[(usize, f64)]) -> Vec<f64> {
        let raw = self.classifier.predict_proba(features);
        match &self.calibrator {
            Some(cal) => cal.transform(&raw),
            None => raw,
        }
    }

    pub fn predict_features_labels(&self, features: &[(usize, f64)]) -> Vec<usize> {
        let probs = self.predict_features_proba(features);
        match &self.classifier {
            Classifier::OvaSgd(_) | Classifier::OvaLbfgs(_) => probs
                .iter()
                .enumerate()
                .filter(|&(_, &prob)| prob >= 0.5)
                .map(|(idx, _)| idx)
                .collect(),
            _ => vec![predict_from_proba(&probs, self.classifier.is_binary())],
        }
    }
}

fn predict_from_proba(probs: &[f64], is_binary: bool) -> usize {
    if is_binary {
        if probs.first().copied().unwrap_or(0.0) > 0.5 {
            1
        } else {
            0
        }
    } else {
        probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0)
    }
}

pub fn load_model(path: &str) -> Result<Model, Box<dyn std::error::Error>> {
    let bytes = fs::read(path)?;
    Ok(bincode::deserialize::<Model>(&bytes)?)
}

#[pyclass(name = "Model", module = "text_toolkit")]
struct PyModel {
    model: Model,
}

#[pymethods]
impl PyModel {
    #[new]
    fn new(path: &str) -> PyResult<Self> {
        let model =
            load_model(path).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self { model })
    }

    fn predict(&self, text: &str) -> usize {
        self.model.predict(text)
    }

    fn predict_proba(&self, text: &str) -> Vec<f64> {
        self.model.predict_proba(text)
    }

    fn n_classes(&self) -> usize {
        self.model.classifier.n_classes()
    }

    fn is_binary(&self) -> bool {
        self.model.classifier.is_binary()
    }

    fn is_calibrated(&self) -> bool {
        self.model.calibrator.is_some()
    }
}

#[pymodule]
fn text_toolkit(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyModel>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::classifier::{Objective, SgdClassifier};

    #[test]
    fn binary_predict_uses_calibrated_probability() {
        let mut classifier = SgdClassifier::new(0, Objective::Binary);
        classifier.biases[0] = 4.0;

        let mut calibrator = calibration::PlattCalibrator::new(1);
        calibrator.fit_binary(&[0.9, 0.8, 0.2, 0.1], &[0, 0, 1, 1]);

        let model = Model {
            tfidf: TfIdf::new(),
            classifier: Classifier::Sgd(classifier),
            calibrator: Some(calibrator),
        };

        let probs = model.predict_proba("");
        assert!(probs[0] < 0.5);
        assert_eq!(model.predict(""), 0);
    }

    #[test]
    fn multiclass_predict_uses_calibrated_argmax() {
        let mut classifier = SgdClassifier::new(0, Objective::Multinomial { n_classes: 2 });
        classifier.biases[0] = 4.0;

        let mut calibrator = calibration::PlattCalibrator::new(2);
        calibrator.fit_multiclass(
            &[
                vec![0.9, 0.1],
                vec![0.8, 0.2],
                vec![0.2, 0.8],
                vec![0.1, 0.9],
            ],
            &[1, 1, 0, 0],
        );

        let model = Model {
            tfidf: TfIdf::new(),
            classifier: Classifier::Sgd(classifier),
            calibrator: Some(calibrator),
        };

        let probs = model.predict_proba("");
        assert!(probs[1] > probs[0]);
        assert_eq!(model.predict(""), 1);
    }
}
