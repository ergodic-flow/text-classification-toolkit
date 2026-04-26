pub mod calibration;
pub mod classifier;
pub mod tfidf;

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
        self.classifier.predict(&features)
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
