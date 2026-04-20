use std::collections::HashSet;
use std::fmt::Write;

use crate::cli::ReportFormat;

pub struct ClassificationReport {
    tp: Vec<usize>,
    fp: Vec<usize>,
    fn_: Vec<usize>,
    support: Vec<usize>,
    total_samples: usize,
    exact_matches: usize,
}

impl ClassificationReport {
    pub fn new(n_classes: usize) -> Self {
        Self {
            tp: vec![0; n_classes],
            fp: vec![0; n_classes],
            fn_: vec![0; n_classes],
            support: vec![0; n_classes],
            total_samples: 0,
            exact_matches: 0,
        }
    }

    pub fn record(&mut self, predicted: &[usize], actual: &[usize]) {
        self.total_samples += 1;

        let max_label = predicted
            .iter()
            .chain(actual.iter())
            .copied()
            .max()
            .unwrap_or(0);
        while self.tp.len() <= max_label {
            self.tp.push(0);
            self.fp.push(0);
            self.fn_.push(0);
            self.support.push(0);
        }

        let pred_set: HashSet<_> = predicted.iter().copied().collect();
        let actual_set: HashSet<_> = actual.iter().copied().collect();

        if pred_set == actual_set {
            self.exact_matches += 1;
        }

        for k in 0..self.tp.len() {
            let pred_has = pred_set.contains(&k);
            let actual_has = actual_set.contains(&k);
            if pred_has && actual_has {
                self.tp[k] += 1;
            }
            if pred_has && !actual_has {
                self.fp[k] += 1;
            }
            if !pred_has && actual_has {
                self.fn_[k] += 1;
            }
            if actual_has {
                self.support[k] += 1;
            }
        }
    }

    fn n_classes(&self) -> usize {
        self.tp.len()
    }

    pub fn precision(&self, k: usize) -> f64 {
        if self.tp[k] + self.fp[k] == 0 {
            0.0
        } else {
            self.tp[k] as f64 / (self.tp[k] + self.fp[k]) as f64
        }
    }

    pub fn recall(&self, k: usize) -> f64 {
        if self.tp[k] + self.fn_[k] == 0 {
            0.0
        } else {
            self.tp[k] as f64 / (self.tp[k] + self.fn_[k]) as f64
        }
    }

    pub fn f1(&self, k: usize) -> f64 {
        let p = self.precision(k);
        let r = self.recall(k);
        if p + r > 0.0 {
            2.0 * p * r / (p + r)
        } else {
            0.0
        }
    }

    pub fn accuracy(&self) -> f64 {
        if self.total_samples == 0 {
            0.0
        } else {
            self.exact_matches as f64 / self.total_samples as f64
        }
    }

    fn macro_avg<F>(&self, metric: F) -> f64
    where
        F: Fn(usize) -> f64,
    {
        let n = self.n_classes();
        if n == 0 {
            return 0.0;
        }
        (0..n).map(|k| metric(k)).sum::<f64>() / n as f64
    }

    fn weighted_avg<F>(&self, metric: F) -> f64
    where
        F: Fn(usize) -> f64,
    {
        let total_support: usize = self.support.iter().sum();
        if total_support == 0 {
            return 0.0;
        }
        (0..self.n_classes())
            .map(|k| metric(k) * self.support[k] as f64)
            .sum::<f64>()
            / total_support as f64
    }

    pub fn format_report(&self, fmt: &ReportFormat) -> String {
        match fmt {
            ReportFormat::Text => self.format_text(),
            ReportFormat::Json => self.format_json(),
        }
    }

    fn format_text(&self) -> String {
        let mut out = String::new();

        writeln!(
            out,
            "{:>14}  {:>9} {:>9} {:>9} {:>9}",
            "", "precision", "recall", "f1-score", "support"
        )
        .unwrap();
        writeln!(out).unwrap();

        for k in 0..self.n_classes() {
            writeln!(
                out,
                "{:>14}  {:>9.2} {:>9.2} {:>9.2} {:>9}",
                k,
                self.precision(k),
                self.recall(k),
                self.f1(k),
                self.support[k]
            )
            .unwrap();
        }

        writeln!(out).unwrap();

        let total = self.total_samples;
        writeln!(
            out,
            "{:>14}  {:>9} {:>9} {:>9.2} {:>9}",
            "accuracy",
            "",
            "",
            self.accuracy(),
            total
        )
        .unwrap();
        writeln!(
            out,
            "{:>14}  {:>9.2} {:>9.2} {:>9.2} {:>9}",
            "macro avg",
            self.macro_avg(|k| self.precision(k)),
            self.macro_avg(|k| self.recall(k)),
            self.macro_avg(|k| self.f1(k)),
            total
        )
        .unwrap();
        writeln!(
            out,
            "{:>14}  {:>9.2} {:>9.2} {:>9.2} {:>9}",
            "weighted avg",
            self.weighted_avg(|k| self.precision(k)),
            self.weighted_avg(|k| self.recall(k)),
            self.weighted_avg(|k| self.f1(k)),
            total
        )
        .unwrap();

        out
    }

    fn format_json(&self) -> String {
        let per_class: Vec<String> = (0..self.n_classes())
            .map(|k| format!(
                "\"{}\":{{\"precision\":{:.4},\"recall\":{:.4},\"f1_score\":{:.4},\"support\":{}}}",
                k, self.precision(k), self.recall(k), self.f1(k), self.support[k]
            ))
            .collect();

        format!(
            "{{\"classes\":{{{}}},\"accuracy\":{:.4},\"macro_avg\":{{\"precision\":{:.4},\"recall\":{:.4},\"f1_score\":{:.4},\"support\":{}}},\"weighted_avg\":{{\"precision\":{:.4},\"recall\":{:.4},\"f1_score\":{:.4},\"support\":{}}},\"total_samples\":{}}}\n",
            per_class.join(","),
            self.accuracy(),
            self.macro_avg(|k| self.precision(k)),
            self.macro_avg(|k| self.recall(k)),
            self.macro_avg(|k| self.f1(k)),
            self.total_samples,
            self.weighted_avg(|k| self.precision(k)),
            self.weighted_avg(|k| self.recall(k)),
            self.weighted_avg(|k| self.f1(k)),
            self.total_samples,
            self.total_samples,
        )
    }
}
