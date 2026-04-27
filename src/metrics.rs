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
    sample_precision_sum: f64,
    sample_recall_sum: f64,
    sample_f1_sum: f64,
}

#[cfg(test)]
mod tests {
    use super::ClassificationReport;
    use crate::cli::ReportFormat;

    #[test]
    fn text_report_includes_micro_and_samples_averages() {
        let mut report = ClassificationReport::new(3);
        report.record(&[0, 1], &[1, 2]);
        report.record(&[2], &[2]);
        report.record(&[], &[0]);

        let text = report.format_report(&ReportFormat::Text, false);

        assert!(text.contains("     micro avg       0.67      0.50      0.57         4"));
        assert!(text.contains("   samples avg       0.50      0.50      0.50         4"));
        assert!(!text.contains("accuracy"));
    }

    #[test]
    fn json_report_includes_micro_and_samples_averages() {
        let mut report = ClassificationReport::new(3);
        report.record(&[0, 1], &[1, 2]);
        report.record(&[2], &[2]);
        report.record(&[], &[0]);

        let json = report.format_report(&ReportFormat::Json, false);

        assert!(json.contains(
            "\"micro_avg\":{\"precision\":0.6667,\"recall\":0.5000,\"f1_score\":0.5714,\"support\":4}"
        ));
        assert!(json.contains(
            "\"samples_avg\":{\"precision\":0.5000,\"recall\":0.5000,\"f1_score\":0.5000,\"support\":4}"
        ));
    }

    #[test]
    fn aggregate_support_counts_labels_not_samples() {
        let mut report = ClassificationReport::new(3);
        report.record(&[0, 1], &[0, 1]);
        report.record(&[2], &[2]);

        let text = report.format_report(&ReportFormat::Text, true);
        let json = report.format_report(&ReportFormat::Json, true);

        assert!(text.contains("     micro avg       1.00      1.00      1.00         3"));
        assert!(text.contains("   samples avg       1.00      1.00      1.00         3"));
        assert!(json.contains(
            "\"micro_avg\":{\"precision\":1.0000,\"recall\":1.0000,\"f1_score\":1.0000,\"support\":3}"
        ));
        assert!(json.contains("\"total_samples\":2"));
    }

    #[test]
    fn multilabel_text_report_includes_subset_accuracy_and_hamming_loss() {
        let mut report = ClassificationReport::new(3);
        report.record(&[0, 1], &[1, 2]);
        report.record(&[2], &[2]);
        report.record(&[], &[0]);

        let text = report.format_report(&ReportFormat::Text, true);

        assert!(text.contains("\nhamming_loss:      0.3333\nsubset_accuracy:   0.3333\n"));
        assert!(!text.contains("accuracy                           "));
    }

    #[test]
    fn multilabel_json_report_includes_subset_accuracy_and_hamming_loss() {
        let mut report = ClassificationReport::new(3);
        report.record(&[0, 1], &[1, 2]);
        report.record(&[2], &[2]);
        report.record(&[], &[0]);

        let json = report.format_report(&ReportFormat::Json, true);

        assert!(json.contains("\"subset_accuracy\":0.3333"));
        assert!(json.contains("\"hamming_loss\":0.3333"));
    }
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
            sample_precision_sum: 0.0,
            sample_recall_sum: 0.0,
            sample_f1_sum: 0.0,
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

        let intersection = pred_set.intersection(&actual_set).count();
        let sample_precision = if pred_set.is_empty() {
            0.0
        } else {
            intersection as f64 / pred_set.len() as f64
        };
        let sample_recall = if actual_set.is_empty() {
            0.0
        } else {
            intersection as f64 / actual_set.len() as f64
        };
        let sample_f1 = if sample_precision + sample_recall > 0.0 {
            2.0 * sample_precision * sample_recall / (sample_precision + sample_recall)
        } else {
            0.0
        };
        self.sample_precision_sum += sample_precision;
        self.sample_recall_sum += sample_recall;
        self.sample_f1_sum += sample_f1;

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

    fn subset_accuracy(&self) -> f64 {
        self.accuracy()
    }

    fn hamming_loss(&self) -> f64 {
        if self.total_samples == 0 || self.n_classes() == 0 {
            return 0.0;
        }

        let mismatches: usize = self
            .fp
            .iter()
            .zip(self.fn_.iter())
            .map(|(fp, fn_)| fp + fn_)
            .sum();
        mismatches as f64 / (self.total_samples * self.n_classes()) as f64
    }

    fn micro_precision(&self) -> f64 {
        let tp: usize = self.tp.iter().sum();
        let fp: usize = self.fp.iter().sum();
        if tp + fp == 0 {
            0.0
        } else {
            tp as f64 / (tp + fp) as f64
        }
    }

    fn micro_recall(&self) -> f64 {
        let tp: usize = self.tp.iter().sum();
        let fn_: usize = self.fn_.iter().sum();
        if tp + fn_ == 0 {
            0.0
        } else {
            tp as f64 / (tp + fn_) as f64
        }
    }

    fn micro_f1(&self) -> f64 {
        let p = self.micro_precision();
        let r = self.micro_recall();
        if p + r > 0.0 {
            2.0 * p * r / (p + r)
        } else {
            0.0
        }
    }

    fn samples_avg_precision(&self) -> f64 {
        if self.total_samples == 0 {
            0.0
        } else {
            self.sample_precision_sum / self.total_samples as f64
        }
    }

    fn samples_avg_recall(&self) -> f64 {
        if self.total_samples == 0 {
            0.0
        } else {
            self.sample_recall_sum / self.total_samples as f64
        }
    }

    fn samples_avg_f1(&self) -> f64 {
        if self.total_samples == 0 {
            0.0
        } else {
            self.sample_f1_sum / self.total_samples as f64
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

    fn total_support(&self) -> usize {
        self.support.iter().sum()
    }

    pub fn format_report(&self, fmt: &ReportFormat, include_multilabel_metrics: bool) -> String {
        match fmt {
            ReportFormat::Text => self.format_text(include_multilabel_metrics),
            ReportFormat::Json => self.format_json(include_multilabel_metrics),
        }
    }

    fn format_text(&self, include_multilabel_metrics: bool) -> String {
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

        let total = self.total_support();
        writeln!(
            out,
            "{:>14}  {:>9.2} {:>9.2} {:>9.2} {:>9}",
            "micro avg",
            self.micro_precision(),
            self.micro_recall(),
            self.micro_f1(),
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
        writeln!(
            out,
            "{:>14}  {:>9.2} {:>9.2} {:>9.2} {:>9}",
            "samples avg",
            self.samples_avg_precision(),
            self.samples_avg_recall(),
            self.samples_avg_f1(),
            total
        )
        .unwrap();

        if include_multilabel_metrics {
            writeln!(out).unwrap();
            writeln!(out, "{:<16} {:>8.4}", "hamming_loss:", self.hamming_loss()).unwrap();
            writeln!(
                out,
                "{:<16} {:>8.4}",
                "subset_accuracy:",
                self.subset_accuracy()
            )
            .unwrap();
        }

        out
    }

    fn format_json(&self, include_multilabel_metrics: bool) -> String {
        let per_class: Vec<String> = (0..self.n_classes())
            .map(|k| format!(
                "\"{}\":{{\"precision\":{:.4},\"recall\":{:.4},\"f1_score\":{:.4},\"support\":{}}}",
                k, self.precision(k), self.recall(k), self.f1(k), self.support[k]
            ))
            .collect();

        let multilabel_metrics = if include_multilabel_metrics {
            format!(
                ",\"subset_accuracy\":{:.4},\"hamming_loss\":{:.4}",
                self.subset_accuracy(),
                self.hamming_loss()
            )
        } else {
            String::new()
        };

        format!(
            "{{\"classes\":{{{}}},\"accuracy\":{:.4}{},\"micro_avg\":{{\"precision\":{:.4},\"recall\":{:.4},\"f1_score\":{:.4},\"support\":{}}},\"macro_avg\":{{\"precision\":{:.4},\"recall\":{:.4},\"f1_score\":{:.4},\"support\":{}}},\"weighted_avg\":{{\"precision\":{:.4},\"recall\":{:.4},\"f1_score\":{:.4},\"support\":{}}},\"samples_avg\":{{\"precision\":{:.4},\"recall\":{:.4},\"f1_score\":{:.4},\"support\":{}}},\"total_samples\":{}}}\n",
            per_class.join(","),
            self.accuracy(),
            multilabel_metrics,
            self.micro_precision(),
            self.micro_recall(),
            self.micro_f1(),
            self.total_support(),
            self.macro_avg(|k| self.precision(k)),
            self.macro_avg(|k| self.recall(k)),
            self.macro_avg(|k| self.f1(k)),
            self.total_support(),
            self.weighted_avg(|k| self.precision(k)),
            self.weighted_avg(|k| self.recall(k)),
            self.weighted_avg(|k| self.f1(k)),
            self.total_support(),
            self.samples_avg_precision(),
            self.samples_avg_recall(),
            self.samples_avg_f1(),
            self.total_support(),
            self.total_samples,
        )
    }
}
