mod cli;
mod metrics;
//mod iterstrat; // UNUSED FOR NOW LEAVE IT ALONE

use std::fs;
use std::io::{BufRead, Write};

use clap::Parser;
use rand::rng;
use rand::seq::SliceRandom;

use metrics::ClassificationReport;
use text_toolkit::classifier::{
    Classifier, LbfgsClassifier, Objective, OvaClassifier, SgdClassifier,
};
use text_toolkit::tfidf::TfIdf;
use text_toolkit::{Model, calibration, load_model as try_load_model};

fn load_model(path: &str) -> Model {
    try_load_model(path).unwrap_or_else(|e| {
        eprintln!("error loading model '{}': {}", path, e);
        std::process::exit(1);
    })
}

fn parse_labels(label: &str, binary: bool) -> Vec<usize> {
    label
        .split(',')
        .map(|l| {
            let v: usize = l.trim().parse().unwrap_or_else(|_| {
                eprintln!(
                    "error: invalid label '{}': expected non-negative integer",
                    l.trim()
                );
                std::process::exit(1);
            });
            if binary && v > 1 {
                eprintln!(
                    "error: invalid label '{}': expected 0 or 1 for binary classification",
                    v
                );
                std::process::exit(1);
            }
            v
        })
        .collect()
}

fn parse_label(label: &str, binary: bool) -> usize {
    let labels = parse_labels(label, binary);
    if labels.len() != 1 {
        eprintln!("error: expected exactly 1 label, got {}", labels.len());
        std::process::exit(1);
    }
    labels[0]
}

fn build_objective(model_type: &cli::ModelType, labels: &[usize]) -> Objective {
    match model_type {
        cli::ModelType::Binary => Objective::Binary,
        cli::ModelType::Multinomial => {
            let n_classes = labels.iter().copied().max().unwrap_or(0) + 1;
            Objective::Multinomial { n_classes }
        }
        cli::ModelType::Ova => {
            eprintln!("error: OvA (one-vs-all) is not yet implemented");
            std::process::exit(1);
        }
    }
}

fn ngram_range(ngrams: &cli::NgramRange) -> (usize, usize) {
    match ngrams {
        cli::NgramRange::Unigrams => (1, 1),
        cli::NgramRange::UnigramsBigrams => (1, 2),
        cli::NgramRange::Bigrams => (2, 2),
    }
}

fn compute_balanced_class_weights(labels: &[usize], n_classes: usize) -> Vec<f64> {
    let n_samples = labels.len() as f64;
    let mut counts = vec![0usize; n_classes];
    for &l in labels {
        counts[l] += 1;
    }
    (0..n_classes)
        .map(|k| {
            if counts[k] == 0 {
                0.0
            } else {
                n_samples / (n_classes as f64 * counts[k] as f64)
            }
        })
        .collect()
}

fn load_class_weights_file(path: &str, n_classes: usize) -> Vec<f64> {
    let content = fs::read_to_string(path).unwrap_or_else(|e| {
        eprintln!("error reading class weights file '{}': {}", path, e);
        std::process::exit(1);
    });
    let weights: Vec<f64> = content
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            l.trim().parse::<f64>().unwrap_or_else(|e| {
                eprintln!("error parsing class weight '{}': {}", l.trim(), e);
                std::process::exit(1);
            })
        })
        .collect();
    if weights.len() != n_classes {
        eprintln!(
            "error: expected {} class weights, got {}",
            n_classes,
            weights.len()
        );
        std::process::exit(1);
    }
    weights
}

fn resolve_class_weights(
    class_weight: &Option<String>,
    labels: &[usize],
    n_classes: usize,
) -> Option<Vec<f64>> {
    let val = class_weight.as_ref()?;
    if val == "balanced" {
        Some(compute_balanced_class_weights(labels, n_classes))
    } else {
        Some(load_class_weights_file(val, n_classes))
    }
}

fn read_sample_weights(path: &str, n_samples: usize) -> Vec<f64> {
    let content = fs::read_to_string(path).unwrap_or_else(|e| {
        eprintln!("error reading sample weights file '{}': {}", path, e);
        std::process::exit(1);
    });
    let weights: Vec<f64> = content
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            l.trim().parse::<f64>().unwrap_or_else(|e| {
                eprintln!("error parsing sample weight '{}': {}", l.trim(), e);
                std::process::exit(1);
            })
        })
        .collect();
    if weights.len() != n_samples {
        eprintln!(
            "error: expected {} sample weights, got {}",
            n_samples,
            weights.len()
        );
        std::process::exit(1);
    }
    weights
}

fn build_tfidf(args: &cli::ModelArgs) -> TfIdf {
    let (min_n, max_n) = ngram_range(&args.ngrams);
    let mut tfidf = TfIdf::new().min_n(min_n).max_n(max_n);
    if let Some(max_f) = args.max_features {
        tfidf = tfidf.max_features(max_f);
    }
    tfidf
}

fn read_tsv(path: &str) -> Vec<(String, String)> {
    let file = fs::File::open(path).unwrap_or_else(|e| {
        eprintln!("error opening {}: {}", path, e);
        std::process::exit(1);
    });
    let mut rows = Vec::new();
    let mut first = true;
    for line in std::io::BufReader::new(file).lines() {
        let line = line.unwrap();
        if first {
            first = false;
            continue;
        }
        let mut parts = line.splitn(2, '\t');
        let label = parts.next().unwrap_or("").to_string();
        let text = parts.next().unwrap_or("").to_string();
        rows.push((label, text));
    }
    rows
}

fn do_calibrate(args: &cli::CalibrateArgs) {
    let model = load_model(&args.model);

    let data = read_tsv(&args.input);
    if data.is_empty() {
        eprintln!("error: calibration set is empty");
        std::process::exit(1);
    }

    let is_binary = model.classifier.is_binary();
    let n_classes = model.classifier.n_classes();
    let is_ova = matches!(
        model.classifier,
        Classifier::OvaSgd(_) | Classifier::OvaLbfgs(_)
    );

    let mut calibrator = calibration::PlattCalibrator::new(n_classes);

    let all_scores: Vec<Vec<f64>> = data
        .iter()
        .map(|(_, text)| model.classifier.predict_proba(&model.tfidf.transform(text)))
        .collect();

    if is_ova {
        let all_labels: Vec<Vec<usize>> =
            data.iter().map(|(l, _)| parse_labels(l, false)).collect();
        calibrator.fit_multilabel(&all_scores, &all_labels);
    } else {
        let all_labels: Vec<usize> = data
            .iter()
            .map(|(l, _)| parse_label(l, is_binary))
            .collect();

        if n_classes == 1 {
            let scores: Vec<f64> = all_scores.iter().map(|s| s[0]).collect();
            calibrator.fit_binary(&scores, &all_labels);
        } else {
            calibrator.fit_multiclass(&all_scores, &all_labels);
        }
    }

    let calibrated_model = Model {
        tfidf: model.tfidf,
        classifier: model.classifier,
        calibrator: Some(calibrator),
    };

    let bytes = bincode::serialize(&calibrated_model).unwrap();
    fs::write(&args.output, &bytes).unwrap();

    eprintln!(
        "calibrated: {} samples, {} classes -> {}",
        data.len(),
        n_classes,
        args.output,
    );
}

fn do_train(args: &cli::TrainArgs) {
    let mut data = read_tsv(&args.model.input);

    let mut sample_weights = args
        .model
        .sample_weight
        .as_ref()
        .map(|path| read_sample_weights(path, data.len()));

    let mut rng = rng();
    if let Some(ref mut sw) = sample_weights {
        let mut combined: Vec<((String, String), f64)> =
            data.into_iter().zip(sw.iter().copied()).collect();
        combined.shuffle(&mut rng);
        let (d, w): (Vec<_>, Vec<_>) = combined.into_iter().unzip();
        data = d;
        *sw = w;
    } else {
        data.shuffle(&mut rng);
    }

    let train_texts: Vec<&str> = data.iter().map(|(_, t)| t.as_str()).collect();
    let is_binary = matches!(&args.model.model_type, cli::ModelType::Binary);

    let mut tfidf = build_tfidf(&args.model);
    tfidf.fit(&train_texts);

    let features: Vec<Vec<(usize, f64)>> = train_texts.iter().map(|t| tfidf.transform(t)).collect();

    let classifier = match (&args.model.model_type, &args.model.solver) {
        (cli::ModelType::Ova, cli::Solver::Sgd) => {
            let labels: Vec<Vec<usize>> =
                data.iter().map(|(l, _)| parse_labels(l, false)).collect();
            let n_classes = labels
                .iter()
                .flat_map(|v| v.iter().copied())
                .max()
                .unwrap_or(0)
                + 1;
            let balanced = args.model.class_weight.as_deref() == Some("balanced");
            let file_cw = match args.model.class_weight.as_deref() {
                Some(s) if s != "balanced" => Some(load_class_weights_file(s, n_classes)),
                _ => None,
            };
            let mut ova: OvaClassifier<SgdClassifier> =
                OvaClassifier::new(tfidf.vocab_size(), n_classes);
            ova.train(
                &features,
                &labels,
                file_cw.as_deref(),
                balanced,
                sample_weights.as_deref(),
                args.l2_reg,
                args.sgd_learning_rate,
                args.sgd_epochs,
                args.sgd_batch_size,
            );
            Classifier::OvaSgd(ova)
        }
        (cli::ModelType::Ova, cli::Solver::Lbfgs) => {
            let labels: Vec<Vec<usize>> =
                data.iter().map(|(l, _)| parse_labels(l, false)).collect();
            let n_classes = labels
                .iter()
                .flat_map(|v| v.iter().copied())
                .max()
                .unwrap_or(0)
                + 1;
            let balanced = args.model.class_weight.as_deref() == Some("balanced");
            let file_cw = match args.model.class_weight.as_deref() {
                Some(s) if s != "balanced" => Some(load_class_weights_file(s, n_classes)),
                _ => None,
            };
            let mut ova: OvaClassifier<LbfgsClassifier> =
                OvaClassifier::new(tfidf.vocab_size(), n_classes);
            ova.train(
                &features,
                &labels,
                file_cw.as_deref(),
                balanced,
                sample_weights.as_deref(),
                args.l2_reg,
                args.lbfgs_memory,
                args.lbfgs_max_iter,
                args.lbfgs_tol,
            );
            Classifier::OvaLbfgs(ova)
        }
        (_, cli::Solver::Sgd) => {
            let labels: Vec<usize> = data
                .iter()
                .map(|(l, _)| parse_label(l, is_binary))
                .collect();
            let objective = build_objective(&args.model.model_type, &labels);
            let n_weight_classes = labels.iter().copied().max().unwrap_or(0) + 1;
            let cw = resolve_class_weights(&args.model.class_weight, &labels, n_weight_classes);
            let mut clf = SgdClassifier::new(tfidf.vocab_size(), objective);
            clf.train(
                &features,
                &labels,
                sample_weights.as_deref(),
                cw.as_deref(),
                args.l2_reg,
                args.sgd_learning_rate,
                args.sgd_epochs,
                args.sgd_batch_size,
            );
            Classifier::Sgd(clf)
        }
        (_, cli::Solver::Lbfgs) => {
            let labels: Vec<usize> = data
                .iter()
                .map(|(l, _)| parse_label(l, is_binary))
                .collect();
            let objective = build_objective(&args.model.model_type, &labels);
            let n_weight_classes = labels.iter().copied().max().unwrap_or(0) + 1;
            let cw = resolve_class_weights(&args.model.class_weight, &labels, n_weight_classes);
            let mut clf = LbfgsClassifier::new(tfidf.vocab_size(), objective);
            clf.train(
                &features,
                &labels,
                sample_weights.as_deref(),
                cw.as_deref(),
                args.l2_reg,
                args.lbfgs_memory,
                args.lbfgs_max_iter,
                args.lbfgs_tol,
            );
            Classifier::Lbfgs(clf)
        }
    };

    let model = Model {
        tfidf,
        classifier,
        calibrator: None,
    };
    let bytes = bincode::serialize(&model).unwrap();
    fs::write(&args.model.output, &bytes).unwrap();

    eprintln!(
        "trained: {} samples, {} features -> {}",
        data.len(),
        model.tfidf.vocab_size(),
        args.model.output,
    );
}

fn do_predict(args: &cli::PredictArgs) {
    let model = load_model(&args.model);

    let input = fs::File::open(&args.input).unwrap_or_else(|e| {
        eprintln!("error opening {}: {}", args.input, e);
        std::process::exit(1);
    });

    let out: Box<dyn Write> = match &args.output {
        Some(path) => Box::new(fs::File::create(path).unwrap()),
        None => Box::new(std::io::stdout()),
    };

    let mut writer = std::io::BufWriter::new(out);

    let has_calibrator = model.calibrator.is_some();

    writeln!(writer, "label\tprobability").unwrap();

    let mut n_samples = 0;
    for line in std::io::BufReader::new(input).lines() {
        let text = line.unwrap();
        let features = model.tfidf.transform(&text);
        let predictions = model.predict_features_labels_with_proba(&features);
        let labels: Vec<String> = predictions
            .iter()
            .map(|(label, _)| label.to_string())
            .collect();
        let probabilities: Vec<String> = predictions
            .iter()
            .map(|(_, probability)| format!("{:.6}", probability))
            .collect();

        writeln!(writer, "{}\t{}", labels.join(","), probabilities.join(",")).unwrap();
        n_samples += 1;
    }

    eprintln!(
        "predicted: {} samples from {}{}",
        n_samples,
        args.model,
        if has_calibrator { " [calibrated]" } else { "" },
    );
}

fn do_evaluate(args: &cli::EvaluateArgs) {
    let model = load_model(&args.model);

    let data = read_tsv(&args.input);

    let is_binary = model.classifier.is_binary();
    let is_multilabel = matches!(
        model.classifier,
        Classifier::OvaSgd(_) | Classifier::OvaLbfgs(_)
    );
    let n_classes = model.classifier.n_classes();
    let mut report = ClassificationReport::new(n_classes);

    for (label, text) in &data {
        let features = model.tfidf.transform(text);
        let predicted = model.predict_features_labels(&features);
        let actual = parse_labels(label, is_binary);
        report.record(&predicted, &actual);
    }

    let report = report.format_report(&args.format, is_multilabel);

    match &args.report {
        Some(path) => fs::write(path, &report).unwrap(),
        None => print!("{}", report),
    }

    eprintln!("evaluated: {} samples from {}", data.len(), args.model);
}

fn do_repl(args: &cli::ReplArgs) {
    let model = load_model(&args.model);

    let reverse_vocab = model.tfidf.reverse_vocab();

    eprintln!(
        "model loaded: {} features, {} classes{}",
        model.tfidf.vocab_size(),
        model.classifier.n_classes(),
        if model.calibrator.is_some() {
            " [calibrated]"
        } else {
            ""
        },
    );
    eprintln!("enter text to classify (ctrl-d to exit)\n");

    let stdin = std::io::stdin();
    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };

        let text = line.trim();
        if text.is_empty() {
            continue;
        }

        let features = model.tfidf.transform(text);
        let (pred, probs) = model.predict_features_with_proba(&features);
        let mut contribs = model.classifier.feature_contributions(&features);

        if matches!(
            model.classifier,
            Classifier::OvaSgd(_) | Classifier::OvaLbfgs(_)
        ) {
            let labels = model.predict_features_labels_with_proba(&features);
            let label_strs: Vec<String> = labels
                .iter()
                .map(|(label, prob)| format!("{}:{:.4}", label, prob))
                .collect();
            let prob_strs: Vec<String> = probs
                .iter()
                .enumerate()
                .map(|(k, p)| format!("{}:{:.4}", k, p))
                .collect();
            let predicted = if label_strs.is_empty() {
                "(none)".to_string()
            } else {
                label_strs.join(", ")
            };
            println!("predicted: {} [all: {}]", predicted, prob_strs.join("  "));
        } else if model.classifier.is_binary() {
            let probability = if pred == 1 { probs[0] } else { 1.0 - probs[0] };
            println!("predicted: {} (prob: {:.4})", pred, probability);
        } else {
            let prob_strs: Vec<String> = probs
                .iter()
                .enumerate()
                .map(|(k, p)| format!("{}:{:.4}", k, p))
                .collect();
            println!("predicted: {} [{}]", pred, prob_strs.join("  "));
        }

        contribs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let top_pos: Vec<_> = contribs.iter().filter(|(_, c)| *c > 0.0).take(5).collect();
        if !top_pos.is_empty() {
            println!("  top positive:");
            for &(idx, contrib) in &top_pos {
                let term = reverse_vocab[*idx].as_deref().unwrap_or("?");
                println!("    {:<20} {:+.4}", format!("\"{}\"", term), contrib);
            }
        }

        let top_neg: Vec<_> = contribs
            .iter()
            .rev()
            .filter(|(_, c)| *c < 0.0)
            .take(5)
            .collect();
        if !top_neg.is_empty() {
            println!("  top negative:");
            for &(idx, contrib) in &top_neg {
                let term = reverse_vocab[*idx].as_deref().unwrap_or("?");
                println!("    {:<20} {:+.4}", format!("\"{}\"", term), contrib);
            }
        }

        println!();
    }
}

fn main() {
    let cli = cli::Cli::parse();

    match cli.command {
        cli::Commands::Train(args) => do_train(&args),
        cli::Commands::Predict(args) => do_predict(&args),
        cli::Commands::Evaluate(args) => do_evaluate(&args),
        cli::Commands::Repl(args) => do_repl(&args),
        cli::Commands::Calibrate(args) => do_calibrate(&args),
    }
}
