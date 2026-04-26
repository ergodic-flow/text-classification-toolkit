# TODO

## Correctness

- [ ] Make prediction semantics consistent for calibrated models.
  - `Model::predict` should agree with `Model::predict_proba` after calibration.
  - CLI, REPL, and Python wrapper should all use the same prediction path.

- [ ] Fix multilabel/OvA calibration behavior.
  - Do not normalize independent multilabel probabilities to sum to 1.
  - Keep multiclass and multilabel calibration post-processing separate.

- [ ] Make multilabel prediction output explicit and consistent.
  - Decide whether CLI `predict` emits top-1 labels or all labels above threshold for OvA models.
  - Expose multilabel prediction from the Python wrapper if multilabel is a supported feature.
  - Align `predict` behavior with `evaluate` behavior.

- [ ] Harden `tune` cross-validation validation.
  - Reject `--folds 1` unless intentionally supporting leave-all-in-test behavior.
  - Reject `--folds > n_samples` or handle empty folds safely.
  - Avoid `NaN` scores from empty test folds.

## Reliability

- [ ] Improve TSV input validation.
  - Validate the header or document that the first row is always skipped.
  - Reject malformed rows without a tab instead of silently treating text as empty.
  - Replace avoidable `unwrap()` calls on file and line I/O with user-facing errors.

- [ ] Add integration/regression tests.
  - Binary train/predict/evaluate round trip.
  - Multinomial train/predict/evaluate round trip.
  - OvA multilabel train/predict/evaluate semantics.
  - Calibration behavior for binary, multiclass, and multilabel models.
  - Serialization/load compatibility.

- [ ] Decide how strict linting should be.
  - Fix simple clippy warnings like unused variables and `Default` for `TfIdf`.
  - Either refactor large argument lists into config structs or explicitly allow them where appropriate.

## Packaging

- [ ] Make Python packaging real if it remains a project feature.
  - Add `pyproject.toml` for maturin/pip installs.
  - Document the tested install path.
  - Ensure CI separates normal Rust tests from PyO3 extension-module builds.

- [ ] Resolve `cargo test --all-features` behavior.
  - It currently fails on macOS with unresolved Python symbols when `extension-module` is enabled.
  - Decide whether all-features testing should be supported or documented as not applicable for extension builds.

## Maintainability

- [ ] Split `src/main.rs` into smaller modules.
  - Keep CLI dispatch in `main.rs`.
  - Move training, tuning, prediction, evaluation, calibration, and TSV parsing into reusable modules.

- [ ] Reduce duplication between SGD/L-BFGS/OvA training paths.
  - Consider shared config structs for solver params and training data.
  - Keep the change minimal; avoid abstracting before the behavior is stable.

## Documentation

- [ ] Document the supported data format more precisely.
  - Required header behavior.
  - Label formats for binary, multiclass, and multilabel.
  - Prediction output format for each model type.

- [ ] Clarify project maturity in the README.
  - Mark experimental features, especially calibration, OvA multilabel, tuning, and Python wrapper support.
