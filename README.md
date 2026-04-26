# Text Classification Toolkit

A lightweight CLI-based toolkit for: binary, multiclass, multilabel text classification problems.

## Features

- SGD and L-BFGS solvers for logistic regression
- L2-regularization
- binary model serialization format
- TFIDF vectorization
- model "debugging"
- a simple TSV format for training
- class and sample weighting
- explicit support for binary, multiclass, and multilabel problems.
- a thin Python wrapper for invoking trained models

## CLI Usage

Train and evaluate use TSV files with a header and `label\ttext` rows.

Predict accepts unlabeled input: one text document per line. It writes TSV output
to stdout by default, or to `--output` when provided:

```sh
text-toolkit predict --model model.bin --input unlabeled.txt
text-toolkit predict --model model.bin --input unlabeled.txt --output predictions.tsv
```

Prediction output is always:

```text
label	probability
```

`probability` is the probability assigned to the predicted label.

## Python Wrapper

The crate also builds a small PyO3 module for loading and invoking trained
models from Python. Install it with maturin or pip from the repository root:

```sh
maturin develop
# or
pip install .
```

Then load a serialized model and run predictions:

```python
from text_toolkit import Model

model = Model("model.bin")

label = model.predict("some text to classify")
probabilities = model.predict_proba("some text to classify")
labels = model.predict_labels("some text to classify")

print(label)
print(probabilities)
print(labels)
```

For binary and multiclass models, `predict()` returns a single integer label. For
OvA multilabel models, `predict()` returns a list of labels whose probabilities
are at least `0.5`. Use `predict_labels()` when you always want a list.

The wrapper also exposes `n_classes()`, `is_binary()`, `is_multilabel()`, and
`is_calibrated()`.

## Why?

I like tools like [vw](https://vowpalwabbit.org/) and [fastText](https://fasttext.cc/), and I wanted my own spin on it.
Too often I find myself using scikit's TF-IDF and Logistic Regression, and while they're good, they require python.
This toolkit is CLI first, with the option to invoke a model from Python.
For datasets that fit on one machine, this toolkit will work just fine.

If you're looking for embeddings and transformers, this is not the toolkit for you.

## References

- Liu, Dong C., and Jorge Nocedal. "On the limited memory BFGS method for large scale optimization." Mathematical programming 45.1 (1989): 503-528.
- Robbins, Herbert, and Sutton Monro. "A stochastic approximation method." The annals of mathematical statistics (1951): 400-407.
- Salton, Gerard, and Christopher Buckley. "Term-weighting approaches in automatic text retrieval." Information processing & management 24.5 (1988): 513-523.
- Joulin, Armand, et al. "Bag of tricks for efficient text classification." Proceedings of the 15th conference of the European chapter of the association for computational linguistics: volume 2, short papers. 2017.
