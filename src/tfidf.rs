use hashbrown::HashMap;

/// Determines whether a document frequency threshold is treated as an
/// absolute count or a proportion of the total documents.
#[derive(serde::Serialize, serde::Deserialize, Clone, Copy, Debug)]
pub enum Threshold {
    Count(usize),
    Fraction(f64),
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct TfIdf {
    vocab: HashMap<String, usize>,
    idf: Vec<f64>,
    min_n: usize,
    max_n: usize,
    max_features: Option<usize>,
    min_df: Threshold,
    max_df: Threshold,
}

impl TfIdf {
    pub fn new() -> Self {
        Self {
            vocab: HashMap::new(),
            idf: Vec::new(),
            min_n: 1,
            max_n: 1,
            max_features: None,
            min_df: Threshold::Count(1), // Default: appears in at least 1 document
            max_df: Threshold::Fraction(1.0), // Default: can appear in up to 100% of documents
        }
    }

    pub fn max_n(mut self, max_n: usize) -> Self {
        self.max_n = max_n;
        self
    }

    pub fn min_n(mut self, min_n: usize) -> Self {
        self.min_n = min_n;
        self
    }

    /// Set the maximum number of features (vocabulary size) to keep,
    /// ordered by term frequency across the entire corpus.
    pub fn max_features(mut self, max_features: usize) -> Self {
        self.max_features = Some(max_features);
        self
    }

    /// Set the minimum document frequency required to keep a term.
    pub fn min_df(mut self, min_df: Threshold) -> Self {
        self.min_df = min_df;
        self
    }

    /// Set the maximum document frequency allowed to keep a term.
    pub fn max_df(mut self, max_df: Threshold) -> Self {
        self.max_df = max_df;
        self
    }

    pub fn fit(&mut self, documents: &[&str]) {
        // state maps: ngram -> (document_frequency, last_seen_doc_id, total_term_frequency)
        // Adding total_term_frequency allows us to correctly sort by max_features later.
        let mut state: HashMap<String, (usize, usize, usize)> = HashMap::new();
        let mut buffer = String::with_capacity(64);

        for (doc_id, &doc) in documents.iter().enumerate() {
            let words: Vec<&str> = doc.split_whitespace().collect();
            for n in self.min_n..=self.max_n {
                for i in 0..words.len().saturating_sub(n - 1) {
                    buffer.clear();
                    let window = &words[i..i + n];
                    if let Some((first, rest)) = window.split_first() {
                        buffer.push_str(first);
                        for word in rest {
                            buffer.push(' ');
                            buffer.push_str(word);
                        }
                    }

                    if let Some(entry) = state.get_mut(&buffer) {
                        // Increment doc frequency if we haven't seen it in *this* doc yet
                        if entry.1 != doc_id {
                            entry.0 += 1;
                            entry.1 = doc_id;
                        }
                        // Always increment total term frequency
                        entry.2 += 1;
                    } else {
                        state.insert(buffer.clone(), (1, doc_id, 1));
                    }
                }
            }
        }

        let n_docs = documents.len() as f64;

        // Calculate absolute thresholds
        let min_df_abs = match self.min_df {
            Threshold::Count(c) => c,
            Threshold::Fraction(f) => (f * n_docs).ceil() as usize,
        };
        let max_df_abs = match self.max_df {
            Threshold::Count(c) => c,
            Threshold::Fraction(f) => (f * n_docs).floor() as usize,
        };

        // Filter by min_df and max_df
        let mut vocab_list: Vec<_> = state
            .into_iter()
            .filter_map(|(term, (df, _, tf))| {
                if df >= min_df_abs && df <= max_df_abs {
                    Some((term, df, tf))
                } else {
                    None
                }
            })
            .collect();

        // Enforce max_features by keeping the top terms by corpus-wide term frequency
        if let Some(max_f) = self.max_features {
            if vocab_list.len() > max_f {
                // Sort by TF descending, tie-breaking alphabetically ascending
                vocab_list.sort_unstable_by(|a, b| b.2.cmp(&a.2).then_with(|| a.0.cmp(&b.0)));
                vocab_list.truncate(max_f);
            }
        }

        // Sort alphabetically to ensure deterministic index assignments
        vocab_list.sort_unstable_by(|a, b| a.0.cmp(&b.0));

        self.vocab.clear();
        self.idf.clear();

        self.vocab.reserve(vocab_list.len());
        self.idf.reserve(vocab_list.len());

        for (i, (term, df, _tf)) in vocab_list.into_iter().enumerate() {
            self.vocab.insert(term, i);
            self.idf
                .push(((1.0 + n_docs) / (1.0 + df as f64)).ln() + 1.0);
        }
    }

    pub fn transform(&self, text: &str) -> Vec<(usize, f64)> {
        // [Unchanged from your original implementation]
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut total = 0;

        let mut term_counts = Vec::new();
        let mut buffer = String::with_capacity(64);

        for n in self.min_n..=self.max_n {
            for i in 0..words.len().saturating_sub(n - 1) {
                total += 1;
                buffer.clear();
                let window = &words[i..i + n];
                if let Some((first, rest)) = window.split_first() {
                    buffer.push_str(first);
                    for word in rest {
                        buffer.push(' ');
                        buffer.push_str(word);
                    }
                }

                if let Some(&idx) = self.vocab.get(&buffer) {
                    term_counts.push(idx);
                }
            }
        }

        if term_counts.is_empty() {
            return Vec::new();
        }

        term_counts.sort_unstable();

        let total_f64 = total as f64;
        let mut result = Vec::with_capacity(term_counts.len());
        let mut current_idx = term_counts[0];
        let mut current_count = 1.0;

        for &idx in term_counts.iter().skip(1) {
            if idx == current_idx {
                current_count += 1.0;
            } else {
                let tf = current_count / total_f64;
                result.push((current_idx, tf * self.idf[current_idx]));
                current_idx = idx;
                current_count = 1.0;
            }
        }

        let tf = current_count / total_f64;
        result.push((current_idx, tf * self.idf[current_idx]));

        let norm = result.iter().map(|(_, v)| v * v).sum::<f64>().sqrt();
        if norm > 0.0 {
            for (_, v) in &mut result {
                *v /= norm;
            }
        }

        result
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    pub fn reverse_vocab(&self) -> Vec<Option<String>> {
        let mut rv = vec![None; self.vocab.len()];
        for (term, &idx) in &self.vocab {
            rv[idx] = Some(term.clone());
        }
        rv
    }
}

#[cfg(test)]
mod tests {
    use super::TfIdf;

    fn feature_value(features: &[(usize, f64)], idx: usize) -> f64 {
        features
            .iter()
            .find_map(|&(feature_idx, value)| (feature_idx == idx).then_some(value))
            .unwrap_or(0.0)
    }

    #[test]
    fn smooth_idf_gives_ubiquitous_terms_idf_one() {
        let docs = ["common rare", "common", "common"];
        let mut tfidf = TfIdf::new();
        tfidf.fit(&docs);

        let common_idx = tfidf.vocab["common"];
        let rare_idx = tfidf.vocab["rare"];

        assert!((tfidf.idf[common_idx] - 1.0).abs() < 1e-12);
        assert!((tfidf.idf[rare_idx] - (2.0_f64.ln() + 1.0)).abs() < 1e-12);
    }

    #[test]
    fn transform_uses_smooth_idf_weights() {
        let docs = ["common rare", "common", "common"];
        let mut tfidf = TfIdf::new();
        tfidf.fit(&docs);

        let common_idx = tfidf.vocab["common"];
        let rare_idx = tfidf.vocab["rare"];
        let features = tfidf.transform("common rare");

        let rare_idf = 2.0_f64.ln() + 1.0;
        let norm = (1.0 + rare_idf * rare_idf).sqrt();

        assert!((feature_value(&features, common_idx) - (1.0 / norm)).abs() < 1e-12);
        assert!((feature_value(&features, rare_idx) - (rare_idf / norm)).abs() < 1e-12);
    }
}
