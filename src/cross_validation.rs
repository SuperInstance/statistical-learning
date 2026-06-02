//! Cross-validation methods: k-fold and leave-one-out.

use rand::seq::SliceRandom;
use rand::Rng;
use rand::SeedableRng;
use serde::{Serialize, Deserialize};

/// Cross-validation result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossValidationResult {
    /// Mean score across all folds.
    pub mean_score: f64,
    /// Standard deviation of scores across folds.
    pub std_score: f64,
    /// Individual fold scores.
    pub fold_scores: Vec<f64>,
    /// Number of folds.
    pub n_folds: usize,
}

/// Cross-validator that can apply k-fold and LOO-CV.
pub struct CrossValidator;

impl CrossValidator {
    /// Generate k-fold split indices.
    ///
    /// Returns a vector of (train_indices, test_indices) tuples.
    pub fn kfold_indices(n: usize, k: usize, shuffle: bool, seed: Option<u64>) -> Vec<(Vec<usize>, Vec<usize>)> {
        assert!(k > 1 && k <= n, "k must be between 2 and n");

        let mut indices: Vec<usize> = (0..n).collect();
        if shuffle {
            let mut rng = match seed {
                Some(s) => {
                    let mut seed_bytes = [0u8; 32];
                    let le = s.to_le_bytes();
                    for (i, b) in le.iter().enumerate() { seed_bytes[i] = *b; }
                    rand::rngs::StdRng::from_seed(seed_bytes)
                }
                None => {
                    let s: u64 = rand::rng().random();
                    let mut seed_bytes = [0u8; 32];
                    let le = s.to_le_bytes();
                    for (i, b) in le.iter().enumerate() { seed_bytes[i] = *b; }
                    rand::rngs::StdRng::from_seed(seed_bytes)
                }
            };
            indices.shuffle(&mut rng);
        }

        let fold_size = n / k;
        let remainder = n % k;

        let mut folds = Vec::with_capacity(k);
        let mut start = 0;
        for i in 0..k {
            let size = fold_size + if i < remainder { 1 } else { 0 };
            let test_indices: Vec<usize> = indices[start..start + size].to_vec();
            let train_indices: Vec<usize> = indices[..start]
                .iter()
                .chain(indices[start + size..].iter())
                .copied()
                .collect();
            folds.push((train_indices, test_indices));
            start += size;
        }
        folds
    }
}

/// Perform k-fold cross-validation using a custom scoring function.
///
/// # Arguments
/// * `n` - Total number of samples
/// * `k` - Number of folds
/// * `score_fn` - Closure that takes (train_indices, test_indices) and returns a score
/// * `shuffle` - Whether to shuffle before splitting
pub fn cross_validate_kfold<F>(
    n: usize,
    k: usize,
    score_fn: F,
    shuffle: bool,
) -> CrossValidationResult
where
    F: Fn(&[usize], &[usize]) -> f64,
{
    let folds = CrossValidator::kfold_indices(n, k, shuffle, None);
    let fold_scores: Vec<f64> = folds
        .iter()
        .map(|(train, test)| score_fn(train, test))
        .collect();

    let mean = fold_scores.iter().sum::<f64>() / fold_scores.len() as f64;
    let variance = fold_scores
        .iter()
        .map(|s| (s - mean).powi(2))
        .sum::<f64>()
        / fold_scores.len() as f64;

    CrossValidationResult {
        mean_score: mean,
        std_score: variance.sqrt(),
        fold_scores,
        n_folds: k,
    }
}

/// Perform leave-one-out cross-validation (LOO-CV).
///
/// Special case of k-fold where k = n.
pub fn cross_validate_loo<F>(n: usize, score_fn: F) -> CrossValidationResult
where
    F: Fn(&[usize], &[usize]) -> f64,
{
    cross_validate_kfold(n, n, score_fn, false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kfold_all_samples_used() {
        let n = 20;
        let folds = CrossValidator::kfold_indices(n, 5, false, None);
        assert_eq!(folds.len(), 5);

        // Each sample appears in exactly one test set
        let mut test_counts = vec![0usize; n];
        for (_, test) in &folds {
            for &idx in test {
                test_counts[idx] += 1;
            }
        }
        for count in test_counts {
            assert_eq!(count, 1);
        }
    }

    #[test]
    fn test_kfold_train_size() {
        let n = 100;
        let folds = CrossValidator::kfold_indices(n, 5, false, None);
        for (train, test) in &folds {
            assert_eq!(train.len() + test.len(), n);
        }
    }

    #[test]
    fn test_kfold_no_overlap() {
        let n = 30;
        let folds = CrossValidator::kfold_indices(n, 3, false, None);
        for (train, test) in &folds {
            for &t in test {
                assert!(!train.contains(&t));
            }
        }
    }

    #[test]
    fn test_kfold_equal_fold_sizes() {
        let n = 20;
        let folds = CrossValidator::kfold_indices(n, 5, false, None);
        for (_, test) in &folds {
            assert_eq!(test.len(), 4);
        }
    }

    #[test]
    fn test_cross_validate_kfold_perfect_score() {
        let result = cross_validate_kfold(20, 5, |_, _| 1.0, false);
        assert!((result.mean_score - 1.0).abs() < 1e-10);
        assert!(result.std_score.abs() < 1e-10);
    }

    #[test]
    fn test_cross_validate_kfold_varying_scores() {
        let result = cross_validate_kfold(20, 4, |train, _| train.len() as f64, false);
        // Each train set should be ~15 samples
        assert!((result.mean_score - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_loo_n_folds() {
        let result = cross_validate_loo(10, |_, _| 1.0);
        assert_eq!(result.n_folds, 10);
        assert_eq!(result.fold_scores.len(), 10);
    }

    #[test]
    fn test_loo_train_size() {
        let n = 15;
        let result = cross_validate_loo(n, |train, test| {
            assert_eq!(test.len(), 1);
            assert_eq!(train.len(), n - 1);
            0.0
        });
        assert_eq!(result.n_folds, 15);
    }

    #[test]
    fn test_kfold_uneven_split() {
        // 23 samples, 5 folds -> sizes 5,5,5,4,4
        let n = 23;
        let folds = CrossValidator::kfold_indices(n, 5, false, None);
        let test_sizes: Vec<usize> = folds.iter().map(|(_, t)| t.len()).collect();
        assert_eq!(test_sizes.iter().sum::<usize>(), n);
    }

    #[test]
    fn test_kfold_shuffled_still_covers_all() {
        let n = 50;
        let folds = CrossValidator::kfold_indices(n, 10, true, Some(42));
        let all_test: Vec<usize> = folds.iter().flat_map(|(_, t)| t.iter().copied()).collect();
        let mut sorted = all_test.clone();
        sorted.sort();
        let expected: Vec<usize> = (0..n).collect();
        assert_eq!(sorted, expected);
    }
}
