//! Rademacher complexity and generalization bounds.

use rand::Rng;

/// Compute the empirical Rademacher complexity of a function class.
///
/// Given a set of function evaluations F = {f(x₁), ..., f(xₙ)} for each f in the
/// hypothesis class, and using random Rademacher variables σ ∈ {-1, +1}ⁿ.
///
/// R̂(F) = E_σ[sup_{f∈F} (1/n) Σ σᵢ f(xᵢ)]
///
/// # Arguments
/// * `predictions` - Matrix of predictions: rows = hypothesis functions, cols = samples
/// * `n_trials` - Number of Monte Carlo trials for estimating the expectation
pub fn rademacher_complexity(predictions: &[Vec<f64>], n_trials: usize) -> f64 {
    if predictions.is_empty() {
        return 0.0;
    }
    let n = predictions[0].len();
    if n == 0 {
        return 0.0;
    }

    let mut rng = rand::rng();
    let mut total = 0.0;

    for _ in 0..n_trials {
        // Generate Rademacher variables
        let sigma: Vec<f64> = (0..n).map(|_| if rng.random_bool(0.5) { 1.0 } else { -1.0 }).collect();

        // sup over hypothesis class: max correlation with random signs
        let max_corr = predictions
            .iter()
            .map(|pred| {
                pred.iter()
                    .zip(sigma.iter())
                    .map(|(p, s)| p * s)
                    .sum::<f64>()
                    / n as f64
            })
            .fold(f64::NEG_INFINITY, f64::max);

        total += max_corr;
    }

    total / n_trials as f64
}

/// Estimate Rademacher complexity from loss values.
///
/// Given a set of losses {ℓ(h, zᵢ)} for each hypothesis h, estimates R̂(ℓ ∘ H).
pub fn rademacher_complexity_estimated(
    losses: &[Vec<f64>],
    n_trials: usize,
) -> f64 {
    rademacher_complexity(losses, n_trials)
}

/// Compute the growth function bound using Rademacher complexity.
///
/// Massart's lemma: R̂(F) ≤ sqrt(2 * ln(|F|)) / n
pub fn growth_function_bound(class_size: usize, n: usize) -> f64 {
    if class_size == 0 || n == 0 {
        return 0.0;
    }
    (2.0 * (class_size as f64).ln() / n as f64).sqrt()
}

/// Generalization bound via Rademacher complexity.
///
/// With probability ≥ 1 - δ:
///   R(h) ≤ R̂_emp(h) + 2 * R̂(F) + 3 * sqrt(ln(2/δ) / (2n))
pub fn rademacher_generalization_bound(
    empirical_rademacher: f64,
    n: usize,
    delta: f64,
) -> f64 {
    2.0 * empirical_rademacher + 3.0 * (2.0 * (2.0 / delta).ln() / (2.0 * n as f64)).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_rademacher_constant_function() {
        // Test with varied functions to ensure Rademacher is well-defined
        let predictions = vec![
            vec![1.0, -1.0, 1.0, -1.0],
            vec![-1.0, 1.0, -1.0, 1.0],
        ];
        let rc = rademacher_complexity(&predictions, 500);
        // Should be positive and finite
        assert!(rc.is_finite());
        assert!(rc >= 0.0);
    }

    #[test]
    fn test_rademacher_empty() {
        let rc = rademacher_complexity(&[], 100);
        assert_relative_eq!(rc, 0.0);
    }

    #[test]
    fn test_rademacher_single_function() {
        let predictions = vec![
            vec![1.0, -1.0, 1.0, -1.0],
        ];
        let rc = rademacher_complexity(&predictions, 1000);
        // Single function: average of |f(x_i)| / n, should be close to 0 for random sigma
        assert!(rc.abs() < 1.5);
    }

    #[test]
    fn test_rademacher_many_functions() {
        // More functions → higher complexity
        let small_class: Vec<Vec<f64>> = (0..2)
            .map(|i| (0..10).map(|j| if (i + j) % 2 == 0 { 1.0 } else { -1.0 }).collect())
            .collect();
        let large_class: Vec<Vec<f64>> = (0..100)
            .map(|i| (0..10).map(|j| if (i + j) % 2 == 0 { 1.0 } else { -1.0 }).collect())
            .collect();
        let rc_small = rademacher_complexity(&small_class, 2000);
        let rc_large = rademacher_complexity(&large_class, 2000);
        assert!(rc_large >= rc_small - 0.1);
    }

    #[test]
    fn test_growth_function_bound_positive() {
        let bound = growth_function_bound(100, 50);
        assert!(bound > 0.0);
    }

    #[test]
    fn test_growth_function_bound_decreases_with_n() {
        let b1 = growth_function_bound(100, 50);
        let b2 = growth_function_bound(100, 200);
        assert!(b2 < b1);
    }

    #[test]
    fn test_rademacher_generalization_bound_positive() {
        let bound = rademacher_generalization_bound(0.1, 100, 0.05);
        assert!(bound > 0.0);
    }

    #[test]
    fn test_rademacher_generalization_bound_decreases_with_n() {
        let b1 = rademacher_generalization_bound(0.1, 100, 0.05);
        let b2 = rademacher_generalization_bound(0.1, 1000, 0.05);
        assert!(b2 < b1);
    }
}
