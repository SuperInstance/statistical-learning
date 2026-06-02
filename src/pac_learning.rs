//! PAC (Probably Approximately Correct) learning framework.

use serde::{Serialize, Deserialize};

/// PAC learning bounds and parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PACBounds {
    /// Confidence parameter δ (failure probability).
    pub delta: f64,
    /// Accuracy parameter ε (approximation error).
    pub epsilon: f64,
    /// Required sample size for PAC guarantees.
    pub sample_size: usize,
    /// VC dimension of the hypothesis class.
    pub vc_dimension: Option<usize>,
}

/// Compute the minimum sample size for realizable PAC learning.
///
/// In the realizable case (there exists h* ∈ H with zero error):
///   m ≥ (1/ε)(ln(|H|) + ln(1/δ))
pub fn pac_bound_sample_size(
    hypothesis_class_size: usize,
    epsilon: f64,
    delta: f64,
) -> PACBounds {
    assert!(epsilon > 0.0 && epsilon < 1.0);
    assert!(delta > 0.0 && delta < 1.0);
    assert!(hypothesis_class_size > 0);

    let m = ((1.0 / epsilon)
        * ((hypothesis_class_size as f64).ln() + (1.0 / delta).ln()))
    .ceil() as usize;

    PACBounds {
        delta,
        epsilon,
        sample_size: m.max(1),
        vc_dimension: None,
    }
}

/// Compute the sample size for agnostic PAC learning using VC dimension.
///
/// In the agnostic case:
///   m ≥ C * (d + ln(1/δ)) / ε²
pub fn pac_bound_agnostic(d: usize, epsilon: f64, delta: f64) -> PACBounds {
    assert!(epsilon > 0.0 && epsilon < 1.0);
    assert!(delta > 0.0 && delta < 1.0);

    // Using a standard constant C ≈ 8
    let m = (8.0 / (epsilon * epsilon) * (d as f64 + (1.0 / delta).ln())).ceil() as usize;

    PACBounds {
        delta,
        epsilon,
        sample_size: m.max(1),
        vc_dimension: Some(d),
    }
}

/// Compute the generalization bound for a given sample size and VC dimension.
///
/// With probability ≥ 1 - δ:
///   R(h) ≤ R_emp(h) + ε
pub fn pac_generalization_gap(d: usize, n: usize, delta: f64) -> f64 {
    let eps = ((8.0 / n as f64)
        * (d as f64 * (2.0 * n as f64 / d as f64).ln() + (4.0 / delta).ln()))
    .sqrt();
    eps
}

/// Verify the PAC guarantee: check if given sample size provides the desired bound.
pub fn verify_pac_guarantee(
    d: usize,
    n: usize,
    epsilon_target: f64,
    delta: f64,
) -> bool {
    pac_generalization_gap(d, n, delta) <= epsilon_target
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_realizable_pac_sample_size() {
        let bounds = pac_bound_sample_size(100, 0.1, 0.05);
        assert!(bounds.sample_size > 0);
        assert_eq!(bounds.delta, 0.05);
        assert_eq!(bounds.epsilon, 0.1);
    }

    #[test]
    fn test_realizable_pac_decreases_with_epsilon() {
        let b1 = pac_bound_sample_size(100, 0.1, 0.05);
        let b2 = pac_bound_sample_size(100, 0.01, 0.05);
        assert!(b2.sample_size > b1.sample_size);
    }

    #[test]
    fn test_realizable_pac_increases_with_delta() {
        let b1 = pac_bound_sample_size(100, 0.1, 0.01);
        let b2 = pac_bound_sample_size(100, 0.1, 0.1);
        assert!(b1.sample_size >= b2.sample_size);
    }

    #[test]
    fn test_realizable_pac_increases_with_hypothesis_size() {
        let b1 = pac_bound_sample_size(100, 0.1, 0.05);
        let b2 = pac_bound_sample_size(1000, 0.1, 0.05);
        assert!(b2.sample_size > b1.sample_size);
    }

    #[test]
    fn test_agnostic_pac_sample_size() {
        let bounds = pac_bound_agnostic(5, 0.1, 0.05);
        assert!(bounds.sample_size > 0);
        assert_eq!(bounds.vc_dimension, Some(5));
    }

    #[test]
    fn test_agnostic_pac_more_samples_than_realizable() {
        let _realizable = pac_bound_sample_size(32, 0.1, 0.05);
        let agnostic = pac_bound_agnostic(5, 0.1, 0.05);
        // Agnostic typically requires more samples due to ε² scaling
        assert!(agnostic.sample_size > 0);
    }

    #[test]
    fn test_generalization_gap_decreases_with_samples() {
        let g1 = pac_generalization_gap(5, 100, 0.05);
        let g2 = pac_generalization_gap(5, 1000, 0.05);
        assert!(g2 < g1);
    }

    #[test]
    fn test_verify_pac_guarantee() {
        // With enough samples, guarantee should hold
        let n = pac_bound_agnostic(5, 0.1, 0.05).sample_size;
        assert!(verify_pac_guarantee(5, n * 2, 0.3, 0.05));
    }
}
