//! Regularization: L1, L2, and elastic net penalties.

use nalgebra::DVector;
use serde::{Serialize, Deserialize};

/// Result of applying regularization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegularizationResult {
    /// Regularized weights.
    pub weights: DVector<f64>,
    /// L1 penalty value.
    pub l1_penalty: f64,
    /// L2 penalty value (squared L2 norm).
    pub l2_penalty: f64,
    /// Total regularization penalty.
    pub total_penalty: f64,
    /// Number of weights shrunk to (near) zero.
    pub n_zeroed: usize,
    /// Sparsity ratio (fraction of zero weights).
    pub sparsity: f64,
}

/// Apply L1 (Lasso) regularization via soft-thresholding proximal operator.
///
/// prox_{λ|·|}(w) = sign(w) * max(|w| - λ, 0)
pub fn regularize_l1(weights: &DVector<f64>, lambda: f64) -> RegularizationResult {
    assert!(lambda >= 0.0, "lambda must be non-negative");

    let regularized: DVector<f64> = weights.map(|w| soft_threshold(w, lambda));
    let l1 = regularized.iter().map(|w| w.abs()).sum::<f64>();
    let _l2 = regularized.norm_squared();
    let n_zeroed = regularized.iter().filter(|w| w.abs() < 1e-12).count();
    let sparsity = n_zeroed as f64 / regularized.len() as f64;

    RegularizationResult {
        weights: regularized,
        l1_penalty: lambda * l1,
        l2_penalty: 0.0,
        total_penalty: lambda * l1,
        n_zeroed,
        sparsity,
    }
}

/// Apply L2 (Ridge) regularization: shrink weights toward zero.
///
/// w_reg = w / (1 + λ)  (proximal form)
pub fn regularize_l2(weights: &DVector<f64>, lambda: f64) -> RegularizationResult {
    assert!(lambda >= 0.0);

    let regularized = weights.scale(1.0 / (1.0 + lambda));
    let l2 = regularized.norm_squared();
    let n_zeroed = regularized.iter().filter(|w| w.abs() < 1e-12).count();
    let sparsity = n_zeroed as f64 / regularized.len() as f64;

    RegularizationResult {
        weights: regularized,
        l1_penalty: 0.0,
        l2_penalty: lambda * l2,
        total_penalty: lambda * l2,
        n_zeroed,
        sparsity,
    }
}

/// Apply elastic net regularization (L1 + L2).
///
/// Combines sparsity (L1) with grouping (L2).
/// prox_{α|·|₁ + β|·|₂²}(w) = soft_threshold(w, α) / (1 + β)
pub fn regularize_elastic_net(
    weights: &DVector<f64>,
    l1_ratio: f64,
    lambda: f64,
) -> RegularizationResult {
    assert!((0.0..=1.0).contains(&l1_ratio), "l1_ratio must be in [0, 1]");
    assert!(lambda >= 0.0);

    let alpha = lambda * l1_ratio;
    let beta = lambda * (1.0 - l1_ratio);

    let soft: DVector<f64> = weights.map(|w| soft_threshold(w, alpha));
    let regularized = soft.scale(1.0 / (1.0 + beta));

    let l1 = regularized.iter().map(|w| w.abs()).sum::<f64>();
    let l2 = regularized.norm_squared();
    let n_zeroed = regularized.iter().filter(|w| w.abs() < 1e-12).count();
    let sparsity = n_zeroed as f64 / regularized.len() as f64;

    RegularizationResult {
        weights: regularized,
        l1_penalty: alpha * l1,
        l2_penalty: beta * l2,
        total_penalty: alpha * l1 + beta * l2,
        n_zeroed,
        sparsity,
    }
}

/// Soft-thresholding operator.
fn soft_threshold(w: f64, threshold: f64) -> f64 {
    if w > threshold {
        w - threshold
    } else if w < -threshold {
        w + threshold
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_l1_zero_lambda() {
        let w = DVector::from_vec(vec![1.0, -2.0, 3.0]);
        let result = regularize_l1(&w, 0.0);
        for i in 0..3 {
            assert_relative_eq!(result.weights[i], w[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_l1_produces_sparsity() {
        let w = DVector::from_vec(vec![0.1, -0.1, 5.0, -5.0]);
        let result = regularize_l1(&w, 0.5);
        // Small weights should be zeroed
        assert!(result.weights[0].abs() < 1e-10);
        assert!(result.weights[1].abs() < 1e-10);
        // Large weights should remain
        assert!(result.weights[2] > 0.0);
        assert!(result.weights[3] < 0.0);
        assert_eq!(result.n_zeroed, 2);
    }

    #[test]
    fn test_l1_large_lambda_zeros_all() {
        let w = DVector::from_vec(vec![1.0, -2.0, 3.0]);
        let result = regularize_l1(&w, 100.0);
        assert_eq!(result.n_zeroed, 3);
        assert_relative_eq!(result.sparsity, 1.0);
    }

    #[test]
    fn test_l2_shrinks_weights() {
        let w = DVector::from_vec(vec![3.0, 4.0]);
        let result = regularize_l2(&w, 1.0);
        // Each weight divided by 2
        assert_relative_eq!(result.weights[0], 1.5, epsilon = 1e-10);
        assert_relative_eq!(result.weights[1], 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_l2_zero_lambda() {
        let w = DVector::from_vec(vec![1.0, -2.0, 3.0]);
        let result = regularize_l2(&w, 0.0);
        for i in 0..3 {
            assert_relative_eq!(result.weights[i], w[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_l2_preserves_direction() {
        let w = DVector::from_vec(vec![3.0, 4.0]);
        let result = regularize_l2(&w, 2.0);
        // Weights should be in same direction (proportional)
        let ratio_original = w[0] / w[1];
        let ratio_reg = result.weights[0] / result.weights[1];
        assert_relative_eq!(ratio_original, ratio_reg, epsilon = 1e-10);
    }

    #[test]
    fn test_elastic_net_l1_only() {
        let w = DVector::from_vec(vec![1.0, -1.0, 0.5]);
        let l1 = regularize_l1(&w, 0.5);
        let en = regularize_elastic_net(&w, 1.0, 0.5);
        // With l1_ratio=1.0, elastic net should behave like L1
        for i in 0..3 {
            assert_relative_eq!(l1.weights[i], en.weights[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_elastic_net_l2_only() {
        let w = DVector::from_vec(vec![2.0, 3.0]);
        let l2 = regularize_l2(&w, 1.0);
        let en = regularize_elastic_net(&w, 0.0, 1.0);
        for i in 0..2 {
            assert_relative_eq!(l2.weights[i], en.weights[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_elastic_net_combined() {
        let w = DVector::from_vec(vec![0.3, 5.0, -0.2]);
        let en = regularize_elastic_net(&w, 0.5, 1.0);
        // Should have some sparsity from L1 and shrinkage from L2
        assert!(en.weights[0].abs() < w[0].abs());
        assert!(en.weights[1].abs() < w[1].abs());
    }

    #[test]
    fn test_regularization_total_penalty_positive() {
        let w = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let l1 = regularize_l1(&w, 0.5);
        let l2 = regularize_l2(&w, 0.5);
        let en = regularize_elastic_net(&w, 0.5, 0.5);
        assert!(l1.total_penalty >= 0.0);
        assert!(l2.total_penalty >= 0.0);
        assert!(en.total_penalty >= 0.0);
    }
}
