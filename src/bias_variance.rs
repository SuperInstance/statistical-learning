//! Bias-variance tradeoff: decomposition and visualization utilities.

use nalgebra::DVector;
use serde::{Serialize, Deserialize};

/// Result of a bias-variance decomposition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiasVarianceDecomposition {
    /// Bias² component.
    pub bias_sq: f64,
    /// Variance component.
    pub variance: f64,
    /// Irreducible noise (Bayes error).
    pub noise: f64,
    /// Total expected prediction error = bias² + variance + noise.
    pub total_error: f64,
    /// Points for plotting the tradeoff curve (model complexity vs error components).
    pub tradeoff_curve: Vec<TradeoffPoint>,
}

/// A single point on the bias-variance tradeoff curve.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeoffPoint {
    /// Model complexity parameter.
    pub complexity: f64,
    /// Bias² at this complexity.
    pub bias_sq: f64,
    /// Variance at this complexity.
    pub variance: f64,
    /// Total error at this complexity.
    pub total_error: f64,
}

/// Perform bias-variance decomposition from a set of predictions.
///
/// Given the true values, a set of predictions from different training sets,
/// and the irreducible noise variance, computes the decomposition:
///   E[(y - ŷ)²] = Bias[ŷ]² + Var[ŷ] + σ²
///
/// # Arguments
/// * `y_true` - True target values
/// * `predictions` - Vector of prediction vectors, each from a different training set
/// * `noise_variance` - Known irreducible noise variance (Bayes error)
pub fn bias_variance_decompose(
    y_true: &DVector<f64>,
    predictions: &[DVector<f64>],
    noise_variance: f64,
) -> BiasVarianceDecomposition {
    let n = y_true.len();
    let m = predictions.len() as f64;

    // Mean prediction across all models (for each sample)
    let mean_pred: DVector<f64> = predictions
        .iter()
        .fold(DVector::zeros(n), |acc, p| acc + p)
        / m;

    // Bias² = E[(E[ŷ] - y)²]
    let bias_sq = (&mean_pred - y_true).norm_squared() / n as f64;

    // Variance = E[(ŷ - E[ŷ])²]
    let variance = predictions
        .iter()
        .map(|p| (p - &mean_pred).norm_squared())
        .sum::<f64>()
        / (m * n as f64);

    let total_error = bias_sq + variance + noise_variance;

    BiasVarianceDecomposition {
        bias_sq,
        variance,
        noise: noise_variance,
        total_error,
        tradeoff_curve: vec![],
    }
}

/// Generate a theoretical bias-variance tradeoff curve.
///
/// Models the typical U-shaped test error curve as a function of model complexity.
/// Uses parametric forms:
/// - bias² ≈ B₀ * exp(-α * complexity)
/// - variance ≈ V₀ * (1 - exp(-β * complexity))
pub fn generate_tradeoff_curve(
    n_points: usize,
    bias_base: f64,
    variance_base: f64,
    noise: f64,
    bias_decay: f64,
    variance_growth: f64,
) -> Vec<TradeoffPoint> {
    (0..n_points)
        .map(|i| {
            let complexity = i as f64 / (n_points - 1).max(1) as f64 * 10.0;
            let b = bias_base * (-bias_decay * complexity).exp();
            let v = variance_base * (1.0 - (-variance_growth * complexity).exp());
            TradeoffPoint {
                complexity,
                bias_sq: b,
                variance: v,
                total_error: b + v + noise,
            }
        })
        .collect()
}

/// Compute bias-variance decomposition with a tradeoff curve included.
pub fn bias_variance_with_curve(
    y_true: &DVector<f64>,
    predictions: &[DVector<f64>],
    noise_variance: f64,
) -> BiasVarianceDecomposition {
    let mut result = bias_variance_decompose(y_true, predictions, noise_variance);
    result.tradeoff_curve = generate_tradeoff_curve(50, result.bias_sq * 2.0, result.variance * 2.0, noise_variance, 0.5, 0.3);
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_bias_variance_constant_model() {
        // Constant predictions: high bias, zero variance
        let y_true = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let predictions = vec![
            DVector::from_vec(vec![2.5, 2.5, 2.5, 2.5]),
            DVector::from_vec(vec![2.5, 2.5, 2.5, 2.5]),
            DVector::from_vec(vec![2.5, 2.5, 2.5, 2.5]),
        ];
        let result = bias_variance_decompose(&y_true, &predictions, 0.0);
        assert_relative_eq!(result.variance, 0.0, epsilon = 1e-10);
        assert!(result.bias_sq > 0.0);
    }

    #[test]
    fn test_bias_variance_perfect_model() {
        // Perfect predictions: zero bias, zero variance
        let y_true = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let predictions = vec![
            DVector::from_vec(vec![1.0, 2.0, 3.0]),
            DVector::from_vec(vec![1.0, 2.0, 3.0]),
        ];
        let result = bias_variance_decompose(&y_true, &predictions, 0.1);
        assert_relative_eq!(result.bias_sq, 0.0, epsilon = 1e-10);
        assert_relative_eq!(result.variance, 0.0, epsilon = 1e-10);
        assert_relative_eq!(result.noise, 0.1);
    }

    #[test]
    fn test_bias_variance_with_varying_predictions() {
        let y_true = DVector::from_vec(vec![0.0, 0.0]);
        let predictions = vec![
            DVector::from_vec(vec![1.0, -1.0]),
            DVector::from_vec(vec![-1.0, 1.0]),
        ];
        let result = bias_variance_decompose(&y_true, &predictions, 0.0);
        // Mean prediction is [0, 0], so bias should be 0
        assert_relative_eq!(result.bias_sq, 0.0, epsilon = 1e-10);
        // Variance should be positive
        assert!(result.variance > 0.0);
    }

    #[test]
    fn test_total_error_equals_components() {
        let y_true = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let predictions = vec![
            DVector::from_vec(vec![1.1, 2.2, 2.9, 4.1, 4.8]),
            DVector::from_vec(vec![0.9, 1.8, 3.1, 3.9, 5.2]),
            DVector::from_vec(vec![1.0, 2.1, 3.0, 4.0, 5.1]),
        ];
        let result = bias_variance_decompose(&y_true, &predictions, 0.05);
        assert_relative_eq!(
            result.total_error,
            result.bias_sq + result.variance + result.noise,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_tradeoff_curve_shape() {
        let curve = generate_tradeoff_curve(50, 1.0, 1.0, 0.1, 0.5, 0.3);
        assert_eq!(curve.len(), 50);
        // At low complexity: bias should be high
        assert!(curve[0].bias_sq > curve[49].bias_sq);
        // At high complexity: variance should be high
        assert!(curve[49].variance > curve[0].variance);
    }

    #[test]
    fn test_tradeoff_curve_u_shape() {
        let curve = generate_tradeoff_curve(100, 2.0, 3.0, 0.1, 0.8, 0.5);
        // Total error should have a minimum somewhere
        let min_idx = curve
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.total_error.partial_cmp(&b.total_error).unwrap())
            .unwrap()
            .0;
        // Minimum should not be at the very start (index 0)
        assert!(min_idx > 0);
        // Bias should decrease and variance should increase
        assert!(curve[0].bias_sq > curve[99].bias_sq);
        assert!(curve[99].variance > curve[0].variance);
    }

    #[test]
    fn test_bias_variance_noise_only() {
        let y_true = DVector::from_vec(vec![1.0, 1.0]);
        let predictions = vec![
            DVector::from_vec(vec![1.0, 1.0]),
        ];
        let result = bias_variance_decompose(&y_true, &predictions, 0.5);
        assert_relative_eq!(result.bias_sq, 0.0, epsilon = 1e-10);
        assert_relative_eq!(result.variance, 0.0, epsilon = 1e-10);
        assert_relative_eq!(result.total_error, 0.5, epsilon = 1e-10);
    }
}
