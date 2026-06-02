//! Learning curves and sample complexity estimation.

use serde::{Serialize, Deserialize};

/// A point on the learning curve.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningCurvePoint {
    /// Number of training samples.
    pub n_samples: usize,
    /// Training error at this sample size.
    pub train_error: f64,
    /// Validation/test error at this sample size.
    pub val_error: f64,
}

/// Generate a theoretical learning curve.
///
/// Models the typical behavior:
/// - Train error: increases slightly with more data (harder to fit perfectly)
/// - Val error: decreases with more data (better generalization)
/// - Both converge to Bayes error as n → ∞
///
/// Uses parametric forms:
/// - E_train(n) = E_bayes + (a - E_bayes) * exp(-α_train * n)
/// - E_val(n) = E_bayes + (b - E_bayes) * n^(-α_val)
pub fn learning_curve(
    sample_sizes: &[usize],
    bayes_error: f64,
    model_capacity: f64,
    learning_rate: f64,
) -> Vec<LearningCurvePoint> {
    sample_sizes
        .iter()
        .map(|&n| {
            let n_f = n as f64;
            let train_error = bayes_error + (model_capacity - bayes_error) * (-learning_rate * n_f * 0.5).exp();
            let val_error = bayes_error + (1.0 - bayes_error) * (n_f).powf(-learning_rate);

            LearningCurvePoint {
                n_samples: n,
                train_error: train_error.max(0.0),
                val_error: val_error.max(0.0),
            }
        })
        .collect()
}

/// Estimate sample complexity from a target error and confidence.
///
/// Uses inverse of the learning curve to estimate how many samples
/// are needed to achieve a given generalization error.
pub fn sample_complexity_estimate(
    target_error: f64,
    bayes_error: f64,
    learning_rate: f64,
    delta: f64,
) -> usize {
    assert!(target_error > bayes_error, "target error must exceed Bayes error");

    // From E_val(n) = E_bayes + (1 - E_bayes) * n^(-α)
    // Solve: n^(-α) = (target - E_bayes) / (1 - E_bayes)
    // n = ((1 - E_bayes) / (target - E_bayes))^(1/α)
    let n_base = ((1.0 - bayes_error) / (target_error - bayes_error)).powf(1.0 / learning_rate);

    // Add confidence margin: multiply by ln(1/δ)
    let n_with_confidence = n_base * (1.0 / delta).ln();

    n_with_confidence.ceil() as usize
}

/// Detect overfitting from a learning curve.
///
/// Returns true if validation error starts increasing while training error
/// continues to decrease.
pub fn detect_overfitting(curve: &[LearningCurvePoint]) -> bool {
    if curve.len() < 3 {
        return false;
    }

    // Find minimum validation error
    let min_val_idx = curve
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.val_error.partial_cmp(&b.val_error).unwrap())
        .unwrap()
        .0;

    // Check if validation error increases after the minimum
    if min_val_idx < curve.len() - 1 {
        let last = &curve[curve.len() - 1];
        let min_val = &curve[min_val_idx];
        // Overfitting if val error increased by more than 10%
        if last.val_error > min_val.val_error * 1.1 {
            return true;
        }
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_learning_curve_decreasing_val_error() {
        let sizes: Vec<usize> = (1..=100).collect();
        let curve = learning_curve(&sizes, 0.05, 0.5, 0.5);
        // Validation error should generally decrease
        assert!(curve[99].val_error < curve[0].val_error);
    }

    #[test]
    fn test_learning_curve_convergence() {
        let sizes: Vec<usize> = vec![10, 100, 1000, 10000];
        let curve = learning_curve(&sizes, 0.05, 0.5, 0.5);
        // Gap between train and val should narrow
        let gap_first = (curve[0].train_error - curve[0].val_error).abs();
        let gap_last = (curve[3].train_error - curve[3].val_error).abs();
        assert!(gap_last < gap_first);
    }

    #[test]
    fn test_learning_curve_non_negative() {
        let sizes: Vec<usize> = vec![5, 10, 50, 100];
        let curve = learning_curve(&sizes, 0.0, 0.5, 0.5);
        for point in &curve {
            assert!(point.train_error >= 0.0);
            assert!(point.val_error >= 0.0);
        }
    }

    #[test]
    fn test_learning_curve_points_count() {
        let sizes: Vec<usize> = vec![10, 20, 30];
        let curve = learning_curve(&sizes, 0.05, 0.5, 0.5);
        assert_eq!(curve.len(), 3);
    }

    #[test]
    fn test_sample_complexity_estimate_positive() {
        let n = sample_complexity_estimate(0.1, 0.05, 0.5, 0.05);
        assert!(n > 0);
    }

    #[test]
    fn test_sample_complexity_decreases_with_tolerance() {
        let n1 = sample_complexity_estimate(0.1, 0.05, 0.5, 0.05);
        let n2 = sample_complexity_estimate(0.2, 0.05, 0.5, 0.05);
        // More tolerance → fewer samples needed
        assert!(n2 < n1);
    }

    #[test]
    fn test_sample_complexity_increases_with_confidence() {
        let n1 = sample_complexity_estimate(0.1, 0.05, 0.5, 0.1);
        let n2 = sample_complexity_estimate(0.1, 0.05, 0.5, 0.01);
        // Higher confidence (lower delta) → more samples
        assert!(n2 > n1);
    }

    #[test]
    fn test_detect_overfitting_no_overfitting() {
        let curve = vec![
            LearningCurvePoint { n_samples: 10, train_error: 0.5, val_error: 0.6 },
            LearningCurvePoint { n_samples: 50, train_error: 0.3, val_error: 0.35 },
            LearningCurvePoint { n_samples: 100, train_error: 0.2, val_error: 0.25 },
        ];
        assert!(!detect_overfitting(&curve));
    }

    #[test]
    fn test_detect_overfitting_with_overfitting() {
        let curve = vec![
            LearningCurvePoint { n_samples: 10, train_error: 0.5, val_error: 0.55 },
            LearningCurvePoint { n_samples: 50, train_error: 0.3, val_error: 0.25 },
            LearningCurvePoint { n_samples: 100, train_error: 0.1, val_error: 0.4 },
        ];
        assert!(detect_overfitting(&curve));
    }

    #[test]
    fn test_detect_overfitting_short_curve() {
        let curve = vec![
            LearningCurvePoint { n_samples: 10, train_error: 0.5, val_error: 0.6 },
        ];
        assert!(!detect_overfitting(&curve));
    }

    #[test]
    fn test_val_error_above_bayes() {
        let sizes: Vec<usize> = vec![10, 100, 1000];
        let curve = learning_curve(&sizes, 0.05, 0.5, 0.5);
        for point in &curve {
            // Val error should be at or above Bayes error
            assert!(point.val_error >= 0.05 - 0.01);
        }
    }
}
