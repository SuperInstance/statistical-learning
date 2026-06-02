//! Support Vector Machines: hard and soft margin via SMO-like optimization.

use nalgebra::{DMatrix, DVector};
use serde::{Serialize, Deserialize};

/// SVM parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SVMParams {
    /// Regularization parameter C (soft margin). None for hard margin.
    pub c: Option<f64>,
    /// Kernel bandwidth (gamma for RBF). If None, uses linear kernel.
    pub gamma: Option<f64>,
    /// Tolerance for convergence.
    pub tol: f64,
    /// Maximum number of iterations.
    pub max_iter: usize,
}

impl Default for SVMParams {
    fn default() -> Self {
        SVMParams {
            c: Some(1.0),
            gamma: None,
            tol: 1e-3,
            max_iter: 1000,
        }
    }
}

/// Trained SVM model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SVM {
    /// Support vector indices.
    pub support_vectors: Vec<usize>,
    /// Lagrange multipliers (alphas).
    pub alphas: Vec<f64>,
    /// Bias term.
    pub bias: f64,
    /// Training data (support vectors).
    pub sv_data: Vec<DVector<f64>>,
    /// Training labels for support vectors.
    pub sv_labels: Vec<f64>,
    /// Parameters used for training.
    pub params: SVMParams,
    /// Number of iterations used.
    pub n_iterations: usize,
}

impl SVM {
    /// Predict the class of a new point.
    pub fn predict(&self, x: &DVector<f64>) -> f64 {
        let mut sum = self.bias;
        for (i, (alpha, label)) in self.alphas.iter().zip(self.sv_labels.iter()).enumerate() {
            sum += alpha * label * self.kernel_value(&self.sv_data[i], x);
        }
        if sum >= 0.0 { 1.0 } else { -1.0 }
    }

    /// Compute the decision function value (before sign).
    pub fn decision_function(&self, x: &DVector<f64>) -> f64 {
        let mut sum = self.bias;
        for (i, (alpha, label)) in self.alphas.iter().zip(self.sv_labels.iter()).enumerate() {
            sum += alpha * label * self.kernel_value(&self.sv_data[i], x);
        }
        sum
    }

    /// Compute the margin width (2/||w|| for linear kernel).
    pub fn margin(&self) -> f64 {
        // For linear: ||w||² = Σᵢ Σⱼ αᵢ αⱼ yᵢ yⱼ K(xᵢ, xⱼ)
        let mut w_norm_sq = 0.0;
        for i in 0..self.sv_data.len() {
            for j in 0..self.sv_data.len() {
                w_norm_sq += self.alphas[i] * self.alphas[j]
                    * self.sv_labels[i] * self.sv_labels[j]
                    * self.kernel_value(&self.sv_data[i], &self.sv_data[j]);
            }
        }
        if w_norm_sq > 0.0 {
            2.0 / w_norm_sq.sqrt()
        } else {
            f64::INFINITY
        }
    }

    fn kernel_value(&self, x: &DVector<f64>, y: &DVector<f64>) -> f64 {
        match self.params.gamma {
            Some(gamma) => (-gamma * (x - y).norm_squared()).exp(),
            None => x.dot(y),
        }
    }
}

/// Hard margin SVM (requires linearly separable data).
pub struct HardMarginSVM;

impl HardMarginSVM {
    /// Train a hard margin SVM using a simplified SMO algorithm.
    ///
    /// Only works for linearly separable data.
    pub fn train(
        x: &[DVector<f64>],
        y: &[f64],
        params: SVMParams,
    ) -> SVM {
        train_smo(x, y, SVMParams {
            c: None, // No bound on alphas for hard margin
            ..params
        })
    }
}

/// Soft margin SVM (allows some misclassification via C parameter).
pub struct SoftMarginSVM;

impl SoftMarginSVM {
    /// Train a soft margin SVM using SMO algorithm.
    pub fn train(
        x: &[DVector<f64>],
        y: &[f64],
        params: SVMParams,
    ) -> SVM {
        let c = params.c.unwrap_or(1.0);
        train_smo(x, y, SVMParams {
            c: Some(c),
            ..params
        })
    }
}

/// Simplified SMO (Sequential Minimal Optimization) training.
fn train_smo(
    x: &[DVector<f64>],
    y: &[f64],
    params: SVMParams,
) -> SVM {
    let n = x.len();
    let mut alphas = vec![0.0; n];
    let mut bias = 0.0;
    let c = params.c.unwrap_or(f64::INFINITY);

    // Error cache
    let mut errors: Vec<f64> = vec![0.0; n];
    for i in 0..n {
        errors[i] = -y[i]; // Initially all alphas are 0
    }

    let mut iter = 0;
    let mut num_changed = 0;
    let mut examine_all = true;

    while (num_changed > 0 || examine_all) && iter < params.max_iter {
        num_changed = 0;
        if examine_all {
            for i in 0..n {
                num_changed += examine_example(i, x, y, &mut alphas, &mut bias, &mut errors, c, &params);
            }
        } else {
            for i in 0..n {
                if alphas[i] > 0.0 && alphas[i] < c {
                    num_changed += examine_example(i, x, y, &mut alphas, &mut bias, &mut errors, c, &params);
                }
            }
        }

        if examine_all {
            examine_all = false;
        } else if num_changed == 0 {
            examine_all = true;
        }
        iter += 1;
    }

    // Extract support vectors
    let mut support_vectors = Vec::new();
    let mut sv_data = Vec::new();
    let mut sv_labels = Vec::new();
    let mut sv_alphas = Vec::new();

    for i in 0..n {
        if alphas[i] > 1e-8 {
            support_vectors.push(i);
            sv_data.push(x[i].clone());
            sv_labels.push(y[i]);
            sv_alphas.push(alphas[i]);
        }
    }

    SVM {
        support_vectors,
        alphas: sv_alphas,
        bias,
        sv_data,
        sv_labels,
        params,
        n_iterations: iter,
    }
}

fn examine_example(
    i2: usize,
    x: &[DVector<f64>],
    y: &[f64],
    alphas: &mut Vec<f64>,
    bias: &mut f64,
    errors: &mut Vec<f64>,
    c: f64,
    params: &SVMParams,
) -> usize {
    let y2 = y[i2];
    let alpha2 = alphas[i2];
    let e2 = errors[i2];
    let r2 = e2 * y2;

    if (r2 < -params.tol && alpha2 < c) || (r2 > params.tol && alpha2 > 0.0) {
        // Try to find a good i1 using heuristic
        let n = x.len();

        // Find maximum step
        let mut best_i1 = 0;
        let mut best_step = 0.0;
        for j in 0..n {
            if j == i2 { continue; }
            if alphas[j] > 0.0 && alphas[j] < c {
                let step = (errors[j] - e2).abs();
                if step > best_step {
                    best_step = step;
                    best_i1 = j;
                }
            }
        }

        if best_step > 0.0 {
            if take_step(best_i1, i2, x, y, alphas, bias, errors, c, params) {
                return 1;
            }
        }

        // Try all non-bound examples
        for j in 0..n {
            if j == i2 { continue; }
            if alphas[j] > 0.0 && alphas[j] < c {
                if take_step(j, i2, x, y, alphas, bias, errors, c, params) {
                    return 1;
                }
            }
        }

        // Try all examples
        for j in 0..n {
            if j == i2 { continue; }
            if take_step(j, i2, x, y, alphas, bias, errors, c, params) {
                return 1;
            }
        }
    }
    0
}

fn kernel_val(x: &DVector<f64>, y: &DVector<f64>, gamma: Option<f64>) -> f64 {
    match gamma {
        Some(g) => (-g * (x - y).norm_squared()).exp(),
        None => x.dot(y),
    }
}

fn take_step(
    i1: usize,
    i2: usize,
    x: &[DVector<f64>],
    y: &[f64],
    alphas: &mut Vec<f64>,
    bias: &mut f64,
    errors: &mut Vec<f64>,
    c: f64,
    params: &SVMParams,
) -> bool {
    if i1 == i2 { return false; }

    let alpha1_old = alphas[i1];
    let alpha2_old = alphas[i2];
    let y1 = y[i1];
    let y2 = y[i2];
    let e1 = errors[i1];
    let e2 = errors[i2];

    let s = y1 * y2;

    let (l, h) = if y1 != y2 {
        (0.0_f64.max(alpha2_old - alpha1_old), c.min(c + alpha2_old - alpha1_old))
    } else {
        (0.0_f64.max(alpha2_old + alpha1_old - c), c.min(alpha2_old + alpha1_old))
    };

    if (h - l).abs() < 1e-10 { return false; }

    let k11 = kernel_val(&x[i1], &x[i1], params.gamma);
    let k12 = kernel_val(&x[i1], &x[i2], params.gamma);
    let k22 = kernel_val(&x[i2], &x[i2], params.gamma);

    let eta = 2.0 * k12 - k11 - k22;
    let alpha2_new = if eta < 0.0 {
        (h.min(l.max(alpha2_old - y2 * (e1 - e2) / eta)))
    } else {
        // Compute objective at bounds
        let f1 = y1 * (e1 + *bias) - alpha1_old * k11 - s * alpha2_old * k12;
        let f2 = y2 * (e2 + *bias) - s * alpha1_old * k12 - alpha2_old * k22;
        let l1 = alpha1_old + s * (alpha2_old - l);
        let h1 = alpha1_old + s * (alpha2_old - h);
        let lobj = l1 * f1 + l * f2 + 0.5 * l1 * l1 * k11 + 0.5 * l * l * k22 + s * l * l1 * k12;
        let hobj = h1 * f1 + h * f2 + 0.5 * h1 * h1 * k11 + 0.5 * h * h * k22 + s * h * h1 * k12;
        if lobj < hobj - 1e-8 { l } else if hobj < lobj - 1e-8 { h } else { alpha2_old }
    };

    if (alpha2_new - alpha2_old).abs() < 1e-8 * (alpha2_new + alpha2_old + 1e-8) {
        return false;
    }

    let alpha1_new = alpha1_old + s * (alpha2_old - alpha2_new);

    // Clamp alpha1
    let alpha1_new = alpha1_new.max(0.0).min(c);

    // Update bias
    let b1 = *bias - e1 - y1 * (alpha1_new - alpha1_old) * k11 - y2 * (alpha2_new - alpha2_old) * k12;
    let b2 = *bias - e2 - y1 * (alpha1_new - alpha1_old) * k12 - y2 * (alpha2_new - alpha2_old) * k22;

    if alpha1_new > 0.0 && alpha1_new < c {
        *bias = b1;
    } else if alpha2_new > 0.0 && alpha2_new < c {
        *bias = b2;
    } else {
        *bias = (b1 + b2) / 2.0;
    }

    // Update alphas
    alphas[i1] = alpha1_new;
    alphas[i2] = alpha2_new;

    // Update error cache
    for i in 0..x.len() {
        if alphas[i] > 0.0 && alphas[i] < c && i != i1 && i != i2 {
            // Don't update non-bound examples' errors; recompute on demand
        }
        errors[i] = 0.0;
        for j in 0..x.len() {
            if alphas[j] > 1e-10 {
                errors[i] += alphas[j] * y[j] * kernel_val(&x[j], &x[i], params.gamma);
            }
        }
        errors[i] += *bias - y[i];
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn make_linearly_separable() -> (Vec<DVector<f64>>, Vec<f64>) {
        let x = vec![
            DVector::from_vec(vec![0.0, 0.0]),
            DVector::from_vec(vec![1.0, 0.0]),
            DVector::from_vec(vec![0.0, 1.0]),
            DVector::from_vec(vec![1.0, 1.0]),
            DVector::from_vec(vec![2.0, 2.0]),
            DVector::from_vec(vec![3.0, 3.0]),
            DVector::from_vec(vec![-1.0, -1.0]),
            DVector::from_vec(vec![-2.0, -2.0]),
        ];
        let y = vec![-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0];
        (x, y)
    }

    #[test]
    fn test_hard_margin_svm_trains() {
        let (x, y) = make_linearly_separable();
        let svm = HardMarginSVM::train(&x, &y, SVMParams::default());
        assert!(!svm.support_vectors.is_empty());
        assert!(svm.n_iterations > 0);
    }

    #[test]
    fn test_hard_margin_svm_predicts() {
        let (x, y) = make_linearly_separable();
        let svm = HardMarginSVM::train(&x, &y, SVMParams::default());

        // Test on training data (should get most right)
        let correct = x.iter().zip(y.iter())
            .filter(|(xi, yi)| svm.predict(xi) == **yi)
            .count();
        assert!(correct as f64 / x.len() as f64 > 0.7);
    }

    #[test]
    fn test_soft_margin_svm_trains() {
        let (x, y) = make_linearly_separable();
        let params = SVMParams { c: Some(1.0), ..SVMParams::default() };
        let svm = SoftMarginSVM::train(&x, &y, params);
        assert!(svm.alphas.len() > 0);
    }

    #[test]
    fn test_svm_margin_positive() {
        let (x, y) = make_linearly_separable();
        let svm = HardMarginSVM::train(&x, &y, SVMParams::default());
        let margin = svm.margin();
        assert!(margin > 0.0);
        assert!(margin.is_finite());
    }

    #[test]
    fn test_svm_bias_exists() {
        let (x, y) = make_linearly_separable();
        let svm = HardMarginSVM::train(&x, &y, SVMParams::default());
        // Bias should be some finite value
        assert!(svm.bias.is_finite());
    }

    #[test]
    fn test_svm_decision_function_sign() {
        let (x, y) = make_linearly_separable();
        let svm = HardMarginSVM::train(&x, &y, SVMParams::default());

        // Positive class point should have positive decision value
        let pos_point = DVector::from_vec(vec![3.0, 3.0]);
        let neg_point = DVector::from_vec(vec![-1.0, -1.0]);
        // These should have consistent signs (at least on training data)
        let d_pos = svm.decision_function(&pos_point);
        let d_neg = svm.decision_function(&neg_point);
        // They should have different signs for a good SVM
        assert!(d_pos > d_neg);
    }

    #[test]
    fn test_svm_with_rbf_kernel() {
        let (x, y) = make_linearly_separable();
        let params = SVMParams {
            c: Some(1.0),
            gamma: Some(1.0),
            ..SVMParams::default()
        };
        let svm = SoftMarginSVM::train(&x, &y, params);
        assert!(!svm.support_vectors.is_empty());
    }
}
