//! VC dimension and generalization bounds.

use serde::{Serialize, Deserialize};

/// Represents a hypothesis class and its VC dimension.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VCDimension {
    /// Name of the hypothesis class.
    pub name: String,
    /// The VC dimension d.
    pub d: usize,
    /// Optional description of the class.
    pub description: Option<String>,
}

impl VCDimension {
    /// Create a new VC dimension descriptor.
    pub fn new(name: &str, d: usize) -> Self {
        VCDimension {
            name: name.to_string(),
            d,
            description: None,
        }
    }

    /// Compute the growth function bound: m_H(n) ≤ Σ_{i=0}^{d} C(n, i).
    pub fn growth_function_bound(&self, n: usize) -> f64 {
        (0..=self.d.min(n))
            .map(|i| comb(n, i) as f64)
            .sum()
    }

    /// Check if the VC dimension is finite.
    pub fn is_agnostic_learnable(&self) -> bool {
        self.d < usize::MAX
    }
}

/// Generalization bound based on VC dimension.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VCBound {
    /// VC dimension.
    pub d: usize,
    /// Number of training samples.
    pub n: usize,
    /// Confidence parameter δ.
    pub delta: f64,
    /// The generalization bound ε.
    pub epsilon: f64,
}

/// Compute VC generalization bound.
///
/// With probability at least 1 - δ:
///   |R(h) - R_emp(h)| ≤ ε
/// where ε = sqrt((8/n)(d ln(2en/d) + ln(4/δ)))
pub fn compute_vc_bound(d: usize, n: usize, delta: f64) -> VCBound {
    assert!(delta > 0.0 && delta < 1.0, "delta must be in (0, 1)");
    assert!(n > 0, "n must be positive");
    assert!(d <= n, "VC dimension should not exceed sample size for meaningful bounds");

    let epsilon = ((8.0 / n as f64)
        * (d as f64 * (2.0 * (n as f64) / d as f64).ln()
            + (4.0 / delta).ln()))
    .sqrt();

    VCBound {
        d,
        n,
        delta,
        epsilon,
    }
}

/// Compute the sample complexity from VC dimension.
///
/// Sample complexity: minimum number of samples needed for generalization
/// gap ≤ ε with probability 1 - δ.
pub fn sample_complexity_vc(d: usize, epsilon: f64, delta: f64) -> usize {
    assert!(epsilon > 0.0 && epsilon < 1.0);
    assert!(delta > 0.0 && delta < 1.0);

    // Iterate to find minimum n
    for n in (d * 2)..1000000 {
        let bound = compute_vc_bound(d, n, delta);
        if bound.epsilon <= epsilon {
            return n;
        }
    }
    1000000
}

/// VC dimension for common hypothesis classes.
pub struct VCClasses;

impl VCClasses {
    /// Intervals on the real line: VC dim = 2.
    pub fn intervals() -> VCDimension {
        VCDimension::new("Intervals on R", 2)
    }

    /// Half-lines (rays) on the real line: VC dim = 1.
    pub fn half_lines() -> VCDimension {
        VCDimension::new("Half-lines on R", 1)
    }

    /// Linear classifiers in R^d: VC dim = d + 1.
    pub fn linear_classifiers(d: usize) -> VCDimension {
        VCDimension::new(&format!("Linear classifiers in R^{}", d), d + 1)
    }

    /// Axis-aligned rectangles in R^d: VC dim = 2d.
    pub fn axis_aligned_rectangles(d: usize) -> VCDimension {
        VCDimension::new(&format!("Axis-aligned rectangles in R^{}", d), 2 * d)
    }

    /// Finite hypothesis class of size |H|: VC dim ≤ log₂(|H|).
    pub fn finite_class(log2_size: usize) -> VCDimension {
        VCDimension::new("Finite hypothesis class", log2_size)
    }
}

/// Compute binomial coefficient C(n, k).
fn comb(n: usize, k: usize) -> u64 {
    if k > n {
        return 0;
    }
    if k > n - k {
        return comb(n, n - k);
    }
    let mut result: u64 = 1;
    for i in 0..k {
        result *= (n - i) as u64;
        result /= (i + 1) as u64;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_vc_dimension_linear_classifiers() {
        let vc = VCClasses::linear_classifiers(2);
        assert_eq!(vc.d, 3);
        assert_eq!(vc.name, "Linear classifiers in R^2");
    }

    #[test]
    fn test_vc_dimension_intervals() {
        let vc = VCClasses::intervals();
        assert_eq!(vc.d, 2);
    }

    #[test]
    fn test_vc_bound_decreases_with_n() {
        let b1 = compute_vc_bound(3, 100, 0.05);
        let b2 = compute_vc_bound(3, 1000, 0.05);
        assert!(b2.epsilon < b1.epsilon);
    }

    #[test]
    fn test_vc_bound_increases_with_d() {
        let b1 = compute_vc_bound(3, 100, 0.05);
        let b2 = compute_vc_bound(10, 100, 0.05);
        assert!(b2.epsilon > b1.epsilon);
    }

    #[test]
    fn test_vc_bound_positive() {
        let bound = compute_vc_bound(5, 200, 0.1);
        assert!(bound.epsilon > 0.0);
    }

    #[test]
    fn test_growth_function_bound() {
        let vc = VCDimension::new("test", 2);
        // For d=2, n=3: C(3,0) + C(3,1) + C(3,2) = 1 + 3 + 3 = 7
        assert_eq!(vc.growth_function_bound(3), 7.0);
    }

    #[test]
    fn test_growth_function_sauer_shelah() {
        // Growth function should be ≤ n^d for n >= d
        let vc = VCDimension::new("test", 3);
        let n = 10usize;
        let gf = vc.growth_function_bound(n);
        assert!(gf <= (n.pow(3) as f64) + 1.0); // Sauer's lemma
    }

    #[test]
    fn test_sample_complexity_vc() {
        let n = sample_complexity_vc(3, 0.1, 0.05);
        assert!(n > 0);
        // Verify: the bound at this sample size should be ≤ epsilon
        let bound = compute_vc_bound(3, n, 0.05);
        assert!(bound.epsilon <= 0.1 + 1e-10);
    }

    #[test]
    fn test_comb() {
        assert_eq!(comb(5, 2), 10);
        assert_eq!(comb(10, 3), 120);
        assert_eq!(comb(4, 0), 1);
        assert_eq!(comb(4, 4), 1);
    }

    #[test]
    fn test_axis_aligned_rectangles() {
        let vc = VCClasses::axis_aligned_rectangles(2);
        assert_eq!(vc.d, 4);
    }

    #[test]
    fn test_finite_class_vc() {
        let vc = VCClasses::finite_class(10);
        assert_eq!(vc.d, 10);
    }
}
