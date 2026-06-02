//! Kernel methods: RBF, polynomial, linear kernels and the kernel trick.

use nalgebra::{DMatrix, DVector};
use serde::{Serialize, Deserialize};

/// A kernel function trait.
pub trait Kernel: Send + Sync {
    /// Compute the kernel value between two vectors.
    fn compute(&self, x: &DVector<f64>, y: &DVector<f64>) -> f64;

    /// Name of the kernel.
    fn name(&self) -> &str;
}

/// RBF (Gaussian) kernel: k(x, y) = exp(-γ ||x - y||²)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RBFKernel {
    pub gamma: f64,
}

impl RBFKernel {
    pub fn new(gamma: f64) -> Self {
        assert!(gamma > 0.0, "gamma must be positive");
        RBFKernel { gamma }
    }

    /// Create with bandwidth σ: γ = 1/(2σ²)
    pub fn with_sigma(sigma: f64) -> Self {
        assert!(sigma > 0.0);
        RBFKernel { gamma: 1.0 / (2.0 * sigma * sigma) }
    }
}

impl Kernel for RBFKernel {
    fn compute(&self, x: &DVector<f64>, y: &DVector<f64>) -> f64 {
        let diff = x - y;
        (-self.gamma * diff.norm_squared()).exp()
    }

    fn name(&self) -> &str {
        "RBF"
    }
}

/// Polynomial kernel: k(x, y) = (x·y + c)^d
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolynomialKernel {
    pub degree: f64,
    pub constant: f64,
}

impl PolynomialKernel {
    pub fn new(degree: f64, constant: f64) -> Self {
        assert!(degree > 0.0, "degree must be positive");
        PolynomialKernel { degree, constant }
    }
}

impl Kernel for PolynomialKernel {
    fn compute(&self, x: &DVector<f64>, y: &DVector<f64>) -> f64 {
        (x.dot(y) + self.constant).powf(self.degree)
    }

    fn name(&self) -> &str {
        "Polynomial"
    }
}

/// Linear kernel: k(x, y) = x·y
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearKernel;

impl Kernel for LinearKernel {
    fn compute(&self, x: &DVector<f64>, y: &DVector<f64>) -> f64 {
        x.dot(y)
    }

    fn name(&self) -> &str {
        "Linear"
    }
}

/// Convenience functions for creating kernel values directly.
pub fn rbf_kernel(x: &DVector<f64>, y: &DVector<f64>, gamma: f64) -> f64 {
    let diff = x - y;
    (-gamma * diff.norm_squared()).exp()
}

pub fn polynomial_kernel(x: &DVector<f64>, y: &DVector<f64>, degree: f64, constant: f64) -> f64 {
    (x.dot(y) + constant).powf(degree)
}

pub fn linear_kernel(x: &DVector<f64>, y: &DVector<f64>) -> f64 {
    x.dot(y)
}

/// Kernel matrix: computes the Gram matrix K where K[i,j] = k(xᵢ, xⱼ).
pub struct KernelMatrix;

impl KernelMatrix {
    /// Compute the kernel (Gram) matrix from a set of vectors.
    pub fn compute<K: Kernel>(kernel: &K, vectors: &[DVector<f64>]) -> DMatrix<f64> {
        let n = vectors.len();
        let mut data = vec![0.0; n * n];
        for i in 0..n {
            for j in i..n {
                let val = kernel.compute(&vectors[i], &vectors[j]);
                data[i * n + j] = val;
                data[j * n + i] = val;
            }
        }
        DMatrix::from_row_slice(n, n, &data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_rbf_identical_vectors() {
        let x = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let kernel = RBFKernel::new(1.0);
        assert_relative_eq!(kernel.compute(&x, &x), 1.0);
    }

    #[test]
    fn test_rbf_distant_vectors() {
        let x = DVector::from_vec(vec![0.0, 0.0]);
        let y = DVector::from_vec(vec![100.0, 100.0]);
        let kernel = RBFKernel::new(1.0);
        assert!(kernel.compute(&x, &y) < 1e-10);
    }

    #[test]
    fn test_rbf_symmetry() {
        let x = DVector::from_vec(vec![1.0, 2.0]);
        let y = DVector::from_vec(vec![3.0, 4.0]);
        let kernel = RBFKernel::new(0.5);
        assert_relative_eq!(kernel.compute(&x, &y), kernel.compute(&y, &x));
    }

    #[test]
    fn test_rbf_decreases_with_distance() {
        let x = DVector::from_vec(vec![0.0]);
        let y1 = DVector::from_vec(vec![1.0]);
        let y2 = DVector::from_vec(vec![2.0]);
        let kernel = RBFKernel::new(1.0);
        assert!(kernel.compute(&x, &y1) > kernel.compute(&x, &y2));
    }

    #[test]
    fn test_polynomial_kernel_degree1() {
        let x = DVector::from_vec(vec![1.0, 2.0]);
        let y = DVector::from_vec(vec![3.0, 4.0]);
        let kernel = PolynomialKernel::new(1.0, 0.0);
        // Should be x·y = 1*3 + 2*4 = 11
        assert_relative_eq!(kernel.compute(&x, &y), 11.0);
    }

    #[test]
    fn test_polynomial_kernel_degree2() {
        let x = DVector::from_vec(vec![1.0, 2.0]);
        let y = DVector::from_vec(vec![3.0, 4.0]);
        let kernel = PolynomialKernel::new(2.0, 1.0);
        // (11 + 1)^2 = 144
        assert_relative_eq!(kernel.compute(&x, &y), 144.0);
    }

    #[test]
    fn test_linear_kernel() {
        let x = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let y = DVector::from_vec(vec![4.0, 5.0, 6.0]);
        let kernel = LinearKernel;
        assert_relative_eq!(kernel.compute(&x, &y), 32.0);
    }

    #[test]
    fn test_kernel_matrix_symmetric() {
        let vecs = vec![
            DVector::from_vec(vec![1.0, 0.0]),
            DVector::from_vec(vec![0.0, 1.0]),
            DVector::from_vec(vec![1.0, 1.0]),
        ];
        let kernel = RBFKernel::new(1.0);
        let kmat = KernelMatrix::compute(&kernel, &vecs);
        assert_eq!(kmat.nrows(), 3);
        assert_eq!(kmat.ncols(), 3);
        // Check symmetry
        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(kmat[(i, j)], kmat[(j, i)]);
            }
        }
    }

    #[test]
    fn test_kernel_matrix_diagonal_one() {
        let vecs = vec![
            DVector::from_vec(vec![1.0, 0.0]),
            DVector::from_vec(vec![0.0, 1.0]),
        ];
        let kernel = RBFKernel::new(1.0);
        let kmat = KernelMatrix::compute(&kernel, &vecs);
        assert_relative_eq!(kmat[(0, 0)], 1.0);
        assert_relative_eq!(kmat[(1, 1)], 1.0);
    }

    #[test]
    fn test_rbf_with_sigma() {
        let kernel = RBFKernel::with_sigma(1.0);
        assert_relative_eq!(kernel.gamma, 0.5);
    }

    #[test]
    fn test_convenience_rbf() {
        let x = DVector::from_vec(vec![0.0, 0.0]);
        assert_relative_eq!(rbf_kernel(&x, &x, 1.0), 1.0);
    }

    #[test]
    fn test_convenience_polynomial() {
        let x = DVector::from_vec(vec![1.0, 2.0]);
        let y = DVector::from_vec(vec![3.0, 4.0]);
        assert_relative_eq!(polynomial_kernel(&x, &y, 2.0, 1.0), 144.0);
    }
}
