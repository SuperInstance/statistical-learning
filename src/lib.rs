//! # statistical-learning
//!
//! Statistical learning theory — the mathematical foundations of machine learning.
//!
//! Covers bias-variance decomposition, VC dimension, PAC learning, Rademacher
//! complexity, cross-validation, regularization, kernel methods, SVMs, learning
//! curves, and agent learning models.

pub mod bias_variance;
pub mod cross_validation;
pub mod kernel;
pub mod learning_curves;
pub mod pac_learning;
pub mod rademacher;
pub mod regularization;
pub mod agent_learning;
pub mod svm;
pub mod vc_dimension;

pub use bias_variance::{BiasVarianceDecomposition, bias_variance_decompose};
pub use cross_validation::{CrossValidator, cross_validate_kfold, cross_validate_loo};
pub use kernel::{Kernel, rbf_kernel, polynomial_kernel, linear_kernel, KernelMatrix};
pub use learning_curves::{learning_curve, sample_complexity_estimate, LearningCurvePoint};
pub use pac_learning::{PACBounds, pac_bound_sample_size};
pub use rademacher::{rademacher_complexity, rademacher_complexity_estimated, growth_function_bound};
pub use regularization::{regularize_l1, regularize_l2, regularize_elastic_net, RegularizationResult};
pub use svm::{SVM, SVMParams, HardMarginSVM, SoftMarginSVM};
pub use vc_dimension::{VCDimension, VCBound, compute_vc_bound};
pub use agent_learning::{AgentLearningModel, AgentLearningConfig, AdaptationResult};
