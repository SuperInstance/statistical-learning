# statistical-learning

**Statistical learning in Rust. The math behind ML, without the framework.**

A Rust library implementing the core theoretical tools of statistical learning: bias-variance decomposition, VC dimension, PAC learning, Rademacher complexity, cross-validation, regularization (L1/L2/Elastic Net), kernel methods (RBF, polynomial, linear), SVMs (hard & soft margin), learning curves, and multi-armed bandit learning.

---

## Modules

1. **`bias_variance`** — Decompose prediction error into bias² + variance + noise. Generate theoretical U-shaped tradeoff curves.
2. **`vc_dimension`** — VC dimension descriptors, growth function bounds (Sauer-Shelah), generalization bounds.
3. **`pac_learning`** — PAC sample complexity for realizable and agnostic settings.
4. **`rademacher`** — Empirical Rademacher complexity via Monte Carlo, Massart's lemma, generalization bounds.
5. **`cross_validation`** — K-fold and leave-one-out cross-validation.
6. **`regularization`** — L1 (Lasso), L2 (Ridge), and Elastic Net penalty computation.
7. **`kernel`** — RBF, polynomial, and linear kernels with a `Kernel` trait and Gram matrix computation.
8. **`svm`** — Hard-margin and soft-margin SVMs with SMO-style optimization.
9. **`learning_curves`** — Inverse-root learning curve models, sample complexity estimation.
10. **`agent_learning`** — Multi-armed bandit (ε-greedy + UCB1), regret analysis.

## Install

```toml
[dependencies]
statistical-learning = "0.1.0"
```

## Quick Start

```rust
use statistical_learning::*;
use nalgebra::DVector;

let y_true = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
let predictions = vec![
    DVector::from_vec(vec![1.1, 2.1, 2.9, 3.9]),
    DVector::from_vec(vec![0.9, 1.9, 3.1, 4.1]),
    DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0]),
];

let result = bias_variance_decompose(&y_true, &predictions, 0.05);
println!("Bias² = {:.4}", result.bias_sq);
println!("Variance = {:.4}", result.variance);
```

See the [full API documentation](https://docs.rs/statistical-learning) for all modules.

## License

MIT OR Apache-2.0
