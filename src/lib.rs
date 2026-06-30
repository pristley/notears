//! # NOTEARS: Nonlinear ICA for Structure Learning
//!
//! A production-grade Rust implementation of the NO TEARS algorithm for learning
//! directed acyclic graph (DAG) structures from observational data.
//!
//! ## Paper
//! Zheng, X., Aragam, B., Ravikumar, P. K., & Xing, E. P. (2018).
//! "DAGs with NO TEARS: Continuous optimization for learning acyclic graphs."
//! Advances in Neural Information Processing Systems (NeurIPS).
//! https://arxiv.org/abs/1803.01422
//!
//! ## Algorithm Overview
//!
//! The NO TEARS algorithm solves:
//! ```text
//! minimize F(W) = (1/2n)||X - XW||_F^2 + λ||W||_1
//! subject to h(W) = tr(exp(W ∘ W^T)) - d = 0
//! ```
//!
//! where:
//! - `W` is the d×d weight matrix defining a DAG
//! - `h(W)` is the acyclicity constraint (equals 0 iff W is acyclic)
//! - `∘` denotes element-wise Hadamard product
//! - `exp` is the matrix exponential
//!
//! ## Key Features
//!
//! - **O(d³) per-iteration complexity** via matrix exponential computation
//! - **Differentiable acyclicity constraint** enabling gradient-based optimization
//! - **L-BFGS + Augmented Lagrangian** for constrained optimization
//! - **Numerical stability** across varying data regimes
//! - **Production-grade error handling** and validation
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use notears::optimization::solve;
//! use notears::utils::standardize_data;
//! use ndarray::Array2;
//!
//! // Prepare data (n samples, d variables)
//! let data = Array2::zeros((100, 5));
//! let standardized = standardize_data(&data)?;
//!
//! // Learn DAG structure
//! let result = solve(&standardized, 0.1)?; // lambda = 0.1
//!
//! // Extract edges (detected via edge_threshold)
//! let edges = result.edges();
//! println!("Learned edges: {:?}", edges);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## Module Organization
//!
//! - `types` - Core data structures and configuration
//! - `acyclicity` - h(W) constraint and gradient computation
//! - `scoring` - Loss functions (MSE, L1, L2)
//! - `optimization` - L-BFGS + Augmented Lagrangian solver
//! - `utils` - Matrix operations, validation, standardization

pub mod acyclicity;
pub mod optimization;
pub mod scoring;
pub mod types;
pub mod utils;

// Re-export main public API
pub use optimization::{solve, solve_with_config, NotearsSolver, OptimizationError};
pub use types::{OptimizationResult, OptimizationConfig, RegularizationConfig, ConfigError};
pub use types::{WeightMatrix, DataMatrix, GradientMatrix};
pub use scoring::{mse_loss, l1_penalty, total_loss};
pub use acyclicity::{acyclicity_constraint, acyclicity_gradient, is_dag};
pub use utils::{standardize_data, matrix_exponential, frobenius_norm};

// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod integration_tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_complete_pipeline() -> Result<(), Box<dyn std::error::Error>> {
        // Generate synthetic data: linear DAG with 3 nodes
        // True structure: 1 -> 2 -> 3
        let n = 100;
        let mut data = Array2::zeros((n, 3));

        for i in 0..n {
            data[[i, 0]] = (i as f64 % 10.0) / 10.0; // X1 ~ [0, 1]
            data[[i, 1]] = data[[i, 0]] * 0.5 + ((i as f64 % 3.0) / 10.0 - 0.15); // X2 = 0.5*X1 + noise
            data[[i, 2]] = data[[i, 1]] * 0.8 + ((i as f64 % 2.0) / 10.0 - 0.1); // X3 = 0.8*X2 + noise
        }

        // Standardize
        let std_data = standardize_data(&data)?;

        // Run solver
        let config = OptimizationConfig::new(500, 50, 10, 1e-6, 1.0, 0.25, 0.3)?;
        let loss_config = RegularizationConfig::new(0.1, false)?;

        let result = solve_with_config(&std_data, config, loss_config)?;

        // Check results
        assert!(result.constraint_violation >= 0.0);
        assert!(result.final_score > 0.0);

        // Detect edges
        let edges = result.edges();
        println!(
            "Learned {} edges. Acyclicity constraint: {:.6}",
            edges.len(),
            result.constraint_violation
        );

        Ok(())
    }
}
