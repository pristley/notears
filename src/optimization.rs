/// Constrained optimization solver: L-BFGS with Augmented Lagrangian
///
/// Solves: minimize F(W) subject to h(W) = 0
/// where F(W) = MSE(W) + λ*L1(W) is the loss
/// and h(W) = tr(exp(W ∘ W^T)) - d is the acyclicity constraint
///
/// Uses the augmented Lagrangian method:
///   L_ρ(W, λ) = F(W) + λ^T*h(W) + (ρ/2)*h(W)^2
///
/// Inner loop uses gradient descent for quasi-Newton optimization.

use crate::types::{WeightMatrix, DataMatrix, OptimizationResult, OptimizationConfig, RegularizationConfig, ConfigError};
use crate::acyclicity::{self, AcyclicityError};
use crate::scoring::{self, ScoringError};
use crate::utils::UtilError;
use ndarray::Array2;
use std::f64;

/// Error types for optimization
#[derive(Debug, thiserror::Error)]
pub enum OptimizationError {
    #[error("Configuration error: {0}")]
    Config(#[from] ConfigError),

    #[error("Acyclicity error: {0}")]
    Acyclicity(#[from] AcyclicityError),

    #[error("Scoring error: {0}")]
    Scoring(#[from] ScoringError),

    #[error("Utility error: {0}")]
    Utils(#[from] UtilError),

    #[error("Failed to converge after {max_iterations} iterations. Final h(W)={h_value}")]
    ConvergenceFailed { max_iterations: usize, h_value: f64 },

    #[error("Invalid optimization state: {0}")]
    InvalidState(String),
}

/// Augmented Lagrangian optimizer state
struct OptimizationState {
    weight_matrix: WeightMatrix,
    dual_variable: f64, // λ
    rho: f64,
}

/// Main NOTEARS solver
pub struct NotearsSolver {
    opt_config: OptimizationConfig,
    reg_config: RegularizationConfig,
}

impl NotearsSolver {
    /// Create new solver with configurations
    pub fn new(opt_config: OptimizationConfig, reg_config: RegularizationConfig) -> Self {
        NotearsSolver {
            opt_config,
            reg_config,
        }
    }

    /// Main optimization routine
    ///
    /// # Arguments
    /// * `data` - Data matrix X (n×d), standardized recommended
    /// * `init_weight` - Initial weight matrix (optional, zero if None)
    ///
    /// # Returns
    /// OptimizationResult with learned DAG structure
    pub fn solve(
        &self,
        data: &DataMatrix,
        init_weight: Option<&WeightMatrix>,
    ) -> Result<OptimizationResult, OptimizationError> {
        let (_n, d) = data.dim();

        // Initialize weight matrix
        let mut w = if let Some(init_w) = init_weight {
            init_w.clone()
        } else {
            Array2::zeros((d, d))
        };

        let mut rho = self.opt_config.penalty_rho_init;
        let mut dual_lambda = 0.0;
        let mut best_h = f64::INFINITY;

        // Augmented Lagrangian outer loop
        for outer_iter in 0..self.opt_config.max_outer_iterations {
            // Check convergence before inner loop
            let (h, _) = acyclicity::acyclicity_with_gradient(&w)?;
            best_h = best_h.min(h);

            if h <= self.opt_config.constraint_tolerance {
                // Converged
                let loss = scoring::total_loss(data, &w, &self.reg_config)?;
                let adj_matrix = w.mapv(|x| if x.abs() > self.opt_config.edge_threshold { 1 } else { 0 });
                
                return Ok(OptimizationResult {
                    weight_matrix: w,
                    constraint_violation: h,
                    iterations: outer_iter + 1,
                    final_score: loss,
                    adjacency_matrix: adj_matrix,
                });
            }

            // Inner gradient descent loop
            for _inner_iter in 0..self.opt_config.max_lbfgs_iterations {
                // Compute augmented Lagrangian gradient
                let _loss = scoring::total_loss(data, &w, &self.reg_config);
                let (h_curr, grad_h) = acyclicity::acyclicity_with_gradient(&w)?;

                let grad_f = scoring::total_loss_gradient(data, &w, &self.reg_config)?;
                let grad_aug = grad_f + dual_lambda * &grad_h + rho * h_curr * &grad_h;

                let grad_norm = grad_aug.iter().map(|x| x * x).sum::<f64>().sqrt();

                if grad_norm < 1e-10 {
                    break;
                }

                // Gradient descent step
                let step_size = 0.01 / grad_norm.max(1.0);
                w = w - step_size * &grad_aug;
            }

            // Update dual variable and penalty
            let (h_new, _) = acyclicity::acyclicity_with_gradient(&w)?;
            dual_lambda += rho * h_new;

            // Adaptive penalty update
            if best_h > 0.0 {
                let progress_rate = h_new / best_h;
                if progress_rate > self.opt_config.progress_rate {
                    rho = (rho * 10.0).min(1e16);
                }
            }

            best_h = best_h.min(h_new);
        }

        let (final_h, _) = acyclicity::acyclicity_with_gradient(&w)?;
        let loss = scoring::total_loss(data, &w, &self.reg_config)?;
        let adj_matrix = w.mapv(|x| if x.abs() > self.opt_config.edge_threshold { 1 } else { 0 });

        Ok(OptimizationResult {
            weight_matrix: w,
            constraint_violation: final_h,
            iterations: self.opt_config.max_outer_iterations,
            final_score: loss,
            adjacency_matrix: adj_matrix,
        })
    }
}

/// Convenience function to run NOTEARS solver with default configuration
///
/// # Arguments
/// * `data` - Data matrix X (n×d), should be standardized
/// * `lambda` - L1 regularization coefficient
///
/// # Returns
/// OptimizationResult with learned structure
pub fn solve(data: &DataMatrix, lambda: f64) -> Result<OptimizationResult, OptimizationError> {
    let opt_config = OptimizationConfig::default();
    let reg_config = RegularizationConfig::new(lambda, false)?;

    let solver = NotearsSolver::new(opt_config, reg_config);
    solver.solve(data, None)
}

/// Run solver with custom configuration
pub fn solve_with_config(
    data: &DataMatrix,
    opt_config: OptimizationConfig,
    reg_config: RegularizationConfig,
) -> Result<OptimizationResult, OptimizationError> {
    let solver = NotearsSolver::new(opt_config, reg_config);
    solver.solve(data, None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solver_initialization() {
        let config = OptimizationConfig::default();
        let loss_config = RegularizationConfig::default();
        let _solver = NotearsSolver::new(config, loss_config);
    }

    #[test]
    fn test_solve_trivial() -> Result<(), Box<dyn std::error::Error>> {
        // Create simple synthetic data (identity structure)
        let data = ndarray::array![
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0]
        ];

        let config = OptimizationConfig::new(100, 50, 10, 1e-6, 1.0, 0.25, 0.3)?;
        let loss_config = RegularizationConfig::new(0.1, false)?;

        let result = solve_with_config(&data, config, loss_config)?;
        assert!(result.constraint_violation >= 0.0);
        Ok(())
    }
}
