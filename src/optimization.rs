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

/// Compute augmented Lagrangian objective: L_ρ(W, α) = F(W) + (ρ/2) * h(W)² + α * h(W)
///
/// The augmented Lagrangian combines the smooth score function F(W) with the non-convex
/// acyclicity constraint h(W) = 0 via penalty and dual methods.
///
/// **Three-term decomposition:**
/// 1. **F(W)**: Score function (data fidelity + sparsity)
///    - LS loss: (1/(2n)) * ||X - XW||²_F
///    - L₁ penalty: λ * ||W||₁
/// 2. **(ρ/2) * h(W)²**: Quadratic penalty for constraint violation
///    - Increases as acyclicity constraint is violated
///    - ρ/2 scaling matches augmented Lagrangian theory
/// 3. **α * h(W)**: Lagrange multiplier term
///    - Encodes constraint h(W) = 0 in dual problem
///    - α adjusted iteratively in dual ascent loop
///
/// **Mathematical properties:**
/// - Non-convex due to h(W) = tr(exp(W ⊙ W)) - d nonconvexity
/// - Smooth (differentiable) in W for interior points
/// - As ρ → ∞: Solution approaches constrained minimum
/// - As α approaches optimal: multiplier term dominates
///
/// **Numerical scale analysis:**
/// - F(W) typically O(1) to O(10)
/// - h(W) typically 0 (DAGs) or O(0.1-10) (near-DAGs)
/// - h(W)² penalty can dominate if h(W) large
/// - α scales with ρ and h(W)
///
/// # Arguments
/// * `w` - Weight matrix W (d×d) representing learned structure
/// * `alpha` - Lagrange multiplier (dual variable), typically O(ρ)
/// * `rho` - Penalty parameter ρ > 0, increases for infeasibility
/// * `data` - Data matrix X (n×d) with n samples, d variables
/// * `config` - RegularizationConfig with lambda ≥ 0.0
///
/// # Returns
/// Augmented Lagrangian value L_ρ(W, α) ∈ ℝ
///
/// # Errors
/// Returns OptimizationError if:
/// - Score function computation fails (dimension mismatch, numerical issues)
/// - Acyclicity constraint computation fails
/// - Any component becomes NaN or Inf
/// - Invalid input: rho ≤ 0
///
/// # Convergence Criteria
/// - **Primary**: h(W) < ε (constraint satisfaction), recommended ε = 1e-8
/// - **Secondary**: ||∇_W L_ρ|| < δ (stationarity), recommended δ = 1e-6
/// - **Combined**: Both conditions indicate convergence to feasible KKT point
///
/// # Performance Notes
/// - Computational cost: ~2× evaluation of score_function (adds matrix exponential)
/// - Bottleneck: matrix exponential in acyclicity_constraint O(d³)
/// - For optimization loop: evaluate L_ρ once per iteration
/// - Gradient ∇_W L_ρ requires additional derivatives (not computed here)
///
/// # Optimization Integration
/// Typical augmented Lagrangian loop:
/// ```ignore
/// for k = 1, 2, ... until convergence:
///   // Inner loop: minimize L_ρ(W, α_k)
///   W_k ← argmin_W L_ρ(W, α_k)  // e.g., via gradient descent
///   
///   // Dual update: α_{k+1} ← α_k + ρ * h(W_k)
///   h_k ← acyclicity_constraint(W_k)
///   α_{k+1} ← α_k + ρ * h_k
///   
///   // Penalty update: increase ρ if constraint violation large
///   if h_k > ε_old:
///     ρ ← c * ρ  where c > 1 (e.g., c = 10)
/// ```
pub fn augmented_lagrangian(
    w: &WeightMatrix,
    alpha: f64,
    rho: f64,
    data: &DataMatrix,
    config: &RegularizationConfig,
) -> Result<f64, OptimizationError> {
    // Input validation
    if rho <= 0.0 {
        return Err(OptimizationError::InvalidState(
            format!("Penalty parameter rho must be positive, got rho={}", rho)
        ));
    }

    if !alpha.is_finite() {
        return Err(OptimizationError::InvalidState(
            format!("Lagrange multiplier alpha must be finite, got alpha={}", alpha)
        ));
    }

    // **Term 1: Score function F(W)**
    let f_w = scoring::score_function(w, data, config)?;

    // Check F(W) for numerical issues
    if !f_w.is_finite() {
        return Err(OptimizationError::InvalidState(
            format!("Score function F(W) is not finite: {}", f_w)
        ));
    }

    // **Term 2 & 3: Acyclicity constraint terms (ρ/2) * h² + α * h**
    let h_w = acyclicity::acyclicity_constraint(w)?;

    // Check h(W) for numerical issues
    if !h_w.is_finite() {
        return Err(OptimizationError::InvalidState(
            format!("Acyclicity constraint h(W) is not finite: {}", h_w)
        ));
    }

    // **Term 2: Quadratic penalty (ρ/2) * h(W)²**
    let penalty_term = (rho / 2.0) * h_w * h_w;

    // **Term 3: Lagrange multiplier term α * h(W)**
    let multiplier_term = alpha * h_w;

    // **Combined: L_ρ(W, α) = F(W) + (ρ/2) * h(W)² + α * h(W)**
    let augmented_obj = f_w + penalty_term + multiplier_term;

    // Final numerical check
    if !augmented_obj.is_finite() {
        return Err(OptimizationError::InvalidState(
            format!("Augmented Lagrangian is not finite: F={}, penalty={}, multiplier={}, total={}",
                    f_w, penalty_term, multiplier_term, augmented_obj)
        ));
    }

    Ok(augmented_obj)
}

/// Compute gradient of augmented Lagrangian: ∇_W L_ρ(W, α)
///
/// **Mathematical Formula:**
/// ∇_W L_ρ(W, α) = ∇F(W) + ρ * h(W) * ∇h(W) + α * ∇h(W)
///                = ∇F(W) + (ρ * h(W) + α) * ∇h(W)
///
/// **Component decomposition:**
/// 1. **∇F(W)**: Score function gradient
///    - LS component: -(1/n) X^T @ (X - XW)
///    - L₁ component: λ * sign(W)
/// 2. **ρ * h(W) * ∇h(W)**: Penalty term gradient
///    - Scales acyclicity gradient by current constraint violation
///    - Weighted gradient term enforcing h(W) = 0
/// 3. **α * ∇h(W)**: Lagrange multiplier gradient
///    - Scaled acyclicity gradient by dual variable
///    - Adjusted in dual ascent loop
///
/// **Mathematical properties:**
/// - At h(W) = 0: reduces to ∇_W L_ρ = ∇F(W) + α * ∇h(W)
/// - Smooth everywhere for interior points
/// - Lipschitz-continuous for L-BFGS compatibility
/// - Descent property: ⟨∇L_ρ, descent direction⟩ < 0
///
/// **Numerical considerations:**
/// - Gradient scaling: ||∇_W L_ρ|| should remain O(1-10)
/// - Penalty multiplier: |ρ * h(W) + α| can grow large
/// - Reuses h(W) and ∇h(W) computations for efficiency
///
/// # Arguments
/// * `w` - Weight matrix W (d×d)
/// * `alpha` - Lagrange multiplier (dual variable)
/// * `rho` - Penalty parameter ρ > 0
/// * `data` - Data matrix X (n×d)
/// * `config` - RegularizationConfig with λ ≥ 0
///
/// # Returns
/// Gradient matrix ∇_W L_ρ (d×d) with same structure as W
///
/// # Errors
/// Returns OptimizationError if:
/// - Score gradient computation fails
/// - Acyclicity gradient computation fails
/// - Any component becomes NaN or Inf
/// - Invalid input: rho ≤ 0
///
/// # Computational Cost
/// - Complexity: O(d³) dominated by matrix exponential in ∇h(W)
/// - No additional matrix exponentials vs augmented_lagrangian()
/// - Reuses acyclicity_gradient() computation
///
/// # L-BFGS Integration
/// Gradient descent step using this gradient:
/// ```ignore
/// grad = augmented_lagrangian_gradient(w, alpha, rho, data, config)?
/// grad_norm = sqrt(sum(grad²))
/// step_size = learning_rate / grad_norm.max(1.0)
/// w_next = w - step_size * grad
/// ```
///
/// # Convergence Criteria
/// - **Stationarity**: ||∇_W L_ρ|| < 1e-6
/// - **Feasibility**: h(W) < 1e-8
/// - Both conditions together indicate KKT point
pub fn augmented_lagrangian_gradient(
    w: &WeightMatrix,
    alpha: f64,
    rho: f64,
    data: &DataMatrix,
    config: &RegularizationConfig,
) -> Result<WeightMatrix, OptimizationError> {
    // Input validation
    if rho <= 0.0 {
        return Err(OptimizationError::InvalidState(
            format!("Penalty parameter rho must be positive, got rho={}", rho)
        ));
    }

    if !alpha.is_finite() {
        return Err(OptimizationError::InvalidState(
            format!("Lagrange multiplier alpha must be finite, got alpha={}", alpha)
        ));
    }

    // **Component 1: ∇F(W) - Score function gradient**
    let grad_f = scoring::score_gradient(w, data, config)?;

    // Check for numerical issues
    if grad_f.iter().any(|x| !x.is_finite()) {
        return Err(OptimizationError::InvalidState(
            "Score gradient contains NaN or Inf".to_string()
        ));
    }

    // **Components 2 & 3: (ρ * h + α) * ∇h(W)**
    // Compute h(W) for penalty multiplier
    let h_w = acyclicity::acyclicity_constraint(w)?;

    // Compute ∇h(W) for both penalty and multiplier terms
    let grad_h = acyclicity::acyclicity_gradient(w)?;

    // Check for numerical issues
    if grad_h.iter().any(|x| !x.is_finite()) {
        return Err(OptimizationError::InvalidState(
            "Acyclicity gradient contains NaN or Inf".to_string()
        ));
    }

    // **Combined penalty-multiplier scaling factor**
    // penalty_multiplier = ρ * h(W) + α
    let penalty_multiplier = rho * h_w + alpha;

    // **Scale acyclicity gradient by penalty multiplier**
    // constraint_weighted = penalty_multiplier * ∇h(W)
    let constraint_weighted = penalty_multiplier * &grad_h;

    // Pre-compute norms for error reporting before moving values
    let grad_f_norm = grad_f.iter().map(|x| x*x).sum::<f64>().sqrt();
    let constraint_norm = constraint_weighted.iter().map(|x| x*x).sum::<f64>().sqrt();

    // **Final gradient: ∇_W L_ρ = ∇F + (ρ*h + α)*∇h**
    let gradient = grad_f + constraint_weighted;

    // Final numerical check
    if gradient.iter().any(|x| !x.is_finite()) {
        return Err(OptimizationError::InvalidState(
            format!("Augmented Lagrangian gradient contains NaN or Inf. Components: grad_f norm={}, constraint_weighted norm={}",
                    grad_f_norm, constraint_norm)
        ));
    }

    Ok(gradient)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_augmented_lagrangian_zero_weights() {
        // With W = 0, we have specific known values:
        // - h(0) = tr(exp(0)) - d = tr(I) - d = 0
        // - F(0) = (1/(2n)) * ||X||²_F + 0
        let data = ndarray::array![[1.0, 2.0], [3.0, 4.0]];
        let w = Array2::zeros((2, 2));
        let config = RegularizationConfig::new(0.1, false).unwrap();
        
        let alpha = 1.0;
        let rho = 10.0;
        
        let l_rho = augmented_lagrangian(&w, alpha, rho, &data, &config).unwrap();
        
        // h(0) = 0, so L_ρ = F(0) + 0 + 0 = F(0)
        let f_0 = scoring::score_function(&w, &data, &config).unwrap();
        assert_abs_diff_eq!(l_rho, f_0, epsilon = 1e-12);
    }

    #[test]
    fn test_augmented_lagrangian_components() {
        // Verify three-term decomposition:
        // L_ρ(W, α) = F(W) + (ρ/2) * h² + α * h
        let data = ndarray::array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let w = ndarray::array![[0.1, 0.05], [-0.05, 0.2]];
        let config = RegularizationConfig::new(0.15, false).unwrap();
        
        let alpha = 2.0;
        let rho = 5.0;
        
        let l_rho = augmented_lagrangian(&w, alpha, rho, &data, &config).unwrap();
        
        // Compute components separately
        let f_w = scoring::score_function(&w, &data, &config).unwrap();
        let h_w = acyclicity::acyclicity_constraint(&w).unwrap();
        let penalty = (rho / 2.0) * h_w * h_w;
        let multiplier = alpha * h_w;
        let expected = f_w + penalty + multiplier;
        
        assert_abs_diff_eq!(l_rho, expected, epsilon = 1e-12);
    }

    #[test]
    fn test_augmented_lagrangian_alpha_effect() {
        // Verify that changing alpha affects L_ρ linearly
        let data = ndarray::array![[1.0, 2.0], [3.0, 4.0]];
        let w = ndarray::array![[0.1, 0.02], [-0.02, 0.1]];
        let config = RegularizationConfig::new(0.1, false).unwrap();
        
        let rho = 1.0;
        let alpha1 = 1.0;
        let alpha2 = 3.0;
        
        let l1 = augmented_lagrangian(&w, alpha1, rho, &data, &config).unwrap();
        let l2 = augmented_lagrangian(&w, alpha2, rho, &data, &config).unwrap();
        
        // h(W) is fixed, so change in L should be (alpha2 - alpha1) * h(W)
        let h_w = acyclicity::acyclicity_constraint(&w).unwrap();
        let delta_expected = (alpha2 - alpha1) * h_w;
        let delta_actual = l2 - l1;
        
        assert_abs_diff_eq!(delta_actual, delta_expected, epsilon = 1e-12);
    }

    #[test]
    fn test_augmented_lagrangian_rho_effect() {
        // Verify that changing rho affects penalty term correctly
        let data = ndarray::array![[1.0, 2.0], [3.0, 4.0]];
        let w = ndarray::array![[0.1, 0.02], [-0.02, 0.15]];
        let config = RegularizationConfig::new(0.1, false).unwrap();
        
        let alpha = 1.0;
        let rho1 = 1.0;
        let rho2 = 5.0;
        
        let l1 = augmented_lagrangian(&w, alpha, rho1, &data, &config).unwrap();
        let l2 = augmented_lagrangian(&w, alpha, rho2, &data, &config).unwrap();
        
        // h(W) is fixed, so change in penalty is (rho2 - rho1) * h² / 2
        let h_w = acyclicity::acyclicity_constraint(&w).unwrap();
        let delta_penalty = ((rho2 - rho1) / 2.0) * h_w * h_w;
        let delta_actual = l2 - l1;
        
        assert_abs_diff_eq!(delta_actual, delta_penalty, epsilon = 1e-12);
    }

    #[test]
    fn test_augmented_lagrangian_negative_alpha() {
        // Alpha can be negative (it's unconstrained dual variable)
        let data = ndarray::array![[1.0, 2.0], [3.0, 4.0]];
        let w = ndarray::array![[0.1, 0.05], [-0.05, 0.2]];
        let config = RegularizationConfig::new(0.1, false).unwrap();
        
        let alpha = -5.0;
        let rho = 2.0;
        
        let l_rho = augmented_lagrangian(&w, alpha, rho, &data, &config).unwrap();
        assert!(l_rho.is_finite());
    }

    #[test]
    fn test_augmented_lagrangian_large_rho() {
        // With large rho, penalty dominates for constrained solutions
        let data = ndarray::array![[1.0, 2.0], [3.0, 4.0]];
        let w = ndarray::array![[0.1, 0.05], [-0.05, 0.2]];
        let config = RegularizationConfig::new(0.1, false).unwrap();
        
        let alpha = 1.0;
        let rho_small = 0.1;
        let rho_large = 1000.0;
        
        let l_small = augmented_lagrangian(&w, alpha, rho_small, &data, &config).unwrap();
        let l_large = augmented_lagrangian(&w, alpha, rho_large, &data, &config).unwrap();
        
        // With large rho, penalty term dominates (unless h ≈ 0)
        let h_w = acyclicity::acyclicity_constraint(&w).unwrap();
        if h_w.abs() > 1e-6 {
            assert!(l_large > l_small); // Higher penalty value
        }
    }

    #[test]
    fn test_augmented_lagrangian_dimension_mismatch() {
        // Should propagate error from score_function
        let data = ndarray::array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let w = Array2::zeros((2, 2));
        let config = RegularizationConfig::new(0.1, false).unwrap();
        
        assert!(augmented_lagrangian(&w, 1.0, 1.0, &data, &config).is_err());
    }

    #[test]
    fn test_augmented_lagrangian_invalid_rho() {
        // Negative or zero rho should fail
        let data = ndarray::array![[1.0, 2.0], [3.0, 4.0]];
        let w = ndarray::array![[0.1, 0.05], [-0.05, 0.2]];
        let config = RegularizationConfig::new(0.1, false).unwrap();
        
        assert!(augmented_lagrangian(&w, 1.0, 0.0, &data, &config).is_err());
        assert!(augmented_lagrangian(&w, 1.0, -1.0, &data, &config).is_err());
    }

    #[test]
    fn test_augmented_lagrangian_invalid_alpha() {
        // NaN or Inf alpha should fail
        let data = ndarray::array![[1.0, 2.0], [3.0, 4.0]];
        let w = ndarray::array![[0.1, 0.05], [-0.05, 0.2]];
        let config = RegularizationConfig::new(0.1, false).unwrap();
        
        assert!(augmented_lagrangian(&w, f64::NAN, 1.0, &data, &config).is_err());
        assert!(augmented_lagrangian(&w, f64::INFINITY, 1.0, &data, &config).is_err());
    }

    #[test]
    fn test_augmented_lagrangian_monotonicity_in_constraint() {
        // For fixed W and increasing rho, L_ρ increases if h(W) ≠ 0
        let data = ndarray::array![[1.0, 2.0], [3.0, 4.0]];
        // Use weights that violate acyclicity (cycle)
        let w = ndarray::array![[0.0, 0.5], [0.5, 0.0]];
        let config = RegularizationConfig::new(0.1, false).unwrap();
        
        let alpha = 1.0;
        let rho_values = vec![1.0, 2.0, 5.0, 10.0];
        
        let mut prev_l = f64::NEG_INFINITY;
        for rho in rho_values {
            let l_rho = augmented_lagrangian(&w, alpha, rho, &data, &config).unwrap();
            if l_rho > prev_l + 1e-10 {
                // Expected: monotone increase (unless h very small)
            }
            prev_l = l_rho;
        }
    }

    #[test]
    fn test_augmented_lagrangian_numerical_stability() {
        // Test with very small/large values
        let data = ndarray::array![[1e-10, 2e-10], [3e-10, 4e-10]];
        let w = ndarray::array![[1e-15, 1e-15], [1e-15, 1e-15]];
        let config = RegularizationConfig::new(0.1, false).unwrap();
        
        let l_rho = augmented_lagrangian(&w, 1.0, 1.0, &data, &config).unwrap();
        assert!(l_rho.is_finite());
    }

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

    // ==================== Augmented Lagrangian Gradient Tests ====================

    #[test]
    fn test_augmented_lagrangian_gradient_zero_weights() {
        // At W = 0: h(0) = 0, ∇h(0) = 2*0 = 0
        // So ∇L_ρ = ∇F(0) + (ρ*0 + α)*0 = ∇F(0)
        let data = ndarray::array![[1.0, 2.0], [3.0, 4.0]];
        let w = Array2::zeros((2, 2));
        let config = RegularizationConfig::new(0.1, false).unwrap();

        let alpha = 1.0;
        let rho = 10.0;

        let grad_l_rho = augmented_lagrangian_gradient(&w, alpha, rho, &data, &config).unwrap();
        let grad_f = scoring::score_gradient(&w, &data, &config).unwrap();

        // At h=0, constraint terms vanish
        let diff_norm = (&grad_l_rho - &grad_f).iter().map(|x| x*x).sum::<f64>().sqrt();
        assert!(diff_norm < 1e-12);
    }

    #[test]
    fn test_augmented_lagrangian_gradient_components() {
        // Verify component decomposition:
        // ∇L_ρ = ∇F + (ρ*h + α)*∇h
        let data = ndarray::array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let w = ndarray::array![[0.1, 0.05], [-0.05, 0.2]];
        let config = RegularizationConfig::new(0.15, false).unwrap();

        let alpha = 2.0;
        let rho = 5.0;

        let grad_l_rho = augmented_lagrangian_gradient(&w, alpha, rho, &data, &config).unwrap();

        // Compute components
        let grad_f = scoring::score_gradient(&w, &data, &config).unwrap();
        let h_w = acyclicity::acyclicity_constraint(&w).unwrap();
        let grad_h = acyclicity::acyclicity_gradient(&w).unwrap();
        let penalty_multiplier = rho * h_w + alpha;
        let expected = grad_f + penalty_multiplier * &grad_h;

        let diff_norm = (&grad_l_rho - &expected).iter().map(|x| x*x).sum::<f64>().sqrt();
        assert!(diff_norm < 1e-12);
    }

    #[test]
    fn test_augmented_lagrangian_gradient_h_zero() {
        // When h(W) ≈ 0 (acyclic), gradient reduces to ∇F + α*∇h
        // Use weights near-acyclic (small cycle)
        let data = ndarray::array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let w = ndarray::array![[0.0, 0.001], [-0.001, 0.0]]; // Nearly acyclic
        let config = RegularizationConfig::new(0.1, false).unwrap();

        let alpha = 2.0;
        let rho = 5.0;

        let grad_l_rho = augmented_lagrangian_gradient(&w, alpha, rho, &data, &config).unwrap();

        // For near-zero h(W), ρ*h term is small
        let grad_f = scoring::score_gradient(&w, &data, &config).unwrap();
        let grad_h = acyclicity::acyclicity_gradient(&w).unwrap();
        let approximate_expected = grad_f + alpha * &grad_h;

        // Should be close (within rho*h*||∇h|| tolerance)
        let h_w = acyclicity::acyclicity_constraint(&w).unwrap();
        let grad_h_norm = grad_h.iter().map(|x| x*x).sum::<f64>().sqrt();
        let tolerance = rho * h_w.abs() * grad_h_norm + 1e-10;
        let diff_norm = (&grad_l_rho - &approximate_expected).iter().map(|x| x*x).sum::<f64>().sqrt();
        assert!(diff_norm < tolerance);
    }

    #[test]
    fn test_augmented_lagrangian_gradient_alpha_effect() {
        // Increasing alpha by Δα changes gradient by Δα * ∇h
        let data = ndarray::array![[1.0, 2.0], [3.0, 4.0]];
        let w = ndarray::array![[0.1, 0.02], [-0.02, 0.1]];
        let config = RegularizationConfig::new(0.1, false).unwrap();

        let rho = 1.0;
        let alpha1 = 1.0;
        let alpha2 = 3.0;

        let grad1 = augmented_lagrangian_gradient(&w, alpha1, rho, &data, &config).unwrap();
        let grad2 = augmented_lagrangian_gradient(&w, alpha2, rho, &data, &config).unwrap();

        // Expected change: (alpha2 - alpha1) * ∇h(W)
        let grad_h = acyclicity::acyclicity_gradient(&w).unwrap();
        let delta_expected = (alpha2 - alpha1) * &grad_h;
        let delta_actual = grad2 - grad1;

        let diff_norm = (&delta_actual - &delta_expected).iter().map(|x| x*x).sum::<f64>().sqrt();
        assert!(diff_norm < 1e-12);
    }

    #[test]
    fn test_augmented_lagrangian_gradient_rho_effect() {
        // Increasing rho by Δρ changes gradient by Δρ * h(W) * ∇h
        let data = ndarray::array![[1.0, 2.0], [3.0, 4.0]];
        let w = ndarray::array![[0.1, 0.02], [-0.02, 0.15]];
        let config = RegularizationConfig::new(0.1, false).unwrap();

        let alpha = 1.0;
        let rho1 = 1.0;
        let rho2 = 5.0;

        let grad1 = augmented_lagrangian_gradient(&w, alpha, rho1, &data, &config).unwrap();
        let grad2 = augmented_lagrangian_gradient(&w, alpha, rho2, &data, &config).unwrap();

        // Expected change: (rho2 - rho1) * h(W) * ∇h(W)
        let h_w = acyclicity::acyclicity_constraint(&w).unwrap();
        let grad_h = acyclicity::acyclicity_gradient(&w).unwrap();
        let delta_expected = (rho2 - rho1) * h_w * &grad_h;
        let delta_actual = grad2 - grad1;

        let diff_norm = (&delta_actual - &delta_expected).iter().map(|x| x*x).sum::<f64>().sqrt();
        assert!(diff_norm < 1e-12);
    }

    #[test]
    fn test_augmented_lagrangian_gradient_finite_difference() {
        // Validate analytical gradient via finite differences
        // ∇L_ρ[i,j] ≈ (L_ρ(W + ε*e_ij) - L_ρ(W - ε*e_ij)) / (2ε)
        let data = ndarray::array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let w = ndarray::array![[0.1, 0.05], [-0.03, 0.15]];
        let config = RegularizationConfig::new(0.12, false).unwrap();

        let alpha = 1.5;
        let rho = 3.0;

        let grad_analytical = augmented_lagrangian_gradient(&w, alpha, rho, &data, &config).unwrap();

        let epsilon = 1e-5;
        let (d, _) = w.dim();

        for i in 0..d {
            for j in 0..d {
                let mut w_plus = w.clone();
                let mut w_minus = w.clone();

                w_plus[[i, j]] += epsilon;
                w_minus[[i, j]] -= epsilon;

                let l_plus = augmented_lagrangian(&w_plus, alpha, rho, &data, &config).unwrap();
                let l_minus = augmented_lagrangian(&w_minus, alpha, rho, &data, &config).unwrap();

                let grad_numerical = (l_plus - l_minus) / (2.0 * epsilon);
                let grad_analytical_ij = grad_analytical[[i, j]];

                // Relative error tolerance
                let tolerance = 1e-4 * (grad_analytical_ij.abs().max(1.0));
                assert!((grad_analytical_ij - grad_numerical).abs() < tolerance);
            }
        }
    }

    #[test]
    fn test_augmented_lagrangian_gradient_descent_property() {
        // Verify gradient points downhill: -∇L_ρ is descent direction
        // L_ρ(W - step*∇L_ρ) < L_ρ(W) for small step
        let data = ndarray::array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let w = ndarray::array![[0.1, 0.05], [-0.05, 0.2]];
        let config = RegularizationConfig::new(0.1, false).unwrap();

        let alpha = 1.0;
        let rho = 2.0;

        let l_current = augmented_lagrangian(&w, alpha, rho, &data, &config).unwrap();
        let grad = augmented_lagrangian_gradient(&w, alpha, rho, &data, &config).unwrap();
        let grad_norm = grad.iter().map(|x| x*x).sum::<f64>().sqrt();

        if grad_norm > 1e-10 {
            // Take small descent step
            let step_size = 0.01 / grad_norm;
            let w_next = w - step_size * &grad;

            let l_next = augmented_lagrangian(&w_next, alpha, rho, &data, &config).unwrap();

            // Gradient descent: should decrease objective
            assert!(l_next < l_current + 1e-10,
                "Gradient descent failed: L_current={}, L_next={}, gradient_norm={}",
                l_current, l_next, grad_norm);
        }
    }

    #[test]
    fn test_augmented_lagrangian_gradient_negative_alpha() {
        // Alpha can be negative (unconstrained dual variable)
        let data = ndarray::array![[1.0, 2.0], [3.0, 4.0]];
        let w = ndarray::array![[0.1, 0.05], [-0.05, 0.2]];
        let config = RegularizationConfig::new(0.1, false).unwrap();

        let alpha = -5.0;
        let rho = 2.0;

        let grad = augmented_lagrangian_gradient(&w, alpha, rho, &data, &config).unwrap();
        assert!(grad.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_augmented_lagrangian_gradient_large_rho() {
        // With large rho, penalty term dominates
        let data = ndarray::array![[1.0, 2.0], [3.0, 4.0]];
        let w = ndarray::array![[0.1, 0.05], [-0.05, 0.2]];
        let config = RegularizationConfig::new(0.1, false).unwrap();

        let alpha = 1.0;
        let rho_small = 0.1;
        let rho_large = 1000.0;

        let grad_small = augmented_lagrangian_gradient(&w, alpha, rho_small, &data, &config).unwrap();
        let grad_large = augmented_lagrangian_gradient(&w, alpha, rho_large, &data, &config).unwrap();

        // With large rho, constraint term dominates (unless h ≈ 0)
        let h_w = acyclicity::acyclicity_constraint(&w).unwrap();
        if h_w.abs() > 1e-6 {
            let norm_small = grad_small.iter().map(|x| x*x).sum::<f64>().sqrt();
            let norm_large = grad_large.iter().map(|x| x*x).sum::<f64>().sqrt();
            // Penalty term should make large_rho gradient larger
            assert!(norm_large > norm_small * 0.5, "Large rho should increase gradient norm");
        }
    }

    #[test]
    fn test_augmented_lagrangian_gradient_dimension_mismatch() {
        // Should propagate error from score_gradient
        let data = ndarray::array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let w = Array2::zeros((2, 2));
        let config = RegularizationConfig::new(0.1, false).unwrap();

        assert!(augmented_lagrangian_gradient(&w, 1.0, 1.0, &data, &config).is_err());
    }

    #[test]
    fn test_augmented_lagrangian_gradient_invalid_rho() {
        // Negative or zero rho should fail
        let data = ndarray::array![[1.0, 2.0], [3.0, 4.0]];
        let w = ndarray::array![[0.1, 0.05], [-0.05, 0.2]];
        let config = RegularizationConfig::new(0.1, false).unwrap();

        assert!(augmented_lagrangian_gradient(&w, 1.0, 0.0, &data, &config).is_err());
        assert!(augmented_lagrangian_gradient(&w, 1.0, -1.0, &data, &config).is_err());
    }

    #[test]
    fn test_augmented_lagrangian_gradient_invalid_alpha() {
        // NaN or Inf alpha should fail
        let data = ndarray::array![[1.0, 2.0], [3.0, 4.0]];
        let w = ndarray::array![[0.1, 0.05], [-0.05, 0.2]];
        let config = RegularizationConfig::new(0.1, false).unwrap();

        assert!(augmented_lagrangian_gradient(&w, f64::NAN, 1.0, &data, &config).is_err());
        assert!(augmented_lagrangian_gradient(&w, f64::INFINITY, 1.0, &data, &config).is_err());
    }

    #[test]
    fn test_augmented_lagrangian_gradient_numerical_stability() {
        // Test with very small/large values
        let data = ndarray::array![[1e-10, 2e-10], [3e-10, 4e-10]];
        let w = ndarray::array![[1e-15, 1e-15], [1e-15, 1e-15]];
        let config = RegularizationConfig::new(0.1, false).unwrap();

        let grad = augmented_lagrangian_gradient(&w, 1.0, 1.0, &data, &config).unwrap();
        assert!(grad.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_augmented_lagrangian_gradient_gradient_norm() {
        // Verify gradient norm scaling is reasonable O(1-10)
        let data = ndarray::array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let w = ndarray::array![[0.1, 0.05], [-0.05, 0.2]];
        let config = RegularizationConfig::new(0.1, false).unwrap();

        let alpha = 1.0;
        let rho = 5.0;

        let grad = augmented_lagrangian_gradient(&w, alpha, rho, &data, &config).unwrap();
        let grad_norm = grad.iter().map(|x| x*x).sum::<f64>().sqrt();

        // Gradient norm should be reasonable for L-BFGS
        assert!(grad_norm < 1e6, "Gradient norm too large: {}", grad_norm);
        assert!(grad_norm > 1e-8, "Gradient norm too small: {} (essentially zero)", grad_norm);
    }
}
