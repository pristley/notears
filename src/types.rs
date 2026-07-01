/// Core type definitions and aliases for the NOTEARS algorithm
///
/// This module defines the fundamental data structures and type aliases
/// used throughout the NOTEARS library for DAG structure learning.
/// All configuration structures include strict validation to ensure
/// numerical stability and correctness of the optimization.
use ndarray::Array2;
use serde::{Deserialize, Serialize};

/// Type alias for floating-point matrix (d × d)
pub type WeightMatrix = Array2<f64>;

/// Type alias for data matrix (n × d)
pub type DataMatrix = Array2<f64>;

/// Type alias for gradient matrix
pub type GradientMatrix = Array2<f64>;

/// Configuration error type
#[derive(Debug, Clone, thiserror::Error)]
pub enum ConfigError {
    #[error("max_outer_iterations must be positive, got {0}")]
    InvalidMaxOuterIterations(usize),

    #[error("max_lbfgs_iterations must be positive, got {0}")]
    InvalidMaxLbfgsIterations(usize),

    #[error("lbfgs_memory must be positive, got {0}")]
    InvalidLbfgsMemory(usize),

    #[error("constraint_tolerance must be in [1e-10, 1e-4], got {0}")]
    InvalidConstraintTolerance(f64),

    #[error("penalty_rho_init must be positive, got {0}")]
    InvalidPenaltyRho(f64),

    #[error("progress_rate must be in (0, 1), got {0}")]
    InvalidProgressRate(f64),

    #[error("edge_threshold must be positive, got {0}")]
    InvalidEdgeThreshold(f64),

    #[error("lambda must be non-negative, got {0}")]
    InvalidLambda(f64),
}

/// Configuration structure for optimization algorithm
///
/// Manages augmented Lagrangian and L-BFGS parameters with strict validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConfig {
    /// Maximum iterations of augmented Lagrangian outer loop
    pub max_outer_iterations: usize,

    /// Maximum iterations for inner L-BFGS solve
    pub max_lbfgs_iterations: usize,

    /// L-BFGS memory size (typically 10-20)
    pub lbfgs_memory: usize,

    /// Convergence tolerance for h(W) acyclicity constraint
    pub constraint_tolerance: f64,

    /// Initial penalty parameter ρ
    pub penalty_rho_init: f64,

    /// Progress rate c for adaptive ρ (typically 0.25)
    pub progress_rate: f64,

    /// Hard threshold for edge detection
    pub edge_threshold: f64,
}

impl OptimizationConfig {
    /// Create new configuration with validation
    ///
    /// # Errors
    /// Returns `ConfigError` if any parameter fails validation
    pub fn new(
        max_outer_iterations: usize,
        max_lbfgs_iterations: usize,
        lbfgs_memory: usize,
        constraint_tolerance: f64,
        penalty_rho_init: f64,
        progress_rate: f64,
        edge_threshold: f64,
    ) -> Result<Self, ConfigError> {
        // Validate outer iterations
        if max_outer_iterations == 0 {
            return Err(ConfigError::InvalidMaxOuterIterations(max_outer_iterations));
        }

        // Validate L-BFGS iterations
        if max_lbfgs_iterations == 0 {
            return Err(ConfigError::InvalidMaxLbfgsIterations(max_lbfgs_iterations));
        }

        // Validate L-BFGS memory
        if lbfgs_memory == 0 {
            return Err(ConfigError::InvalidLbfgsMemory(lbfgs_memory));
        }

        // Validate constraint tolerance: must be in [1e-10, 1e-4]
        if !(1e-10..=1e-4).contains(&constraint_tolerance) {
            return Err(ConfigError::InvalidConstraintTolerance(
                constraint_tolerance,
            ));
        }

        // Validate penalty rho
        if penalty_rho_init <= 0.0 || !penalty_rho_init.is_finite() {
            return Err(ConfigError::InvalidPenaltyRho(penalty_rho_init));
        }

        // Validate progress rate: must be in (0, 1)
        if progress_rate <= 0.0 || progress_rate >= 1.0 || !progress_rate.is_finite() {
            return Err(ConfigError::InvalidProgressRate(progress_rate));
        }

        // Validate edge threshold
        if edge_threshold <= 0.0 || !edge_threshold.is_finite() {
            return Err(ConfigError::InvalidEdgeThreshold(edge_threshold));
        }

        Ok(OptimizationConfig {
            max_outer_iterations,
            max_lbfgs_iterations,
            lbfgs_memory,
            constraint_tolerance,
            penalty_rho_init,
            progress_rate,
            edge_threshold,
        })
    }
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        OptimizationConfig {
            max_outer_iterations: 1000,
            max_lbfgs_iterations: 50,
            lbfgs_memory: 10,
            constraint_tolerance: 1e-8,
            penalty_rho_init: 1.0,
            progress_rate: 0.25,
            edge_threshold: 0.3,
        }
    }
}

/// Configuration for regularization
///
/// Controls sparsity and smoothness of learned structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegularizationConfig {
    /// L₁ regularization coefficient λ
    pub lambda: f64,

    /// Whether to apply adaptive scaling
    pub adaptive: bool,
}

impl RegularizationConfig {
    /// Create new regularization configuration with validation
    ///
    /// # Errors
    /// Returns `ConfigError` if lambda is negative
    pub fn new(lambda: f64, adaptive: bool) -> Result<Self, ConfigError> {
        if lambda < 0.0 || !lambda.is_finite() {
            return Err(ConfigError::InvalidLambda(lambda));
        }

        Ok(RegularizationConfig { lambda, adaptive })
    }
}

impl Default for RegularizationConfig {
    fn default() -> Self {
        RegularizationConfig {
            lambda: 0.1,
            adaptive: false,
        }
    }
}

/// Optimization result structure
///
/// Contains complete information about the optimization outcome.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationResult {
    /// Optimized weight matrix W (d×d)
    pub weight_matrix: WeightMatrix,

    /// Final acyclicity constraint value h(W)
    pub constraint_violation: f64,

    /// Number of augmented Lagrangian iterations executed
    pub iterations: usize,

    /// Final score F(W)
    pub final_score: f64,

    /// Estimated adjacency matrix (binary, i32)
    pub adjacency_matrix: Array2<i32>,
}

impl OptimizationResult {
    /// Create a new optimization result
    ///
    /// # Arguments
    /// * `weight_matrix` - Learned weight matrix
    /// * `constraint_violation` - Final h(W) value
    /// * `iterations` - Number of iterations
    /// * `final_score` - Final loss value F(W)
    /// * `edge_threshold` - Threshold for adjacency detection
    pub fn new(
        weight_matrix: WeightMatrix,
        constraint_violation: f64,
        iterations: usize,
        final_score: f64,
        edge_threshold: f64,
    ) -> Self {
        let (_d, _) = weight_matrix.dim();
        let adjacency = weight_matrix.mapv(|x| if x.abs() > edge_threshold { 1 } else { 0 });

        OptimizationResult {
            weight_matrix,
            constraint_violation,
            iterations,
            final_score,
            adjacency_matrix: adjacency,
        }
    }

    /// Extract edges as list of (source, target) tuples
    pub fn edges(&self) -> Vec<(usize, usize)> {
        let (n, _) = self.adjacency_matrix.dim();
        let mut edges = Vec::new();

        for i in 0..n {
            for j in 0..n {
                if self.adjacency_matrix[[i, j]] > 0 {
                    edges.push((i, j));
                }
            }
        }
        edges
    }

    /// Get number of edges in learned structure
    pub fn edge_count(&self) -> usize {
        self.adjacency_matrix.iter().filter(|&&x| x > 0).count()
    }

    /// Check if structure satisfies acyclicity constraint
    pub fn is_acyclic(&self, tolerance: f64) -> bool {
        self.constraint_violation <= tolerance
    }
}

/// Comprehensive validation result for DAG verification
///
/// Contains detailed diagnostics to verify DAG properties and debug issues.
/// Includes both constraint-based (h(W)) and graph-based (topological sort) acyclicity checks.
///
/// # Fields
/// * `is_acyclic_by_constraint` - h(W) < tolerance (optimization succeeded)
/// * `is_acyclic_by_topological_sort` - No cycles in binary adjacency (graph is DAG)
/// * `constraint_value` - Actual h(W) value (0 = perfectly acyclic)
/// * `max_cycle_weight` - Sum of weights on longest cycle (0 if no cycles)
/// * `num_edges` - Number of non-zero entries in adjacency matrix
/// * `sparsity` - Fraction of zero entries: (d² - edges) / d²
///
/// # Interpretation
/// - Both flags true: Graph is definitely a valid DAG ✓
/// - Constraint true, topo false: Numerical artifact; likely valid
/// - Constraint false, topo true: Solver didn't converge but structure is acyclic
/// - Both false: Serious issue; graph contains cycles
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    /// Acyclicity check via h(W) < tolerance
    pub is_acyclic_by_constraint: bool,
    /// Acyclicity check via topological sort (Kahn's algorithm)
    pub is_acyclic_by_topological_sort: bool,
    /// Constraint value h(W): should be ~0 for acyclic
    pub constraint_value: f64,
    /// Sum of weights on longest cycle: 0 if DAG
    pub max_cycle_weight: f64,
    /// Number of edges (non-zero adjacency entries)
    pub num_edges: usize,
    /// Sparsity ratio: fraction of zeros
    pub sparsity: f64,
}

impl ValidationResult {
    /// Check if validation passed all criteria
    ///
    /// Returns true only if both constraint and topological sort confirm acyclicity
    pub fn is_valid_dag(&self) -> bool {
        self.is_acyclic_by_constraint && self.is_acyclic_by_topological_sort
    }

    /// Summary status: PASS if valid DAG, diagnostic message if not
    pub fn status_summary(&self) -> String {
        if self.is_valid_dag() {
            "✓ PASS: Valid DAG structure".to_string()
        } else if !self.is_acyclic_by_constraint && !self.is_acyclic_by_topological_sort {
            format!(
                "✗ FAIL: Cycles detected; h(W) = {:.2e}",
                self.constraint_value
            )
        } else if !self.is_acyclic_by_constraint {
            format!(
                "⚠ WARN: h(W) > tolerance ({:.2e}); but no cycles detected",
                self.constraint_value
            )
        } else {
            format!(
                "⚠ WARN: Topological sort detected cycles; h(W) = {:.2e}",
                self.constraint_value
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimization_config_valid() {
        let config = OptimizationConfig::new(1000, 50, 10, 1e-8, 1.0, 0.25, 0.3).unwrap();
        assert_eq!(config.max_outer_iterations, 1000);
    }

    #[test]
    fn test_optimization_config_invalid_outer_iterations() {
        let result = OptimizationConfig::new(0, 50, 10, 1e-8, 1.0, 0.25, 0.3);
        assert!(result.is_err());
    }

    #[test]
    fn test_optimization_config_invalid_constraint_tolerance() {
        let result = OptimizationConfig::new(1000, 50, 10, 1e-12, 1.0, 0.25, 0.3);
        assert!(result.is_err());
    }

    #[test]
    fn test_optimization_config_invalid_progress_rate() {
        let result = OptimizationConfig::new(1000, 50, 10, 1e-8, 1.0, 0.0, 0.3);
        assert!(result.is_err());
    }

    #[test]
    fn test_regularization_config_valid() {
        let config = RegularizationConfig::new(0.1, false).unwrap();
        assert_eq!(config.lambda, 0.1);
    }

    #[test]
    fn test_regularization_config_invalid_lambda() {
        let result = RegularizationConfig::new(-0.1, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_optimization_result_edges() {
        let weight_matrix = ndarray::array![[0.0, 0.5, 0.0], [-0.3, 0.0, 0.2], [0.0, 0.1, 0.0]];
        let result = OptimizationResult::new(weight_matrix, 0.0, 100, 0.1, 0.2);

        let edges = result.edges();
        assert_eq!(edges.len(), 2);
        assert!(edges.contains(&(0, 1)));
        assert!(edges.contains(&(1, 0)));
    }

    #[test]
    fn test_optimization_result_edge_count() {
        let weight_matrix = ndarray::array![[0.0, 0.5], [-0.3, 0.0]];
        let result = OptimizationResult::new(weight_matrix, 0.0, 100, 0.1, 0.2);
        assert_eq!(result.edge_count(), 2);
    }

    #[test]
    fn test_optimization_result_is_acyclic() {
        let weight_matrix = ndarray::array![[0.0, 0.5], [-0.3, 0.0]];
        let result = OptimizationResult::new(weight_matrix, 1e-9, 100, 0.1, 0.2);
        assert!(result.is_acyclic(1e-8));
    }
}
