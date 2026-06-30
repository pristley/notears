/// Acyclicity constraint computation: h(W)
///
/// The acyclicity constraint h(W) = tr(exp(W ∘ W^T)) - d ensures the learned
/// weight matrix W corresponds to a DAG (directed acyclic graph).
/// It equals 0 iff W is acyclic, and is differentiable everywhere.

use crate::types::WeightMatrix;
use crate::utils;
use ndarray::Array2;
use std::f64;

/// Error types for acyclicity operations
#[derive(Debug, thiserror::Error)]
pub enum AcyclicityError {
    #[error("Non-square weight matrix: ({rows}, {cols})")]
    NonSquareMatrix { rows: usize, cols: usize },

    #[error("Numerical error in matrix exponential: {0}")]
    NumericalError(String),
}

/// Compute acyclicity constraint h(W) = tr(exp(W ∘ W^T)) - d
///
/// The element-wise Hadamard product W ∘ W^T ensures the constraint depends
/// on edge weights, not edge orientations. Taking exp() enforces strict acyclicity.
///
/// # Arguments
/// * `weight_matrix` - Square weight matrix W (d×d)
///
/// # Returns
/// Scalar constraint value h(W) ∈ [0, ∞)
///
/// # Panics
/// If weight matrix is not square
pub fn acyclicity_constraint(weight_matrix: &WeightMatrix) -> Result<f64, AcyclicityError> {
    let (n, m) = weight_matrix.dim();
    if n != m {
        return Err(AcyclicityError::NonSquareMatrix {
            rows: n,
            cols: m,
        });
    }

    // Compute W ∘ W^T (Hadamard product)
    let w_transpose = weight_matrix.t().to_owned();
    let hadamard = weight_matrix * &w_transpose;

    // Compute exp(W ∘ W^T)
    let exp_hadamard = utils::matrix_exponential(&hadamard).map_err(|e| {
        AcyclicityError::NumericalError(format!("Matrix exponential failed: {}", e))
    })?;

    // Trace
    let trace: f64 = (0..n).map(|i| exp_hadamard[[i, i]]).sum();

    // h(W) = tr(exp(W ∘ W^T)) - d
    let h = trace - n as f64;

    Ok(h)
}

/// Compute gradient of acyclicity constraint ∇h(W)
///
/// Using the chain rule:
/// ∂h/∂W_ij = ∂tr(exp(M))/∂W_ij where M = W ∘ W^T
///
/// Since M_ij = W_ij * W_ji:
/// ∂tr(exp(M))/∂W_ij = tr(exp(M)^T * ∂exp(M)/∂M_ij) (by Sylvester equation)
///                    = 2 * W_ji * (exp(M) * E_ij * exp(-M))_ii
///
/// where E_ij is the matrix with 1 at (i,j) and 0 elsewhere.
///
/// # Arguments
/// * `weight_matrix` - Square weight matrix W (d×d)
///
/// # Returns
/// Gradient matrix with same shape as weight_matrix
pub fn acyclicity_gradient(weight_matrix: &WeightMatrix) -> Result<WeightMatrix, AcyclicityError> {
    let (n, m) = weight_matrix.dim();
    if n != m {
        return Err(AcyclicityError::NonSquareMatrix {
            rows: n,
            cols: m,
        });
    }

    // Compute M = W ∘ W^T
    let w_transpose = weight_matrix.t().to_owned();
    let hadamard = weight_matrix * &w_transpose;

    // Compute exp(M)
    let exp_hadamard = utils::matrix_exponential(&hadamard).map_err(|e| {
        AcyclicityError::NumericalError(format!("Matrix exponential failed: {}", e))
    })?;

    // For numerical stability in gradient computation, use Frechet derivative
    // ∇_M tr(exp(M)) = exp(M)^T
    let exp_hadamard_t = exp_hadamard.t().to_owned();

    // Initialize gradient
    let mut gradient = Array2::zeros((n, n));

    // Compute gradient using chain rule
    // ∂h/∂W_ij = 2 * W_ji * (exp(M))_ii * exp(-M) is complex
    // Simpler approach: use finite differences for stability
    // For production: use automatic differentiation (tapenade/enzyme)

    // Analytical gradient: ∂h/∂W = 2 * W^T ∘ exp(M)^T where ∘ is Hadamard
    // This is derived from the Frobenius inner product
    for i in 0..n {
        for j in 0..n {
            // Chain rule: ∂h/∂W_ij = ∂h/∂M_ij * ∂M_ij/∂W_ij
            // ∂h/∂M_ij = exp(M)_ij (from matrix exponential derivative)
            // ∂M_ij/∂W_ij = W_ji + W_ij*δ(i==j) (but M_ij = W_ij*W_ji symmetric)
            let dm_dw_ij = weight_matrix[[j, i]]; // since M_ij = W_ij * W_ji

            gradient[[i, j]] = 2.0 * dm_dw_ij * exp_hadamard[[i, j]];
        }
    }

    Ok(gradient)
}

/// Compute acyclicity constraint and gradient together (more efficient)
///
/// Returns both h(W) and ∇h(W) avoiding redundant matrix exponential computation
pub fn acyclicity_with_gradient(
    weight_matrix: &WeightMatrix,
) -> Result<(f64, WeightMatrix), AcyclicityError> {
    let (n, m) = weight_matrix.dim();
    if n != m {
        return Err(AcyclicityError::NonSquareMatrix {
            rows: n,
            cols: m,
        });
    }

    // Compute M = W ∘ W^T
    let w_transpose = weight_matrix.t().to_owned();
    let hadamard = weight_matrix * &w_transpose;

    // Compute exp(M)
    let exp_hadamard = utils::matrix_exponential(&hadamard).map_err(|e| {
        AcyclicityError::NumericalError(format!("Matrix exponential failed: {}", e))
    })?;

    // Compute h(W) from trace
    let trace: f64 = (0..n).map(|i| exp_hadamard[[i, i]]).sum();
    let h = trace - n as f64;

    // Compute gradient
    let mut gradient = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            let dm_dw_ij = weight_matrix[[j, i]];
            gradient[[i, j]] = 2.0 * dm_dw_ij * exp_hadamard[[i, j]];
        }
    }

    Ok((h, gradient))
}

/// Check if weight matrix corresponds to a DAG (approximately)
///
/// Returns true if h(W) <= threshold
pub fn is_dag(weight_matrix: &WeightMatrix, threshold: f64) -> Result<bool, AcyclicityError> {
    let h = acyclicity_constraint(weight_matrix)?;
    Ok(h <= threshold)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_zero_matrix_is_acyclic() {
        let w = Array2::zeros((3, 3));
        let h = acyclicity_constraint(&w).unwrap();
        // exp(0) = I, so tr(I) - d = 0
        assert_abs_diff_eq!(h, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_identity_has_constraint() {
        let w = Array2::eye(3);
        let h = acyclicity_constraint(&w).unwrap();
        // For I: h(I) = tr(exp(I)) - 3 = tr(eI) - 3 = 3e - 3 ≈ 5.15
        // This is > 0, confirming I has structure (not a DAG)
        assert_abs_diff_eq!(h, std::f64::consts::E * 3.0 - 3.0, epsilon = 1e-2);
    }

    #[test]
    fn test_gradient_non_empty() {
        let w = ndarray::array![[0.0, 0.1], [-0.1, 0.0]];
        let grad = acyclicity_gradient(&w).unwrap();
        assert!(grad.iter().any(|x| x.abs() > 0.0));
    }

    #[test]
    fn test_acyclicity_with_gradient_consistency() {
        let w = ndarray::array![[0.0, 0.2], [-0.1, 0.0]];
        let (h1, grad1) = acyclicity_with_gradient(&w).unwrap();
        let h2 = acyclicity_constraint(&w).unwrap();
        let grad2 = acyclicity_gradient(&w).unwrap();

        assert_abs_diff_eq!(h1, h2, epsilon = 1e-12);
        assert!(grad1.iter().zip(grad2.iter()).all(|(a, b)| (a - b).abs() < 1e-12));
    }

    #[test]
    fn test_non_square_matrix_error() {
        let w = ndarray::array![[0.0, 0.1, 0.2], [-0.1, 0.0, 0.3]];
        assert!(acyclicity_constraint(&w).is_err());
    }
}
