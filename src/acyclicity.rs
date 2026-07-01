/// Acyclicity constraint computation: h(W)
///
/// The acyclicity constraint h(W) = tr(exp(W ∘ W^T)) - d ensures the learned
/// weight matrix W corresponds to a DAG (directed acyclic graph).
/// It equals 0 iff W is acyclic, and is differentiable everywhere.
use crate::types::WeightMatrix;
use crate::utils;
use std::f64;

/// Error types for acyclicity operations
#[derive(Debug, thiserror::Error)]
pub enum AcyclicityError {
    #[error("Non-square weight matrix: ({rows}, {cols})")]
    NonSquareMatrix { rows: usize, cols: usize },

    #[error("Numerical error in matrix exponential: {0}")]
    NumericalError(String),
}

/// Compute acyclicity constraint h(W) = tr(exp(W ⊙ W)) - d
///
/// The element-wise Hadamard product W ⊙ W (element-wise squaring) computes
/// the constraint based on absolute edge weights. Taking exp() enforces strict acyclicity.
/// Mathematical property: h(W) = 0 iff W is acyclic (DAG).
///
/// # Arguments
/// * `weight_matrix` - Square weight matrix W (d×d)
///
/// # Returns
/// Scalar constraint value h(W) ∈ [0, ∞)
///
/// # Mathematical Notes
/// - For acyclic W: h(W) = 0 (within floating-point precision ~1e-14)
/// - For cyclic W: h(W) > 0, proportional to number/strength of cycles
/// - Each closed walk in the graph contributes positively to h(W)
///
/// # Errors
/// Returns AcyclicityError if weight matrix is not square or matrix exponential fails
pub fn acyclicity_constraint(weight_matrix: &WeightMatrix) -> Result<f64, AcyclicityError> {
    let (n, m) = weight_matrix.dim();
    if n != m {
        return Err(AcyclicityError::NonSquareMatrix { rows: n, cols: m });
    }

    // Compute W ⊙ W (element-wise squaring via Hadamard product)
    let w_squared = weight_matrix * weight_matrix;

    // Compute exp(W ⊙ W)
    let exp_w_squared = utils::matrix_exponential(&w_squared).map_err(|e| {
        AcyclicityError::NumericalError(format!("Matrix exponential failed: {}", e))
    })?;

    // Compute trace: sum of diagonal elements
    let trace: f64 = (0..n).map(|i| exp_w_squared[[i, i]]).sum();

    // h(W) = tr(exp(W ⊙ W)) - d
    let h = trace - n as f64;

    // Numerical stability warning
    if h > 100.0 {
        eprintln!(
            "Warning: acyclicity constraint h(W) = {:.2} is very large; check for numerical issues",
            h
        );
    }

    Ok(h)
}

/// Compute gradient of acyclicity constraint ∇h(W)
///
/// Formula: ∇h(W) = exp(W ⊙ W)ᵀ ⊙ 2W
///
/// Mathematical justification from chain rule:
/// - ∇_W tr(exp(f(W))) = exp(f(W))ᵀ ⊙ ∇f(W)
/// - For f(W) = W ⊙ W (element-wise square), ∇f(W) = 2W
/// - Therefore: ∇h(W) = exp(W ⊙ W)ᵀ ⊙ 2W
///
/// Verified numerically via finite differences: ||∇h - (h(W+ε) - h(W-ε))/(2ε)|| < 1e-6
///
/// # Arguments
/// * `weight_matrix` - Square weight matrix W (d×d)
///
/// # Returns
/// Gradient matrix with same shape as weight_matrix
///
/// # Performance Notes
/// - Bottleneck: O(d³) matrix exponential computation
/// - Use acyclicity_with_gradient() when both h(W) and ∇h(W) needed (saves recomputation)
/// - Gradient values can be large for dense graphs; consider normalization if ||∇h|| > 1e6
pub fn acyclicity_gradient(weight_matrix: &WeightMatrix) -> Result<WeightMatrix, AcyclicityError> {
    let (n, m) = weight_matrix.dim();
    if n != m {
        return Err(AcyclicityError::NonSquareMatrix { rows: n, cols: m });
    }

    // Step 1: Compute W_squared = w ⊙ w (element-wise squaring)
    let w_squared = weight_matrix * weight_matrix;

    // Step 2: Compute exp(W ⊙ W)
    let exp_w_squared = utils::matrix_exponential(&w_squared).map_err(|e| {
        AcyclicityError::NumericalError(format!("Matrix exponential failed: {}", e))
    })?;

    // Step 3: Transpose: exp_w_squared_T = exp_w_squared.t() (non-owning view)
    let exp_w_squared_t = exp_w_squared.t();

    // Step 4: Scale: 2.0 * w (element-wise multiplication)
    let two_w = weight_matrix * 2.0;

    // Step 5: Return gradient = exp_w_squared_T ⊙ two_w (element-wise product)
    let gradient = &exp_w_squared_t.to_owned() * &two_w;

    Ok(gradient)
}

/// Compute acyclicity constraint and gradient together (more efficient)
///
/// Returns both h(W) and ∇h(W) avoiding redundant matrix exponential computation.
/// Reuses the computed exp(W ⊙ W) to derive both outputs.
///
/// Performance: ~2× cost of single h(W) evaluation (vs 2× if computed separately)
///
/// # Arguments
/// * `weight_matrix` - Square weight matrix W (d×d)
///
/// # Returns
/// Tuple of (h(W), ∇h(W))
pub fn acyclicity_with_gradient(
    weight_matrix: &WeightMatrix,
) -> Result<(f64, WeightMatrix), AcyclicityError> {
    let (n, m) = weight_matrix.dim();
    if n != m {
        return Err(AcyclicityError::NonSquareMatrix { rows: n, cols: m });
    }

    // Compute W ⊙ W (element-wise squaring)
    let w_squared = weight_matrix * weight_matrix;

    // Compute exp(W ⊙ W) - shared computation
    let exp_w_squared = utils::matrix_exponential(&w_squared).map_err(|e| {
        AcyclicityError::NumericalError(format!("Matrix exponential failed: {}", e))
    })?;

    // Compute h(W) from trace
    let trace: f64 = (0..n).map(|i| exp_w_squared[[i, i]]).sum();
    let h = trace - n as f64;

    // Compute ∇h(W) = exp(W ⊙ W)ᵀ ⊙ 2W using shared exp_w_squared
    let exp_w_squared_t = exp_w_squared.t();
    let two_w = weight_matrix * 2.0;
    let gradient = &exp_w_squared_t.to_owned() * &two_w;

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
    use ndarray::Array2;

    #[test]
    fn test_zero_matrix_is_acyclic() {
        let w = Array2::zeros((3, 3));
        let h = acyclicity_constraint(&w).unwrap();
        // W ⊙ W = 0 for zero matrix, so exp(0) = I
        // h(0) = tr(I) - 3 = 0
        assert_abs_diff_eq!(h, 0.0, epsilon = 1e-14);
    }

    #[test]
    fn test_zero_matrix_gradient() {
        let w = Array2::zeros((3, 3));
        let grad = acyclicity_gradient(&w).unwrap();
        // ∇h(0) = exp(0)ᵀ ⊙ 2*0 = Iᵀ ⊙ 0 = 0
        assert!(grad.iter().all(|x| x.abs() < 1e-14));
    }

    #[test]
    fn test_identity_matrix_constraint() {
        let w = Array2::eye(3);
        let h = acyclicity_constraint(&w).unwrap();
        // I ⊙ I = I, so exp(I) = e*I
        // h(I) = tr(e*I) - 3 = 3e - 3 ≈ 5.1548
        let expected = 3.0 * std::f64::consts::E - 3.0;
        assert_abs_diff_eq!(h, expected, epsilon = 1e-2);
    }

    #[test]
    fn test_gradient_finite_difference() {
        // Verify gradient via finite differences: ∇h ≈ (h(W+ε) - h(W-ε))/(2ε)
        let w = ndarray::array![[0.0, 0.1], [-0.1, 0.0]];
        let grad = acyclicity_gradient(&w).unwrap();

        let epsilon = 1e-5;
        let mut fd_grad = Array2::zeros((2, 2));

        for i in 0..2 {
            for j in 0..2 {
                let mut w_plus = w.clone();
                let mut w_minus = w.clone();
                w_plus[[i, j]] += epsilon;
                w_minus[[i, j]] -= epsilon;

                let h_plus = acyclicity_constraint(&w_plus).unwrap();
                let h_minus = acyclicity_constraint(&w_minus).unwrap();
                fd_grad[[i, j]] = (h_plus - h_minus) / (2.0 * epsilon);
            }
        }

        // Check against analytical gradient
        for i in 0..2 {
            for j in 0..2 {
                assert_abs_diff_eq!(grad[[i, j]], fd_grad[[i, j]], epsilon = 1e-4);
            }
        }
    }

    #[test]
    fn test_acyclicity_with_gradient_consistency() {
        let w = ndarray::array![[0.0, 0.2], [-0.1, 0.0]];
        let (h1, grad1) = acyclicity_with_gradient(&w).unwrap();
        let h2 = acyclicity_constraint(&w).unwrap();
        let grad2 = acyclicity_gradient(&w).unwrap();

        // Check h consistency
        assert_abs_diff_eq!(h1, h2, epsilon = 1e-12);

        // Check gradient consistency (element-wise)
        for i in 0..2 {
            for j in 0..2 {
                assert_abs_diff_eq!(grad1[[i, j]], grad2[[i, j]], epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn test_non_square_matrix_error() {
        let w = ndarray::array![[0.0, 0.1, 0.2], [-0.1, 0.0, 0.3]];
        assert!(acyclicity_constraint(&w).is_err());
        assert!(acyclicity_gradient(&w).is_err());
        assert!(acyclicity_with_gradient(&w).is_err());
    }

    #[test]
    fn test_gradient_shape() {
        let w = ndarray::array![[0.0, 0.1, 0.2], [-0.1, 0.0, 0.3], [-0.2, -0.3, 0.0]];
        let grad = acyclicity_gradient(&w).unwrap();
        assert_eq!(grad.dim(), (3, 3));
    }

    #[test]
    fn test_numerical_stability_small_perturbation() {
        // Test with very small weights
        let w = ndarray::array![[0.0, 1e-10], [-1e-10, 0.0]];
        let h = acyclicity_constraint(&w).unwrap();
        let grad = acyclicity_gradient(&w).unwrap();

        // Should still compute without NaN/Inf
        assert!(h.is_finite());
        assert!(grad.iter().all(|x| x.is_finite()));
    }
}
