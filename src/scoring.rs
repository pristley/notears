/// Loss function computation: F(W)
///
/// This module provides implementations of various loss functions used in structure learning:
/// - MSE (Mean Squared Error) for continuous data
/// - L1 regularization for sparsity
///
/// The total loss is F(W) = data_fidelity(W) + λ * L1(W)
use crate::types::{DataMatrix, RegularizationConfig, WeightMatrix};

/// Error types for loss computation
#[derive(Debug, thiserror::Error)]
pub enum ScoringError {
    #[error("Dimension mismatch: data has {data_vars} variables but weight matrix is {weight_dim}×{weight_dim}")]
    DimensionMismatch { data_vars: usize, weight_dim: usize },

    #[error("Empty data matrix")]
    EmptyData,

    #[error("Weight matrix must be square")]
    NonSquareWeight,

    #[error("Numerical error: {0}")]
    NumericalError(String),
}

/// Compute least-squares loss: ℓ(W; X) = (1/(2n)) * ||X - X @ W||²_F
///
/// For each sample x_i: residual_i = x_i - W^T @ x_i
/// The LS loss measures how well W explains the data through linear relationships.
/// Factor of 1/(2n) ensures gradient has clean form and coordinates well with acyclicity constraint.
///
/// # Arguments
/// * `data` - Data matrix X (n×d) where n=samples, d=variables
/// * `weight_matrix` - Weight matrix W (d×d)
///
/// # Returns
/// Least-squares loss value ℓ(W; X)
///
/// # Errors
/// Returns ScoringError if dimensions mismatch or data is empty
pub fn mse_loss(data: &DataMatrix, weight_matrix: &WeightMatrix) -> Result<f64, ScoringError> {
    let (n, d) = data.dim();
    let (w_rows, w_cols) = weight_matrix.dim();

    if d != w_rows || w_rows != w_cols {
        return Err(ScoringError::DimensionMismatch {
            data_vars: d,
            weight_dim: w_rows,
        });
    }

    if n == 0 {
        return Err(ScoringError::EmptyData);
    }

    // Compute X @ W (d×d weight matrix applied to n×d data)
    let xw = data.dot(weight_matrix);

    // Compute residuals: X - X @ W (element-wise subtraction)
    let residuals = data - &xw;

    // Frobenius norm squared: sum of all squared elements
    let sum_sq: f64 = residuals.iter().map(|x| x * x).sum();

    // Normalize by 2n (factor of 2 appears in gradient of LS)
    Ok(sum_sq / (2.0 * n as f64))
}

/// Compute L1 penalty: ||W||_1 = sum of absolute values
///
/// Promotes sparsity in the weight matrix. Combined with data fidelity,
/// encourages learning sparse DAG structures.
///
/// # Arguments
/// * `weight_matrix` - Weight matrix W (d×d)
///
/// # Returns
/// L1 penalty value
pub fn l1_penalty(weight_matrix: &WeightMatrix) -> f64 {
    weight_matrix.iter().map(|x| x.abs()).sum()
}

/// Compute L2 penalty: (1/2) * ||W||_2^2 = (1/2) * sum(W_ij^2)
///
/// Provides Tikhonov regularization for smoother solutions.
///
/// # Arguments
/// * `weight_matrix` - Weight matrix W (d×d)
///
/// # Returns
/// L2 penalty value
pub fn l2_penalty(weight_matrix: &WeightMatrix) -> f64 {
    0.5 * weight_matrix.iter().map(|x| x * x).sum::<f64>()
}

/// Compute smooth scoring function: F(W) = ℓ(W; X) + λ||W||₁
///
/// The scoring function combines:
/// 1. **Least-squares loss** ℓ(W; X) = (1/(2n)) * ||X - XW||²_F
///    - Data fidelity term: measures how well W explains the data
///    - Scaling by 1/(2n) ensures gradient coordination with acyclicity constraint
/// 2. **L₁ regularization** λ||W||₁ = λ * sum(|W_ij|)
///    - Sparsity penalty: encourages learning sparse DAG structures
///    - λ (lambda) controls sparsity-fidelity trade-off
///
/// # Arguments
/// * `w` - Weight matrix W (d×d) representing the learned structure
/// * `data` - Data matrix X (n×d) with n samples, d variables
/// * `config` - RegularizationConfig containing lambda ≥ 0.0
///
/// # Returns
/// Smooth score F(W) ∈ [0, ∞), lower is better during optimization
///
/// # Errors
/// Returns ScoringError if:
/// - data shape (n, d) doesn't match W shape (d, d)
/// - data is empty (n = 0)
/// - NaN or Inf detected in residuals
///
/// # Mathematical Notes
/// - **Statistical consistency**: LS loss is statistically consistent in finite-sample and high-dimensional regimes
/// - **High-dimensional regime**: Converges with d >> n when lambda > 0
/// - **Lambda selection**: Typically lambda ∈ [0.0, 0.5] for structure learning
/// - **BIC-style selection**: lambda = log(n)/(2n) often effective
/// - **No assumptions**: Works without Gaussian or faithfulness assumptions
///
/// # Computational Complexity
/// - Matrix multiplication X @ W: O(n·d²)
/// - Residual computation: O(n·d²)
/// - Frobenius norm: O(n·d²)
/// - L₁ regularization: O(d²)
/// - **Total**: O(n·d²) per evaluation (linear in sample size)
pub fn score_function(
    w: &WeightMatrix,
    data: &DataMatrix,
    config: &RegularizationConfig,
) -> Result<f64, ScoringError> {
    let (n, d) = data.dim();
    let (w_rows, w_cols) = w.dim();

    // Input validation
    if d != w_rows || w_rows != w_cols {
        return Err(ScoringError::DimensionMismatch {
            data_vars: d,
            weight_dim: w_rows,
        });
    }

    if n == 0 {
        return Err(ScoringError::EmptyData);
    }

    if w_cols == 0 {
        return Err(ScoringError::NonSquareWeight);
    }

    // **Component 1: Least-squares loss ℓ(W; X) = (1/(2n)) * ||X - XW||²_F**

    // Step 1: Compute residual matrix R = X - X @ W
    let xw = data.dot(w);
    let residuals = data - &xw;

    // Step 2: Compute Frobenius norm squared: ||R||²_F = sum(R_ij²)
    let sum_sq: f64 = residuals.iter().map(|x| x * x).sum();

    // Check for numerical issues
    if !sum_sq.is_finite() {
        return Err(ScoringError::NumericalError(
            "Residual sum-of-squares is NaN or Inf".to_string(),
        ));
    }

    // Step 3: Normalize by 2n
    let loss_ls = sum_sq / (2.0 * n as f64);

    // **Component 2: L₁ regularization λ||W||₁**
    let l1_penalty = w.iter().map(|x| x.abs()).sum::<f64>();
    let regularization = config.lambda * l1_penalty;

    // **Combined score: F(W) = ℓ(W; X) + λ||W||₁**
    let score = loss_ls + regularization;

    Ok(score)
}

/// Compute total loss: F(W) = MSE(W) + λ1 * L1(W)
///
/// # Arguments
/// * `data` - Data matrix X (n×d)
/// * `weight_matrix` - Weight matrix W (d×d)
/// * `reg_config` - Regularization configuration with lambda coefficient
///
/// # Returns
/// Total loss value F(W)
pub fn total_loss(
    data: &DataMatrix,
    weight_matrix: &WeightMatrix,
    reg_config: &RegularizationConfig,
) -> Result<f64, ScoringError> {
    let mse = mse_loss(data, weight_matrix)?;
    let l1 = l1_penalty(weight_matrix);

    Ok(mse + reg_config.lambda * l1)
}

/// Compute gradient of least-squares loss: ∇_W ℓ(W; X) = -(1/n) * X^T @ (X - X@W)
///
/// Derived from differentiating the LS loss ℓ(W; X) = (1/(2n)) * ||X - XW||²_F:
/// - ∂ℓ/∂W = (1/(2n)) * ∂||X - XW||²_F / ∂W
/// - = (1/(2n)) * 2 * (X - XW)^T * (-X)
/// - = -(1/n) * X^T @ (X - XW)
///
/// This gradient is used in the L-BFGS optimization step.
///
/// # Arguments
/// * `data` - Data matrix X (n×d)
/// * `weight_matrix` - Weight matrix W (d×d)
///
/// # Returns
/// Gradient matrix with same shape as weight_matrix
pub fn mse_gradient(
    data: &DataMatrix,
    weight_matrix: &WeightMatrix,
) -> Result<WeightMatrix, ScoringError> {
    let (n, d) = data.dim();
    let (w_rows, w_cols) = weight_matrix.dim();

    if d != w_rows || w_rows != w_cols {
        return Err(ScoringError::DimensionMismatch {
            data_vars: d,
            weight_dim: w_rows,
        });
    }

    if n == 0 {
        return Err(ScoringError::EmptyData);
    }

    // Compute X @ W
    let xw = data.dot(weight_matrix);

    // Residuals: X - X @ W
    let residuals = data - &xw;

    // Gradient: -(1/n) * X^T @ residuals (factor 1/n comes from (1/(2n)) normalization)
    let gradient = data.t().dot(&residuals) * (-1.0 / n as f64);

    Ok(gradient)
}

/// Compute gradient of L1 penalty: ∂||W||_1 / ∂W_ij = sign(W_ij)
///
/// # Arguments
/// * `weight_matrix` - Weight matrix W (d×d)
///
/// # Returns
/// Subgradient matrix (0 where W_ij = 0, ±1 where W_ij ≠ 0)
pub fn l1_gradient(weight_matrix: &WeightMatrix) -> WeightMatrix {
    weight_matrix.mapv(|x| {
        if x > 0.0 {
            1.0
        } else if x < 0.0 {
            -1.0
        } else {
            0.0
        }
    })
}

/// Compute total loss gradient: ∇F(W) = ∇MSE + λ * ∇L1
///
/// # Arguments
/// * `data` - Data matrix X (n×d)
/// * `weight_matrix` - Weight matrix W (d×d)
/// * `reg_config` - Regularization configuration
///
/// # Returns
/// Total gradient matrix
pub fn total_loss_gradient(
    data: &DataMatrix,
    weight_matrix: &WeightMatrix,
    reg_config: &RegularizationConfig,
) -> Result<WeightMatrix, ScoringError> {
    let grad_mse = mse_gradient(data, weight_matrix)?;
    let grad_l1 = l1_gradient(weight_matrix);

    Ok(grad_mse + reg_config.lambda * grad_l1)
}

/// Compute analytical gradient of smooth scoring function: ∇F(W) = ∇ℓ(W; X) + λ·∇||W||₁
///
/// The gradient combines:
/// 1. **Gradient of least-squares loss**: ∇ℓ(W; X) = -(1/n) X^T @ (X - XW)
///    - Residual form: R = X - XW
///    - Gradient: -(1/n) X^T @ R
/// 2. **Subgradient of L₁ regularization**: λ·sign(W)
///    - Element-wise sign: sign(w) = {+1 if w>0, -1 if w<0, 0 if w=0}
///    - Scaled by regularization strength λ
///
/// Mathematical derivation:
/// - ∂ℓ/∂W = ∂/∂W [(1/(2n)) ||X - XW||²_F]
///   = (1/(2n)) * 2 * (X - XW)^T * (-X)
///   = -(1/n) * X^T @ (X - XW)
/// - ∂||W||₁/∂W = sign(W) (element-wise subdifferential)
///
/// Optimization integration:
/// - Suitable for proximal L-BFGS (handles non-smooth L₁ term)
/// - Can alternatively smooth L₁: ||W||₁ ≈ √(W² + ε)
/// - Recommended: Proximal operator for true L₁ term
///
/// # Arguments
/// * `w` - Weight matrix W (d×d)
/// * `data` - Data matrix X (n×d) with n samples
/// * `config` - RegularizationConfig with lambda ≥ 0.0
///
/// # Returns
/// Gradient matrix ∇F(W) with same shape as W
///
/// # Errors
/// Returns ScoringError if dimensions mismatch or data is empty
///
/// # Performance Notes
/// - Dominated by matrix multiply X^T @ R: O(n·d²)
/// - Comparable to loss evaluation cost
/// - Consider caching X^T if gradient called repeatedly
///
/// # Mathematical Properties
/// - Gradient descent direction: W_new = W_old - α·∇F(W)
/// - Negative gradient decreases loss (with small enough step size α)
/// - Sign function creates non-smoothness at w=0 (handled by proximal methods)
pub fn score_gradient(
    w: &WeightMatrix,
    data: &DataMatrix,
    config: &RegularizationConfig,
) -> Result<WeightMatrix, ScoringError> {
    let (n, d) = data.dim();
    let (w_rows, w_cols) = w.dim();

    // Input validation
    if d != w_rows || w_rows != w_cols {
        return Err(ScoringError::DimensionMismatch {
            data_vars: d,
            weight_dim: w_rows,
        });
    }

    if n == 0 {
        return Err(ScoringError::EmptyData);
    }

    if w_cols == 0 {
        return Err(ScoringError::NonSquareWeight);
    }

    // **Component 1: Gradient of LS loss ∇ℓ(W; X) = -(1/n) X^T @ (X - XW)**

    // Step 1: Compute residual R = X - X @ W
    let xw = data.dot(w);
    let residuals = data - &xw;

    // Step 2: Compute X^T (non-owning transpose view)
    let data_t = data.t();

    // Step 3: Compute gradient: -(1/n) * X^T @ R
    let grad_ls = data_t.dot(&residuals) * (-1.0 / n as f64);

    // **Component 2: Subgradient of L₁ regularization λ·sign(W)**
    let grad_l1 = l1_gradient(w);
    let scaled_grad_l1 = &grad_l1 * config.lambda;

    // **Combined gradient: ∇F(W) = ∇ℓ + λ·∇||W||₁**
    let gradient = grad_ls + &scaled_grad_l1;

    Ok(gradient)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::Array2;

    #[test]
    fn test_mse_zero_weights() {
        let data = ndarray::array![[1.0, 2.0], [3.0, 4.0]];
        let weights = Array2::zeros((2, 2));
        let mse = mse_loss(&data, &weights).unwrap();

        // With W=0, residuals = data
        // ||residuals||²_F = 1 + 4 + 9 + 16 = 30
        // MSE = ||residuals||²_F / (2*n) = 30 / (2*2) = 7.5
        let expected = (1.0 + 4.0 + 9.0 + 16.0) / (2.0 * 2.0);
        assert_abs_diff_eq!(mse, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_l1_penalty() {
        let weights = ndarray::array![[1.0, -2.0], [0.5, 0.3]];
        let penalty = l1_penalty(&weights);
        assert_abs_diff_eq!(penalty, 3.8, epsilon = 1e-10);
    }

    #[test]
    fn test_dimension_mismatch() {
        let data = ndarray::array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let weights = Array2::zeros((2, 2));
        assert!(mse_loss(&data, &weights).is_err());
    }

    #[test]
    fn test_l1_gradient() {
        let weights = ndarray::array![[1.0, -2.0, 0.0], [0.5, 0.0, -0.3]];
        let grad = l1_gradient(&weights);
        assert_eq!(grad[[0, 0]], 1.0);
        assert_eq!(grad[[0, 1]], -1.0);
        assert_eq!(grad[[0, 2]], 0.0);
    }

    #[test]
    fn test_score_function_zero_weights() {
        let data = ndarray::array![[1.0, 2.0], [3.0, 4.0]];
        let weights = Array2::zeros((2, 2));
        let config = RegularizationConfig::new(0.1, false).unwrap();
        let score = score_function(&weights, &data, &config).unwrap();

        // With W=0 and lambda=0.1:
        // F(W) = (1/(2n)) * ||X - X@W||²_F + 0.1 * ||W||_1
        // = (1/4) * 30 + 0.1 * 0 = 7.5
        assert_abs_diff_eq!(score, 7.5, epsilon = 1e-10);
    }

    #[test]
    fn test_score_function_with_regularization() {
        let data = ndarray::array![[1.0, 2.0], [3.0, 4.0]];
        let weights = ndarray::array![[0.5, 0.0], [0.0, 0.5]];
        let config = RegularizationConfig::new(0.1, false).unwrap();
        let score = score_function(&weights, &data, &config).unwrap();

        // F(W) = loss_ls + 0.1 * ||W||_1
        // Loss should be lower with non-zero W (fitting the data better)
        // L1 penalty = 0.1 * (0.5 + 0.5) = 0.1
        assert!(score.is_finite());
        assert!(score > 0.0); // Loss should be positive
    }

    #[test]
    fn test_score_function_dimension_mismatch() {
        let data = ndarray::array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let weights = Array2::zeros((2, 2));
        let config = RegularizationConfig::new(0.1, false).unwrap();
        assert!(score_function(&weights, &data, &config).is_err());
    }

    #[test]
    fn test_score_function_components() {
        // Verify that score_function = mse_loss + lambda * l1_penalty
        let data = ndarray::array![[1.0, 2.0], [3.0, 4.0]];
        let weights = ndarray::array![[0.1, 0.2], [-0.1, 0.3]];
        let lambda = 0.2;
        let config = RegularizationConfig::new(lambda, false).unwrap();

        let score = score_function(&weights, &data, &config).unwrap();
        let ls_loss = mse_loss(&data, &weights).unwrap();
        let l1_pen = l1_penalty(&weights);
        let expected = ls_loss + lambda * l1_pen;

        assert_abs_diff_eq!(score, expected, epsilon = 1e-12);
    }

    #[test]
    fn test_score_function_sparsity() {
        // Verify score component decomposition with different regularization strengths
        let data = ndarray::array![[1.0, 2.0], [3.0, 4.0]];
        let weights = ndarray::array![[0.1, 0.2], [-0.1, 0.3]];

        // With lambda=0 (no regularization), score equals LS loss only
        let config_no_reg = RegularizationConfig::new(0.0, false).unwrap();
        let score_no_reg = score_function(&weights, &data, &config_no_reg).unwrap();
        let loss_only = mse_loss(&data, &weights).unwrap();
        assert_abs_diff_eq!(score_no_reg, loss_only, epsilon = 1e-12);

        // With lambda > 0, score = loss + regularization term
        let config_reg = RegularizationConfig::new(0.5, false).unwrap();
        let score_with_reg = score_function(&weights, &data, &config_reg).unwrap();
        assert!(score_with_reg > score_no_reg); // Regularization adds to score
    }

    // ========== Score Gradient Tests ==========

    #[test]
    fn test_score_gradient_zero_weights() {
        let data = ndarray::array![[1.0, 2.0], [3.0, 4.0]];
        let weights = Array2::zeros((2, 2));
        let config = RegularizationConfig::new(0.1, false).unwrap();
        let grad = score_gradient(&weights, &data, &config).unwrap();

        // With W=0, residuals = X, so ∇ℓ = -(1/n) * X^T @ X
        // ∇||W||₁ = 0 (all zero weights have zero subgradient)
        let expected_grad_ls = data.t().dot(&data) * (-1.0 / 2.0);

        for i in 0..2 {
            for j in 0..2 {
                assert_abs_diff_eq!(grad[[i, j]], expected_grad_ls[[i, j]], epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn test_score_gradient_shape() {
        let data = ndarray::array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let weights = ndarray::array![[0.1, 0.0, 0.2], [0.0, 0.3, 0.0], [0.1, 0.0, 0.2]];
        let config = RegularizationConfig::new(0.2, false).unwrap();
        let grad = score_gradient(&weights, &data, &config).unwrap();

        // Shape must match weight matrix
        assert_eq!(grad.dim(), (3, 3));
    }

    #[test]
    fn test_score_gradient_finite_difference() {
        // Verify gradient via finite differences: ∇F ≈ (F(W+ε) - F(W-ε))/(2ε)
        let data = ndarray::array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let w = ndarray::array![[0.1, 0.05], [-0.05, 0.2]];
        let config = RegularizationConfig::new(0.15, false).unwrap();

        let grad_analytical = score_gradient(&w, &data, &config).unwrap();

        let epsilon = 1e-5;
        let mut grad_numerical = Array2::zeros((2, 2));

        for i in 0..2 {
            for j in 0..2 {
                let mut w_plus = w.clone();
                let mut w_minus = w.clone();
                w_plus[[i, j]] += epsilon;
                w_minus[[i, j]] -= epsilon;

                let f_plus = score_function(&w_plus, &data, &config).unwrap();
                let f_minus = score_function(&w_minus, &data, &config).unwrap();
                grad_numerical[[i, j]] = (f_plus - f_minus) / (2.0 * epsilon);
            }
        }

        // Check element-wise accuracy
        for i in 0..2 {
            for j in 0..2 {
                assert_abs_diff_eq!(
                    grad_analytical[[i, j]],
                    grad_numerical[[i, j]],
                    epsilon = 1e-4
                );
            }
        }
    }

    #[test]
    fn test_score_gradient_components() {
        // Verify gradient decomposition: ∇F = ∇ℓ + λ·∇||W||₁
        let data = ndarray::array![[1.0, 2.0], [3.0, 4.0]];
        let weights = ndarray::array![[0.2, -0.1], [0.3, 0.0]];
        let lambda = 0.25;
        let config = RegularizationConfig::new(lambda, false).unwrap();

        let grad_full = score_gradient(&weights, &data, &config).unwrap();
        let grad_ls = mse_gradient(&data, &weights).unwrap();
        let grad_l1 = l1_gradient(&weights);
        let expected = grad_ls + &grad_l1 * lambda;

        for i in 0..2 {
            for j in 0..2 {
                assert_abs_diff_eq!(grad_full[[i, j]], expected[[i, j]], epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn test_score_gradient_sign_function() {
        // Verify L₁ subgradient (sign function) correctness
        let data = ndarray::array![[1.0, 2.0], [3.0, 4.0]];
        let weights = ndarray::array![[0.5, -0.3], [0.0, 0.1]];
        let config = RegularizationConfig::new(1.0, false).unwrap();

        let grad = score_gradient(&weights, &data, &config).unwrap();

        // Extract L₁ component (at high lambda, should dominate for small LS gradients)
        // At least verify sign patterns exist
        assert!(grad.iter().any(|x| x.abs() > 0.0));
        assert!(grad.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_score_gradient_descent_direction() {
        // Verify gradient points toward descent direction
        let data = ndarray::array![[1.0, 2.0], [3.0, 4.0]];
        let weights = ndarray::array![[0.1, 0.05], [0.05, 0.1]];
        let config = RegularizationConfig::new(0.1, false).unwrap();

        let f_current = score_function(&weights, &data, &config).unwrap();
        let grad = score_gradient(&weights, &data, &config).unwrap();

        // Take small step in negative gradient direction
        let step_size = 0.01;
        let weights_new = &weights - &grad * step_size;
        let f_new = score_function(&weights_new, &data, &config).unwrap();

        // Loss should decrease (f_new < f_current) with small enough step
        assert!(f_new < f_current, "Gradient descent should reduce loss");
    }

    #[test]
    fn test_score_gradient_dimension_mismatch() {
        let data = ndarray::array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let weights = Array2::zeros((2, 2));
        let config = RegularizationConfig::new(0.1, false).unwrap();
        assert!(score_gradient(&weights, &data, &config).is_err());
    }

    #[test]
    fn test_score_gradient_empty_data() {
        let data: DataMatrix = Array2::zeros((0, 2));
        let weights = Array2::zeros((2, 2));
        let config = RegularizationConfig::new(0.1, false).unwrap();
        assert!(score_gradient(&weights, &data, &config).is_err());
    }

    #[test]
    fn test_score_gradient_numerical_stability() {
        // Test with very small weights
        let data = ndarray::array![[1e-10, 2e-10], [3e-10, 4e-10]];
        let weights = ndarray::array![[1e-15, -1e-15], [1e-15, 1e-15]];
        let config = RegularizationConfig::new(0.1, false).unwrap();

        let grad = score_gradient(&weights, &data, &config).unwrap();

        // Should still compute without NaN/Inf
        assert!(grad.iter().all(|x| x.is_finite()));
    }
}
