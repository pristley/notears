/// Loss function computation: F(W)
///
/// This module provides implementations of various loss functions used in structure learning:
/// - MSE (Mean Squared Error) for continuous data
/// - L1 regularization for sparsity
///
/// The total loss is F(W) = data_fidelity(W) + λ * L1(W)

use crate::types::{WeightMatrix, DataMatrix, RegularizationConfig};
use rayon::prelude::*;

/// Error types for loss computation
#[derive(Debug, thiserror::Error)]
pub enum ScoringError {
    #[error("Dimension mismatch: data has {data_vars} variables but weight matrix is {weight_dim}×{weight_dim}")]
    DimensionMismatch {
        data_vars: usize,
        weight_dim: usize,
    },

    #[error("Empty data matrix")]
    EmptyData,

    #[error("Weight matrix must be square")]
    NonSquareWeight,
}

/// Compute Mean Squared Error loss: MSE(W) = (1/n) * ||X - X @ W||_F^2
///
/// For each sample x_i: residual_i = x_i - W^T @ x_i
/// The MSE measures how well W explains the data through linear relationships.
///
/// # Arguments
/// * `data` - Data matrix X (n×d) where n=samples, d=variables
/// * `weight_matrix` - Weight matrix W (d×d)
///
/// # Returns
/// MSE loss value
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

    // Compute X @ W
    let xw = data.dot(weight_matrix);

    // Compute residuals: X - X @ W
    let residuals = data - &xw;

    // Frobenius norm squared
    let sum_sq: f64 = residuals.iter().map(|x| x * x).sum();

    // Divide by n for normalization
    Ok(sum_sq / n as f64)
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

/// Compute gradient of MSE loss: ∇_W MSE(W) = -2/n * X^T @ (X - X@W)
///
/// This gradient is used in the L-BFGS optimization step.
///
/// # Arguments
/// * `data` - Data matrix X (n×d)
/// * `weight_matrix` - Weight matrix W (d×d)
///
/// # Returns
/// Gradient matrix with same shape as weight_matrix
pub fn mse_gradient(data: &DataMatrix, weight_matrix: &WeightMatrix) -> Result<WeightMatrix, ScoringError> {
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

    // Gradient: -2/n * X^T @ residuals
    let gradient = data.t().dot(&residuals) * (-2.0 / n as f64);

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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_mse_zero_weights() {
        let data = ndarray::array![[1.0, 2.0], [3.0, 4.0]];
        let weights = Array2::zeros((2, 2));
        let mse = mse_loss(&data, &weights).unwrap();
        
        // With W=0, prediction is 0, so MSE = mean(data^2)
        let expected = (1.0 + 4.0 + 9.0 + 16.0) / 2.0;
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
}
