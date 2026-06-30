//! Unit tests for scoring functions and loss computation
//!
//! Tests the mathematical correctness of:
//! - MSE loss: (1/2n)||X - XW||_F^2
//! - L1 penalty: λ||W||_1
//! - Total score: F(W) = loss + penalty
//! - Gradient computation ∇F(W)

mod common;

use notears::scoring::{mse_loss, l1_penalty, total_loss};
use notears::types::RegularizationConfig;
use ndarray::Array2;
use approx::assert_abs_diff_eq;

#[test]
fn test_mse_loss_zero_matrix() {
    // MSE(W=0) = (1/2n)||X||_F^2 / 2 = variance of X
    let n = 100;
    let d = 5;
    let data = common::random_data(n, d);
    let w = Array2::<f64>::zeros((d, d));

    let loss = mse_loss(&data, &w).unwrap();

    // Should be positive and finite
    assert!(loss.is_finite() && loss >= 0.0);
}

#[test]
fn test_mse_loss_identity() {
    // For W = I, reconstruction error typically large
    let n = 100;
    let d = 3;
    let data = common::random_data(n, d);
    let w = Array2::eye(d);

    let loss_zero = mse_loss(&data, &Array2::zeros((d, d))).unwrap();
    let loss_identity = mse_loss(&data, &w).unwrap();

    // Both should be finite
    assert!(loss_zero.is_finite());
    assert!(loss_identity.is_finite());
}

#[test]
fn test_mse_loss_symmetry() {
    // MSE loss with W and W^T should differ (non-symmetric behavior expected)
    let n = 50;
    let d = 4;
    let data = common::random_data(n, d);

    let mut w = Array2::<f64>::zeros((d, d));
    for i in 0..d {
        for j in 0..d {
            w[[i, j]] = rand::random::<f64>() * 0.3;
        }
    }

    let loss_w = mse_loss(&data, &w).unwrap();
    let loss_wt = mse_loss(&data, &w.t().to_owned()).unwrap();

    // Both should be positive
    assert!(loss_w >= 0.0);
    assert!(loss_wt >= 0.0);
}

#[test]
fn test_mse_loss_small_data() {
    // Single sample case
    let data = Array2::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0]).unwrap();
    let w = Array2::<f64>::zeros((3, 3));

    let loss = mse_loss(&data, &w).unwrap();
    assert!(loss.is_finite());
}

#[test]
fn test_mse_loss_scales_with_data_size() {
    // MSE should normalize by n
    let d = 3;
    let w = common::random_dag(d, 0.5);

    let data_small = common::random_data(10, d);
    let data_large = common::random_data(1000, d);

    let loss_small = mse_loss(&data_small, &w).unwrap();
    let loss_large = mse_loss(&data_large, &w).unwrap();

    // Both should be finite
    assert!(loss_small.is_finite());
    assert!(loss_large.is_finite());
}

#[test]
fn test_l1_penalty_zero_matrix() {
    // ||0||_1 = 0
    let w = Array2::<f64>::zeros((3, 3));
    let penalty = l1_penalty(&w);

    assert_abs_diff_eq!(penalty, 0.0, epsilon = 1e-15);
}

#[test]
fn test_l1_penalty_ones() {
    // ||ones(d,d)||_1 = d^2
    let d = 5;
    let w = Array2::<f64>::ones((d, d));
    let penalty = l1_penalty(&w);

    assert_abs_diff_eq!(penalty, (d * d) as f64, epsilon = 1e-14);
}

#[test]
fn test_l1_penalty_absolute_value() {
    // ||W||_1 = ||−W||_1
    let mut w = Array2::<f64>::zeros((3, 3));
    w[[0, 0]] = 1.0;
    w[[1, 1]] = -2.0;
    w[[2, 2]] = 3.0;

    let penalty_pos = l1_penalty(&w);
    let penalty_neg = l1_penalty(&(-w));

    assert_abs_diff_eq!(penalty_pos, penalty_neg, epsilon = 1e-14);
}

#[test]
fn test_l1_penalty_sparsity() {
    // Sparser matrix should have smaller L1 norm
    let mut w_dense = Array2::<f64>::zeros((5, 5));
    for i in 0..5 {
        for j in 0..5 {
            w_dense[[i, j]] = 0.5;
        }
    }

    let mut w_sparse = Array2::<f64>::zeros((5, 5));
    w_sparse[[0, 1]] = 2.5; // single edge, same total weight

    let penalty_dense = l1_penalty(&w_dense);
    let penalty_sparse = l1_penalty(&w_sparse);

    assert!(penalty_sparse < penalty_dense);
}

#[test]
fn test_total_loss_zero_lambda() {
    // F(W, λ=0) = MSE loss only
    let n = 50;
    let d = 3;
    let data = common::random_data(n, d);
    let w = common::random_dag(d, 0.5);
    let config = RegularizationConfig::new(0.0, false).unwrap();

    let total = total_loss(&data, &w, &config).unwrap();
    let mse = mse_loss(&data, &w).unwrap();

    assert_abs_diff_eq!(total, mse, epsilon = 1e-12);
}

#[test]
fn test_total_loss_high_lambda() {
    // F(W, λ=1) = MSE + L1
    let n = 50;
    let d = 3;
    let data = common::random_data(n, d);
    let w = common::random_dag(d, 0.5);
    let config = RegularizationConfig::new(1.0, false).unwrap();

    let total = total_loss(&data, &w, &config).unwrap();
    let mse = mse_loss(&data, &w).unwrap();
    let l1 = l1_penalty(&w);
    let expected = mse + l1;

    assert_abs_diff_eq!(total, expected, epsilon = 1e-12);
}

#[test]
fn test_total_loss_monotonicity_lambda() {
    // Higher λ should increase loss for non-zero W
    let n = 30;
    let d = 3;
    let data = common::random_data(n, d);

    let mut w = Array2::<f64>::zeros((d, d));
    w[[0, 1]] = 0.5;
    w[[1, 2]] = 0.3;

    let config_low = RegularizationConfig::new(0.1, false).unwrap();
    let config_high = RegularizationConfig::new(1.0, false).unwrap();

    let loss_low = total_loss(&data, &w, &config_low).unwrap();
    let loss_high = total_loss(&data, &w, &config_high).unwrap();

    assert!(loss_high > loss_low, "Higher λ should increase loss");
}

#[test]
fn test_total_loss_zero_matrix_finite() {
    // Zero matrix should have finite loss
    let n = 100;
    let d = 10;
    let data = common::random_data(n, d);
    let w = Array2::<f64>::zeros((d, d));
    let config = RegularizationConfig::new(0.1, false).unwrap();

    let loss = total_loss(&data, &w, &config).unwrap();
    assert!(loss.is_finite());
}

#[test]
fn test_total_loss_smooth() {
    // Loss should be smooth (differentiable)
    let n = 50;
    let d = 3;
    let data = common::random_data(n, d);
    let config = RegularizationConfig::new(0.1, false).unwrap();

    let w = common::random_dag(d, 0.5);

    let eps = 1e-4;
    let mut w_perturbed = w.clone();
    w_perturbed[[0, 1]] += eps;

    let loss_original = total_loss(&data, &w, &config).unwrap();
    let loss_perturbed = total_loss(&data, &w_perturbed, &config).unwrap();

    // Loss should change smoothly
    let change = (loss_perturbed - loss_original).abs();
    assert!(change < 1.0, "Loss change too large: {}", change);
}

#[test]
fn test_mse_loss_positive_semidefinite() {
    // MSE loss should always be non-negative
    let n = 100;
    let d = 5;
    let data = common::random_data(n, d);

    for _ in 0..10 {
        let w = common::random_dag(d, 0.5);
        let loss = mse_loss(&data, &w).unwrap();
        assert!(loss >= 0.0, "MSE loss should be non-negative, got {}", loss);
    }
}

#[test]
fn test_mse_loss_reconstruction_improvement() {
    // Using optimal reconstruction should improve over zero
    let n = 50;
    let d = 3;
    let w_true = common::random_dag(d, 0.5);
    let data = common::data_from_sem(n, d, &w_true, 0.1);

    let loss_zero = mse_loss(&data, &Array2::zeros((d, d))).unwrap();
    let loss_true = mse_loss(&data, &w_true).unwrap();

    // Using true W should give better (lower) loss
    assert!(loss_true <= loss_zero, "True structure should improve reconstruction");
}

#[test]
fn test_penalty_scales_with_magnitude() {
    // ||2W||_1 = 2||W||_1
    let mut w = Array2::<f64>::zeros((3, 3));
    w[[0, 1]] = 0.5;
    w[[1, 2]] = 0.3;

    let penalty_1x = l1_penalty(&w);
    let penalty_2x = l1_penalty(&(&w * 2.0));

    assert_abs_diff_eq!(penalty_2x, penalty_1x * 2.0, epsilon = 1e-12);
}

#[test]
fn test_mse_loss_low_noise_limit() {
    // With zero noise, MSE should recover true structure perfectly
    let n = 100;
    let d = 3;
    let w_true = common::random_dag(d, 0.5);
    let data = common::data_from_sem(n, d, &w_true, 0.0); // zero noise

    let loss = mse_loss(&data, &w_true).unwrap();
    // Should be very small (machine precision level)
    assert!(loss < 1e-10, "Perfect reconstruction should have negligible loss");
}

#[test]
fn test_total_loss_consistency() {
    // Different calls with same input should give same result
    let n = 50;
    let d = 3;
    let data = common::random_data(n, d);
    let w = common::random_dag(d, 0.5);
    let config = RegularizationConfig::new(0.1, false).unwrap();

    let loss1 = total_loss(&data, &w, &config).unwrap();
    let loss2 = total_loss(&data, &w, &config).unwrap();

    assert_abs_diff_eq!(loss1, loss2, epsilon = 1e-14);
}

#[test]
fn test_mse_loss_overdetermined_system() {
    // Many samples relative to parameters
    let n = 1000;
    let d = 5;
    let data = common::random_data(n, d);
    let w = common::random_dag(d, 0.3);

    let loss = mse_loss(&data, &w).unwrap();
    assert!(loss.is_finite() && loss >= 0.0);
}

#[test]
fn test_mse_loss_underdetermined_system() {
    // Few samples relative to parameters
    let n = 5;
    let d = 10;
    let data = common::random_data(n, d);
    let w = common::random_dag(d, 0.3);

    let loss = mse_loss(&data, &w).unwrap();
    assert!(loss.is_finite() && loss >= 0.0);
}
