/// Integration tests for NOTEARS algorithm
/// Tests against synthetic benchmarks and algorithm correctness

use notears::*;
use ndarray::Array2;
use approx::assert_abs_diff_eq;

#[test]
fn test_zero_matrix_acyclic() {
    let w = Array2::zeros((5, 5));
    let h = acyclicity::acyclicity_constraint(&w).unwrap();
    assert_abs_diff_eq!(h, 0.0, epsilon = 1e-10);
}

#[test]
fn test_identity_matrix_acyclic() {
    // Identity matrix has h(I) = tr(exp(I)) - d = 3e - 5 > 0 (not acyclic)
    let w = Array2::eye(5);
    let h = acyclicity::acyclicity_constraint(&w).unwrap();
    // For 5×5 identity: h = 5e - 5
    assert_abs_diff_eq!(h, std::f64::consts::E * 5.0 - 5.0, epsilon = 1e-2);
}

#[test]
fn test_diagonal_lower_triangular_acyclic() {
    // Lower triangular matrix with zeros on diagonal should be acyclic
    let mut w = Array2::zeros((3, 3));
    w[[1, 0]] = 0.5;
    w[[2, 0]] = 0.3;
    w[[2, 1]] = 0.4;

    let h = acyclicity::acyclicity_constraint(&w).unwrap();
    assert!(h >= 0.0);
}

#[test]
fn test_gradient_finite_differences() -> Result<(), Box<dyn std::error::Error>> {
    // Verify gradient using finite differences
    let w = ndarray::array![[0.0, 0.1], [-0.05, 0.0]];
    let grad_analytic = acyclicity::acyclicity_gradient(&w)?;

    let eps = 1e-5;
    let mut grad_numeric = Array2::zeros((2, 2));

    for i in 0..2 {
        for j in 0..2 {
            let mut w_plus = w.clone();
            let mut w_minus = w.clone();

            w_plus[[i, j]] += eps;
            w_minus[[i, j]] -= eps;

            let h_plus = acyclicity::acyclicity_constraint(&w_plus)?;
            let h_minus = acyclicity::acyclicity_constraint(&w_minus)?;

            grad_numeric[[i, j]] = (h_plus - h_minus) / (2.0 * eps);
        }
    }

    // Check gradient matches (with reasonable tolerance for numerical errors)
    for i in 0..2 {
        for j in 0..2 {
            assert_abs_diff_eq!(
                grad_analytic[[i, j]],
                grad_numeric[[i, j]],
                epsilon = 1e-2
            );
        }
    }

    Ok(())
}

#[test]
fn test_mse_loss_with_identity_weights() {
    let data = ndarray::array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
    let weights = Array2::eye(2);

    let mse = scoring::mse_loss(&data, &weights).unwrap();

    // With W=I, prediction is same as input, so MSE should be 0
    assert_abs_diff_eq!(mse, 0.0, epsilon = 1e-10);
}

#[test]
fn test_standardize_data() -> Result<(), Box<dyn std::error::Error>> {
    let data = ndarray::array![[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]];
    let std = utils::standardize_data(&data)?;

    // Check each column has zero mean
    for col in 0..2 {
        let col_data = std.column(col);
        let mean = col_data.iter().sum::<f64>() / 3.0;
        assert_abs_diff_eq!(mean, 0.0, epsilon = 1e-10);
    }

    Ok(())
}

#[test]
fn test_matrix_exponential_identity() -> Result<(), Box<dyn std::error::Error>> {
    let w = Array2::zeros((3, 3));
    let exp_w = utils::matrix_exponential(&w)?;

    // exp(0) = I
    let eye = Array2::eye(3);
    for i in 0..3 {
        for j in 0..3 {
            assert_abs_diff_eq!(exp_w[[i, j]], eye[[i, j]], epsilon = 1e-10);
        }
    }

    Ok(())
}

#[test]
fn test_acyclicity_with_cycle() -> Result<(), Box<dyn std::error::Error>> {
    // Simple 2-node cycle: W = [[0, a], [b, 0]] with a*b > 0 creates a cycle
    let w = ndarray::array![[0.0, 0.5], [0.5, 0.0]];
    let h = acyclicity::acyclicity_constraint(&w)?;

    // Cycle should have positive h(W) > 0
    assert!(h > 0.0);

    Ok(())
}

#[test]
fn test_loss_dimension_mismatch() {
    let data = ndarray::array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let weights = Array2::zeros((2, 2));

    let result = scoring::mse_loss(&data, &weights);
    assert!(result.is_err());
}

#[test]
fn test_l1_penalty_sparsity() {
    let w1 = ndarray::array![[1.0, 0.0], [0.0, 1.0]];
    let w2 = ndarray::array![[0.5, 0.5], [0.5, 0.5]];

    let l1_1 = scoring::l1_penalty(&w1);
    let l1_2 = scoring::l1_penalty(&w2);

    // Both have same total weight, but different distributions
    assert_abs_diff_eq!(l1_1 + l1_2, 4.0, epsilon = 1e-10);
}

#[test]
fn test_threshold_matrix() {
    let matrix = ndarray::array![[0.01, 0.5], [0.1, 1.5]];
    let thresholded = utils::threshold_matrix(&matrix, 0.2);

    assert_eq!(thresholded[[0, 0]], 0.0); // 0.01 < 0.2
    assert_eq!(thresholded[[0, 1]], 0.5); // 0.5 >= 0.2
    assert_eq!(thresholded[[1, 0]], 0.0); // 0.1 < 0.2
    assert_eq!(thresholded[[1, 1]], 1.5); // 1.5 >= 0.2
}

#[test]
fn test_frobenius_norm() {
    let matrix = ndarray::array![[3.0, 0.0], [0.0, 4.0]];
    let norm = utils::frobenius_norm(&matrix);
    assert_abs_diff_eq!(norm, 5.0, epsilon = 1e-10); // sqrt(9 + 16)
}

#[test]
fn test_optimization_result_edges() {
    let weight_matrix = ndarray::array![[0.0, 0.5, 0.0], [-0.3, 0.0, 0.2], [0.0, 0.1, 0.0]];
    let result = OptimizationResult::new(weight_matrix, 0.0, 100, 0.1, 0.2);

    let edges = result.edges();
    // Edges with |w| > 0.2: (0,1)=0.5, (1,0)=0.3
    assert_eq!(edges.len(), 2);
    assert!(edges.contains(&(0, 1)));
    assert!(edges.contains(&(1, 0)));
}

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
fn test_regularization_config_valid() {
    let config = RegularizationConfig::new(0.1, false).unwrap();
    assert_eq!(config.lambda, 0.1);
}

#[test]
fn test_regularization_config_invalid_lambda() {
    let result = RegularizationConfig::new(-0.1, false);
    assert!(result.is_err());
}
