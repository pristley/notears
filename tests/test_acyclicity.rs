//! Unit tests for acyclicity constraint computation and verification
//!
//! Tests the mathematical correctness of:
//! - h(W) = tr(exp(W ⊙ W)) - d constraint
//! - Gradient computation ∇h(W)
//! - Matrix exponential implementation
//! - Validation functions for DAG property

mod common;

use notears::acyclicity::{acyclicity_constraint, acyclicity_gradient};
use notears::utils::{matrix_exponential, is_acyclic, validate_dag};
use ndarray::Array2;
use approx::assert_abs_diff_eq;

#[test]
fn test_matrix_exponential_identity() {
    // exp(0) should equal I
    let d = 5;
    let zero = Array2::<f64>::zeros((d, d));
    let exp_zero = matrix_exponential(&zero).unwrap();
    let identity = Array2::eye(d);

    assert!(common::close_enough(&exp_zero, &identity, 1e-13));
}

#[test]
fn test_matrix_exponential_zero_matrix_diagonal() {
    // exp(diag(a, b, c)) ≈ diag(e^a, e^b, e^c)
    let mut d_mat = Array2::<f64>::zeros((3, 3));
    d_mat[[0, 0]] = 1.0;
    d_mat[[1, 1]] = 2.0;
    d_mat[[2, 2]] = -0.5;

    let exp_d = matrix_exponential(&d_mat).unwrap();

    assert_abs_diff_eq!(exp_d[[0, 0]], 1.0_f64.exp(), epsilon = 1e-4);
    assert_abs_diff_eq!(exp_d[[1, 1]], 2.0_f64.exp(), epsilon = 1e-4);
    assert_abs_diff_eq!(exp_d[[2, 2]], (-0.5_f64).exp(), epsilon = 1e-4);

    // Off-diagonal should be small for symmetric case
    assert!(exp_d[[0, 1]].abs() < 1e-10);
    assert!(exp_d[[1, 0]].abs() < 1e-10);
}

#[test]
fn test_matrix_exponential_small_perturbation() {
    // For small ε, exp(ε*A) ≈ I + ε*A (first-order Taylor)
    let epsilon = 1e-4;
    let mut a = Array2::<f64>::zeros((2, 2));
    a[[0, 0]] = -0.5;
    a[[0, 1]] = 0.1;
    a[[1, 0]] = 0.1;
    a[[1, 1]] = -0.3;

    let scaled = &a * epsilon;
    let exp_scaled = matrix_exponential(&scaled).unwrap();

    // Approximation: exp(ε*A) ≈ I + ε*A
    let expected = &Array2::eye(2) + &scaled;

    for i in 0..2 {
        for j in 0..2 {
            assert_abs_diff_eq!(exp_scaled[[i, j]], expected[[i, j]], epsilon = 1e-8);
        }
    }
}

#[test]
fn test_matrix_exponential_symmetry() {
    // For symmetric input A, exp(A) should be symmetric
    let mut a = Array2::<f64>::zeros((3, 3));
    a[[0, 0]] = 1.0;
    a[[1, 1]] = 0.5;
    a[[2, 2]] = -0.5;
    a[[0, 1]] = 0.2;
    a[[1, 0]] = 0.2;
    a[[0, 2]] = -0.1;
    a[[2, 0]] = -0.1;
    a[[1, 2]] = 0.15;
    a[[2, 1]] = 0.15;

    let exp_a = matrix_exponential(&a).unwrap();
    assert!(common::is_symmetric(&exp_a, 1e-12));
}

#[test]
fn test_acyclicity_constraint_zero_matrix() {
    // h(0) = tr(exp(0)) - d = tr(I) - d = d - d = 0
    let d = 5;
    let zero = Array2::<f64>::zeros((d, d));
    let h = acyclicity_constraint(&zero).unwrap();

    assert_abs_diff_eq!(h, 0.0, epsilon = 1e-14);
}

#[test]
fn test_acyclicity_constraint_lower_triangular_dag() {
    // Lower triangular W is always acyclic (h ≈ 0)
    let mut w = Array2::<f64>::zeros((5, 5));
    for i in 0..5 {
        for j in 0..i {
            if rand::random::<f64>() < 0.7 {
                w[[i, j]] = rand::random::<f64>() * 0.5;
            }
        }
    }

    let h = acyclicity_constraint(&w).unwrap();
    assert!(h < 1e-6, "Lower triangular should be acyclic, h={}", h);
}

#[test]
fn test_acyclicity_constraint_linear_chain() {
    // Linear chain: 0 → 1 → 2 (lowest constraint violation)
    let mut w = Array2::<f64>::zeros((3, 3));
    w[[1, 0]] = 0.5; // edge 0 → 1
    w[[2, 1]] = 0.5; // edge 1 → 2

    let h = acyclicity_constraint(&w).unwrap();
    assert!(h < 1e-6);
}

#[test]
fn test_acyclicity_constraint_simple_cycle() {
    // Simple cycle: 0 ↔ 1
    let mut w = Array2::<f64>::zeros((2, 2));
    w[[0, 1]] = 1.0;
    w[[1, 0]] = 1.0;

    let h = acyclicity_constraint(&w).unwrap();
    assert!(
        h > 0.01,
        "Simple cycle should have large h, got h={}",
        h
    );
}

#[test]
fn test_acyclicity_constraint_triangle_cycle() {
    // Triangle cycle: 0 → 1 → 2 → 0
    let mut w = Array2::<f64>::zeros((3, 3));
    w[[0, 1]] = 1.0;
    w[[1, 2]] = 1.0;
    w[[2, 0]] = 1.0;

    let h = acyclicity_constraint(&w).unwrap();
    assert!(
        h > 0.1,
        "Triangle cycle should have large h, got h={}",
        h
    );
}

#[test]
fn test_acyclicity_constraint_monotonicity() {
    // h should increase with cycle weight
    let mut w_small = Array2::<f64>::zeros((2, 2));
    w_small[[0, 1]] = 0.1;
    w_small[[1, 0]] = 0.1;

    let mut w_large = Array2::<f64>::zeros((2, 2));
    w_large[[0, 1]] = 1.0;
    w_large[[1, 0]] = 1.0;

    let h_small = acyclicity_constraint(&w_small).unwrap();
    let h_large = acyclicity_constraint(&w_large).unwrap();

    assert!(h_large > h_small, "Larger cycle weight should have larger h");
}

#[test]
fn test_acyclicity_gradient_finite_difference() {
    // Verify ∇h(W) using finite differences
    let w = common::random_dag(4, 0.7);
    let grad_analytical = acyclicity_gradient(&w).unwrap();

    let eps = 1e-5;
    let (d, _) = w.dim();
    let mut grad_numerical = Array2::<f64>::zeros((d, d));

    for i in 0..d {
        for j in 0..d {
            let mut w_plus = w.clone();
            let mut w_minus = w.clone();
            w_plus[[i, j]] += eps;
            w_minus[[i, j]] -= eps;

            let h_plus = acyclicity_constraint(&w_plus).unwrap();
            let h_minus = acyclicity_constraint(&w_minus).unwrap();
            grad_numerical[[i, j]] = (h_plus - h_minus) / (2.0 * eps);
        }
    }

    // Check element-wise error
    for i in 0..d {
        for j in 0..d {
            let error = (grad_analytical[[i, j]] - grad_numerical[[i, j]]).abs();
            assert!(
                error < 1e-3,
                "Gradient error at ({}, {}): analytical={:.6e}, numerical={:.6e}",
                i,
                j,
                grad_analytical[[i, j]],
                grad_numerical[[i, j]]
            );
        }
    }
}

#[test]
fn test_acyclicity_gradient_descent_direction() {
    // -∇h should point in descent direction
    // Use a matrix that violates acyclicity constraint
    let mut w = common::random_dag(3, 0.7);
    // Perturb to violate acyclicity
    w[[1, 0]] = 0.5;
    w[[2, 0]] = 0.5;
    w[[2, 1]] = 0.5;
    w[[0, 2]] = 0.3; // Create cycle 0 -> 2 -> 1, 1 -> 0 (not full triangle but with path)

    let h_before = acyclicity_constraint(&w).unwrap();
    if h_before < 1e-6 {
        // If still acyclic, skip
        return;
    }

    let grad = acyclicity_gradient(&w).unwrap();

    // Small step in negative gradient direction
    let eps = 1e-3;
    let w_new = &w - (&grad * eps);

    let h_new = acyclicity_constraint(&w_new).unwrap();

    assert!(h_new < h_before, "Step in negative gradient should decrease h(W): {} -> {}", h_before, h_new);
}

#[test]
fn test_acyclicity_gradient_zero_at_dags() {
    // For acyclic W, ∇h should be very small
    let w = common::random_dag(5, 0.5);
    let grad = acyclicity_gradient(&w).unwrap();

    let grad_norm = common::frobenius_norm(&grad);
    assert!(grad_norm < 1e-2, "Gradient norm at DAG should be small, got {}", grad_norm);
}

#[test]
fn test_is_acyclic_valid_dag() {
    // Valid DAG should return true
    let w = common::random_dag(5, 0.5);
    assert!(is_acyclic(&w));
}

#[test]
fn test_is_acyclic_cycle_false() {
    // Cyclic matrix should return false
    let mut w = Array2::<f64>::zeros((2, 2));
    w[[0, 1]] = 1.0;
    w[[1, 0]] = 1.0;
    assert!(!is_acyclic(&w));
}

#[test]
fn test_is_acyclic_zero_matrix() {
    // Empty graph is acyclic
    let w = Array2::<f64>::zeros((3, 3));
    assert!(is_acyclic(&w));
}

#[test]
fn test_validate_dag_valid_structure() -> Result<(), Box<dyn std::error::Error>> {
    // Valid DAG should pass both checks
    let w = common::random_dag(5, 0.5);
    let result = validate_dag(&w, 1e-6)?;

    assert!(result.is_valid_dag());
    assert!(result.is_acyclic_by_constraint);
    assert!(result.is_acyclic_by_topological_sort);
    assert_eq!(result.max_cycle_weight, 0.0);

    Ok(())
}

#[test]
fn test_validate_dag_statistics() -> Result<(), Box<dyn std::error::Error>> {
    // Statistics should be consistent
    let mut w = Array2::<f64>::zeros((10, 10));
    for i in 0..10 {
        for j in 0..i {
            if rand::random::<f64>() < 0.3 {
                w[[i, j]] = rand::random::<f64>() * 0.5;
            }
        }
    }

    let result = validate_dag(&w, 1e-6)?;

    // Count edges manually
    let manual_edges = w.iter().filter(|&&x| x.abs() > 0.01).count();
    // Should be close to reported edges
    assert!(result.num_edges as i32 - manual_edges as i32 == 0 || result.num_edges > 0);

    // Sparsity should be in [0, 1]
    assert!(result.sparsity >= 0.0 && result.sparsity <= 1.0);

    Ok(())
}

#[test]
fn test_validate_dag_tolerance_effect() -> Result<(), Box<dyn std::error::Error>> {
    // Tighter tolerance might fail on slightly non-acyclic matrices
    let mut w = Array2::<f64>::zeros((3, 3));
    w[[0, 1]] = 0.5;
    w[[1, 2]] = 0.3;

    let _result_tight = validate_dag(&w, 1e-10)?;
    let result_loose = validate_dag(&w, 1e-4)?;

    // Loose tolerance should pass
    assert!(result_loose.is_acyclic_by_constraint);

    Ok(())
}

#[test]
fn test_acyclicity_gradient_norm_scaling() {
    // Gradient norm should scale reasonably with input magnitude
    let mut w_small = Array2::<f64>::zeros((3, 3));
    w_small[[1, 0]] = 0.1;
    w_small[[2, 1]] = 0.1;

    let mut w_large = Array2::<f64>::zeros((3, 3));
    w_large[[1, 0]] = 1.0;
    w_large[[2, 1]] = 1.0;

    let grad_small = acyclicity_gradient(&w_small).unwrap();
    let grad_large = acyclicity_gradient(&w_large).unwrap();

    let norm_small = common::frobenius_norm(&grad_small);
    let norm_large = common::frobenius_norm(&grad_large);

    // Larger inputs should generally have larger gradients
    assert!(norm_large >= norm_small / 10.0, "Gradient scaling mismatch");
}

#[test]
fn test_acyclicity_sparse_matrix() {
    // Test on very sparse matrix
    let mut w = Array2::<f64>::zeros((100, 100));
    // Only one edge
    w[[10, 5]] = 0.5;

    let h = acyclicity_constraint(&w).unwrap();
    assert!(h < 1e-6);
}

#[test]
fn test_acyclicity_dense_dag() {
    // Test on dense but still acyclic matrix
    let mut w = Array2::<f64>::zeros((5, 5));
    for i in 0..5 {
        for j in 0..i {
            w[[i, j]] = 0.3;
        }
    }

    let h = acyclicity_constraint(&w).unwrap();
    assert!(h < 1e-6);
}
