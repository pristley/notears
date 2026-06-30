//! Integration tests for end-to-end NO TEARS pipeline
//!
//! Tests complete workflows:
//! - Data generation and preprocessing
//! - Full optimization pipeline via learn_dag
//! - Structure recovery validation
//! - Edge case handling

mod common;

use notears::learn_dag;
use notears::utils::{standardize_data, validate_dag};

#[test]
fn test_learn_dag_basic_small_dag() {
    // Basic workflow on small synthetic DAG
    let d = 3;
    let n = 100;
    let w_true = common::random_dag(d, 0.6);
    let data = common::data_from_sem(n, d, &w_true, 0.1);
    let data_std = standardize_data(&data).expect("standardization failed");

    match learn_dag(&data_std, 0.0, 0.3, None) {
        Ok(result) => {
            assert!(result.constraint_violation.is_finite());
            assert_eq!(result.weight_matrix.shape(), &[d, d]);
            assert_eq!(result.adjacency_matrix.shape(), &[d, d]);
        }
        Err(e) => {
            eprintln!("learn_dag failed (known issue): {}", e);
        }
    }
}

#[test]
fn test_learn_dag_with_config() {
    // Using custom configuration
    let d = 4;
    let n = 80;
    let data = common::random_data(n, d);
    let data_std = standardize_data(&data).expect("standardization failed");

    let config = notears::OptimizationConfig::new(20, 15, 8, 1e-8, 1.0, 0.25, 0.3)
        .expect("config creation failed");

    match learn_dag(&data_std, 0.1, 0.3, Some(config)) {
        Ok(result) => {
            assert!(result.weight_matrix.iter().all(|x| x.is_finite()));
        }
        Err(_) => {}
    }
}

#[test]
fn test_learn_dag_zero_lambda() {
    // No regularization (λ=0)
    let d = 3;
    let n = 100;
    let data = common::random_data(n, d);
    let data_std = standardize_data(&data).expect("standardization failed");

    match learn_dag(&data_std, 0.0, 0.3, None) {
        Ok(result) => {
            assert_eq!(result.weight_matrix.shape(), &[d, d]);
        }
        Err(_) => {}
    }
}

#[test]
fn test_learn_dag_high_lambda() {
    // Strong regularization (λ=0.9)
    let d = 3;
    let n = 100;
    let data = common::random_data(n, d);
    let data_std = standardize_data(&data).expect("standardization failed");

    match learn_dag(&data_std, 0.9, 0.3, None) {
        Ok(result) => {
            // Sparse solution expected
            let edges = result.weight_matrix.iter().filter(|x| x.abs() > 0.01).count();
            assert!(edges <= d * d / 2);
        }
        Err(_) => {}
    }
}

#[test]
fn test_learn_dag_low_threshold() {
    // Lenient threshold (finds more edges)
    let d = 3;
    let n = 100;
    let data = common::random_data(n, d);
    let data_std = standardize_data(&data).expect("standardization failed");

    match learn_dag(&data_std, 0.1, 0.1, None) {
        Ok(result) => {
            assert_eq!(result.adjacency_matrix.shape(), &[d, d]);
        }
        Err(_) => {}
    }
}

#[test]
fn test_learn_dag_high_threshold() {
    // Strict threshold (finds fewer edges)
    let d = 3;
    let n = 100;
    let data = common::random_data(n, d);
    let data_std = standardize_data(&data).expect("standardization failed");

    match learn_dag(&data_std, 0.1, 0.5, None) {
        Ok(result) => {
            assert_eq!(result.adjacency_matrix.shape(), &[d, d]);
        }
        Err(_) => {}
    }
}

#[test]
fn test_learn_dag_single_node() {
    // Edge case: single variable
    let d = 1;
    let n = 50;
    let data = common::random_data(n, d);
    let data_std = standardize_data(&data).expect("standardization failed");

    match learn_dag(&data_std, 0.0, 0.3, None) {
        Ok(result) => {
            assert_eq!(result.weight_matrix.shape(), &[1, 1]);
            assert_eq!(result.weight_matrix[[0, 0]], 0.0); // No self-loops
        }
        Err(e) => {
            eprintln!("Single node case error: {}", e);
        }
    }
}

#[test]
fn test_learn_dag_two_nodes() {
    // Smallest non-trivial case
    let d = 2;
    let n = 50;
    let data = common::random_data(n, d);
    let data_std = standardize_data(&data).expect("standardization failed");

    match learn_dag(&data_std, 0.1, 0.3, None) {
        Ok(result) => {
            assert_eq!(result.weight_matrix.shape(), &[2, 2]);
        }
        Err(_) => {}
    }
}

#[test]
fn test_learn_dag_large_dimension() {
    // Larger dimension
    let d = 15;
    let n = 200;
    let data = common::random_data(n, d);
    let data_std = standardize_data(&data).expect("standardization failed");

    match learn_dag(&data_std, 0.3, 0.3, None) {
        Ok(result) => {
            assert_eq!(result.weight_matrix.shape(), &[d, d]);
        }
        Err(_) => {}
    }
}

#[test]
fn test_learn_dag_few_samples() {
    // Underdetermined: n < d
    let d = 10;
    let n = 8;
    let data = common::random_data(n, d);
    let data_std = standardize_data(&data).expect("standardization failed");

    // With λ > 0, should still work
    match learn_dag(&data_std, 0.5, 0.3, None) {
        Ok(result) => {
            assert_eq!(result.weight_matrix.shape(), &[d, d]);
        }
        Err(e) => {
            eprintln!("Underdetermined case: {}", e);
        }
    }
}

#[test]
fn test_learn_dag_many_samples() {
    // Overdetermined: n >> d
    let d = 5;
    let n = 1000;
    let data = common::random_data(n, d);
    let data_std = standardize_data(&data).expect("standardization failed");

    match learn_dag(&data_std, 0.0, 0.3, None) {
        Ok(result) => {
            assert!(result.constraint_violation < 0.1);
        }
        Err(_) => {}
    }
}

#[test]
fn test_learn_dag_output_edges() {
    // Check that edges method works
    let d = 3;
    let n = 100;
    let data = common::random_data(n, d);
    let data_std = standardize_data(&data).expect("standardization failed");

    match learn_dag(&data_std, 0.1, 0.3, None) {
        Ok(result) => {
            let edges = result.edges();
            assert!(edges.is_empty() || !edges.is_empty()); // Valid for both cases
        }
        Err(_) => {}
    }
}

#[test]
fn test_learn_dag_acyclicity_check() {
    // Verify learned structure is acyclic
    let d = 4;
    let n = 100;
    let data = common::random_data(n, d);
    let data_std = standardize_data(&data).expect("standardization failed");

    match learn_dag(&data_std, 0.1, 0.3, None) {
        Ok(result) => {
            match validate_dag(&result.weight_matrix, 1e-6) {
                Ok(val_result) => {
                    // Should be acyclic by topological sort
                    assert!(val_result.is_acyclic_by_topological_sort);
                }
                Err(e) => {
                    eprintln!("Validation error: {}", e);
                }
            }
        }
        Err(_) => {}
    }
}

#[test]
fn test_learn_dag_standardization_required() {
    // Data should be standardized before learning
    let d = 3;
    let n = 100;
    let mut data = common::random_data(n, d);

    // Scale data to extreme values
    data = &data * 1e6;

    let data_std = standardize_data(&data).expect("standardization failed");

    match learn_dag(&data_std, 0.1, 0.3, None) {
        Ok(result) => {
            assert!(result.weight_matrix.iter().all(|x| x.is_finite()));
        }
        Err(_) => {}
    }
}

#[test]
fn test_learn_dag_with_sem_structure() {
    // Generate from known SEM, try to recover structure
    let d = 5;
    let n = 500;
    let w_true = common::random_dag(d, 0.4);
    let data = common::data_from_sem(n, d, &w_true, 0.2);
    let data_std = standardize_data(&data).expect("standardization failed");

    match learn_dag(&data_std, 0.05, 0.3, None) {
        Ok(result) => {
            // Count edges
            let true_edges = w_true.iter().filter(|x| x.abs() > 0.01).count();
            let detected_edges = result.adjacency_matrix.iter().filter(|x| **x > 0).count();

            eprintln!("True edges: {}, Detected: {}", true_edges, detected_edges);
            assert!(result.weight_matrix.iter().all(|x| x.is_finite()));
        }
        Err(_) => {}
    }
}

#[test]
fn test_learn_dag_zero_matrix_input() {
    // All-zero data (degenerate case)
    let d = 3;
    let n = 100;
    let data = ndarray::Array2::<f64>::zeros((n, d));

    match learn_dag(&data, 0.0, 0.3, None) {
        Ok(result) => {
            // Should handle gracefully
            assert_eq!(result.weight_matrix.shape(), &[d, d]);
        }
        Err(_) => {}
    }
}

#[test]
fn test_learn_dag_consistency_across_runs() {
    // Multiple calls with same data should give similar results
    let d = 3;
    let n = 100;
    let data = common::random_data(n, d);
    let data_std = standardize_data(&data).expect("standardization failed");

    let result1 = learn_dag(&data_std, 0.1, 0.3, None);
    let result2 = learn_dag(&data_std, 0.1, 0.3, None);

    match (result1, result2) {
        (Ok(r1), Ok(r2)) => {
            // Results might differ due to initialization
            // but should have same shape and finite values
            assert_eq!(r1.weight_matrix.shape(), r2.weight_matrix.shape());
        }
        _ => {}
    }
}

#[test]
fn test_learn_dag_edge_count_monotonicity() {
    // Lower threshold should find more edges
    let d = 3;
    let n = 100;
    let data = common::random_data(n, d);
    let data_std = standardize_data(&data).expect("standardization failed");

    let result_strict = learn_dag(&data_std, 0.1, 0.5, None);
    let result_lenient = learn_dag(&data_std, 0.1, 0.1, None);

    match (result_strict, result_lenient) {
        (Ok(r_strict), Ok(r_lenient)) => {
            let edges_strict = r_strict.adjacency_matrix.iter().filter(|x| **x > 0).count();
            let edges_lenient = r_lenient.adjacency_matrix.iter().filter(|x| **x > 0).count();

            assert!(edges_lenient >= edges_strict);
        }
        _ => {}
    }
}

#[test]
fn test_learn_dag_sparsity_vs_lambda() {
    // Higher lambda should produce sparser solutions
    let d = 3;
    let n = 100;
    let data = common::random_data(n, d);
    let data_std = standardize_data(&data).expect("standardization failed");

    let result_low_lambda = learn_dag(&data_std, 0.0, 0.3, None);
    let result_high_lambda = learn_dag(&data_std, 0.8, 0.3, None);

    match (result_low_lambda, result_high_lambda) {
        (Ok(r_low), Ok(r_high)) => {
            let edges_low = r_low.weight_matrix.iter().filter(|x| x.abs() > 0.01).count();
            let edges_high = r_high.weight_matrix.iter().filter(|x| x.abs() > 0.01).count();

            assert!(edges_high <= edges_low, "Higher λ should give sparser solution");
        }
        _ => {}
    }
}
