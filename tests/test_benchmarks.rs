//! Benchmark tests for performance profiling
//!
//! Tests execution speed and resource usage of key operations:
//! - Matrix exponential computation
//! - Acyclicity constraint evaluation
//! - Scoring functions
//! - Optimization iterations

mod common;

use notears::acyclicity::acyclicity_constraint;
use notears::scoring::{mse_loss, l1_penalty};
use notears::utils::matrix_exponential;
use notears::types::RegularizationConfig;
use std::time::Instant;

/// Simple timing utility
fn time_operation<F: FnOnce() -> R, R>(op: F) -> (R, u128) {
    let start = Instant::now();
    let result = op();
    let elapsed = start.elapsed().as_micros();
    (result, elapsed)
}

#[test]
fn bench_matrix_exponential_small() {
    // Time matrix_exp for small matrices
    let d = 5;
    let w = common::random_dag(d, 0.5);

    let (_, time_us) = time_operation(|| {
        matrix_exponential(&w).unwrap()
    });

    println!("Matrix exp ({} x {}): {} μs", d, d, time_us);
    assert!(time_us < 10000); // Should be < 10ms for small matrix
}

#[test]
fn bench_matrix_exponential_medium() {
    // Time matrix_exp for medium matrices
    let d = 20;
    let w = common::random_dag(d, 0.3);

    let (_, time_us) = time_operation(|| {
        matrix_exponential(&w).unwrap()
    });

    println!("Matrix exp ({} x {}): {} μs", d, d, time_us);
    assert!(time_us < 100000); // Should be < 100ms for medium matrix
}

#[test]
#[ignore] // Skip by default - takes longer
fn bench_matrix_exponential_large() {
    // Time matrix_exp for large matrices
    let d = 50;
    let w = common::random_dag(d, 0.2);

    let (_, time_us) = time_operation(|| {
        matrix_exponential(&w).unwrap()
    });

    println!("Matrix exp ({} x {}): {} μs", d, d, time_us);
}

#[test]
fn bench_acyclicity_constraint_small() {
    // Time h(W) computation for small matrices
    let d = 5;
    let w = common::random_dag(d, 0.5);

    let (_, time_us) = time_operation(|| {
        acyclicity_constraint(&w).unwrap()
    });

    println!("Acyclicity constraint ({} x {}): {} μs", d, d, time_us);
    assert!(time_us < 20000);
}

#[test]
fn bench_acyclicity_constraint_medium() {
    // Time h(W) computation for medium matrices
    let d = 20;
    let w = common::random_dag(d, 0.3);

    let (_, time_us) = time_operation(|| {
        acyclicity_constraint(&w).unwrap()
    });

    println!("Acyclicity constraint ({} x {}): {} μs", d, d, time_us);
    assert!(time_us < 200000);
}

#[test]
fn bench_mse_loss_small() {
    // Time MSE loss for small data
    let n = 50;
    let d = 5;
    let data = common::random_data(n, d);
    let w = common::random_dag(d, 0.5);

    let (_, time_us) = time_operation(|| {
        mse_loss(&data, &w).unwrap()
    });

    println!("MSE loss ({} x {}): {} μs", n, d, time_us);
    assert!(time_us < 5000);
}

#[test]
fn bench_mse_loss_large() {
    // Time MSE loss for large data
    let n = 1000;
    let d = 20;
    let data = common::random_data(n, d);
    let w = common::random_dag(d, 0.3);

    let (_, time_us) = time_operation(|| {
        mse_loss(&data, &w).unwrap()
    });

    println!("MSE loss ({} x {}): {} μs", n, d, time_us);
    assert!(time_us < 50000);
}

#[test]
fn bench_l1_penalty_small() {
    // Time L1 penalty for small matrices
    let d = 5;
    let w = common::random_dag(d, 0.5);

    let (_, time_us) = time_operation(|| {
        l1_penalty(&w)
    });

    println!("L1 penalty ({} x {}): {} μs", d, d, time_us);
    assert!(time_us < 100);
}

#[test]
fn bench_l1_penalty_large() {
    // Time L1 penalty for large matrices
    let d = 100;
    let w = common::random_dag(d, 0.2);

    let (_, time_us) = time_operation(|| {
        l1_penalty(&w)
    });

    println!("L1 penalty ({} x {}): {} μs", d, d, time_us);
    assert!(time_us < 10000);
}

#[test]
fn bench_data_generation_small() {
    // Time SEM data generation
    let n = 100;
    let d = 5;
    let w = common::random_dag(d, 0.5);

    let (_, time_us) = time_operation(|| {
        common::data_from_sem(n, d, &w, 0.1)
    });

    println!("Data generation ({} x {}): {} μs", n, d, time_us);
    assert!(time_us < 10000);
}

#[test]
fn bench_data_generation_large() {
    // Time SEM data generation for large dataset
    let n = 10000;
    let d = 20;
    let w = common::random_dag(d, 0.3);

    let (_, time_us) = time_operation(|| {
        common::data_from_sem(n, d, &w, 0.1)
    });

    println!("Data generation ({} x {}): {} μs", n, d, time_us);
}

#[test]
fn bench_random_dag_generation() {
    // Time random DAG generation
    let d = 50;

    let (_, time_us) = time_operation(|| {
        common::random_dag(d, 0.3)
    });

    println!("Random DAG generation ({} x {}): {} μs", d, d, time_us);
    assert!(time_us < 5000);
}

#[test]
fn bench_standardization() {
    // Time data standardization
    let n = 1000;
    let d = 20;
    let data = common::random_data(n, d);

    let (_, time_us) = time_operation(|| {
        common::standardize(&data)
    });

    println!("Standardization ({} x {}): {} μs", n, d, time_us);
    assert!(time_us < 50000);
}

#[test]
fn bench_frobenius_norm() {
    // Time Frobenius norm computation
    let d = 100;
    let w = common::random_dag(d, 0.2);

    let (_, time_us) = time_operation(|| {
        common::frobenius_norm(&w)
    });

    println!("Frobenius norm ({} x {}): {} μs", d, d, time_us);
    assert!(time_us < 1000);
}

#[test]
fn bench_shd_computation() {
    // Time Structural Hamming Distance computation
    let d = 50;
    let g1 = ndarray::Array2::<i32>::zeros((d, d));
    let g2 = ndarray::Array2::<i32>::zeros((d, d));

    let (_, time_us) = time_operation(|| {
        common::structural_hamming_distance(&g1, &g2)
    });

    println!("SHD computation ({} x {}): {} μs", d, d, time_us);
    assert!(time_us < 10000);
}

#[test]
fn bench_total_loss() {
    // Time total loss computation
    let n = 100;
    let d = 10;
    let data = common::random_data(n, d);
    let w = common::random_dag(d, 0.5);
    let config = RegularizationConfig::new(0.1, false).unwrap();

    let (_, time_us) = time_operation(|| {
        notears::scoring::total_loss(&data, &w, &config).unwrap()
    });

    println!("Total loss ({} x {}): {} μs", n, d, time_us);
    assert!(time_us < 20000);
}

#[test]
fn bench_edge_statistics() {
    // Time edge statistics computation
    let d = 50;
    let w = common::random_dag(d, 0.3);

    let (_, time_us) = time_operation(|| {
        common::edge_statistics(&w, 0.3)
    });

    println!("Edge statistics ({} x {}): {} μs", d, d, time_us);
    assert!(time_us < 5000);
}

#[test]
fn bench_is_symmetric() {
    // Time symmetry check
    let d = 100;
    let mut w = common::random_dag(d, 0.2);
    // Make it symmetric
    let wt = w.t().to_owned();
    w = &w + &wt;

    let (_, time_us) = time_operation(|| {
        common::is_symmetric(&w, 1e-10)
    });

    println!("Symmetry check ({} x {}): {} μs", d, d, time_us);
    assert!(time_us < 5000);
}

#[test]
fn bench_lower_triangular_check() {
    // Time lower triangular check
    let d = 100;
    let w = common::random_dag(d, 0.2);

    let (_, time_us) = time_operation(|| {
        common::is_lower_triangular(&w, 1e-10)
    });

    println!("Lower triangular check ({} x {}): {} μs", d, d, time_us);
    assert!(time_us < 5000);
}

// Composite benchmarks

#[test]
fn bench_full_scoring_pipeline() {
    // Time complete scoring computation
    let n = 100;
    let d = 10;
    let data = common::random_data(n, d);
    let w = common::random_dag(d, 0.4);
    let config = RegularizationConfig::new(0.1, false).unwrap();

    let (_, time_us) = time_operation(|| {
        let _mse = mse_loss(&data, &w).unwrap();
        let _l1 = l1_penalty(&w);
        let _h = acyclicity_constraint(&w).unwrap();
        notears::scoring::total_loss(&data, &w, &config).unwrap()
    });

    println!("Full scoring pipeline ({} x {}): {} μs", n, d, time_us);
}

#[test]
fn bench_dimension_scaling_constraint() {
    // Analyze constraint computation scaling with dimension
    for d in [5, 10, 20, 30].iter() {
        let w = common::random_dag(*d, 0.4);
        let (_, time_us) = time_operation(|| {
            acyclicity_constraint(&w).unwrap()
        });
        println!("Constraint d={}: {} μs", d, time_us);
    }
}

#[test]
fn bench_dimension_scaling_mse() {
    // Analyze MSE computation scaling with dimension
    let n = 500;
    for d in [5, 10, 20, 30].iter() {
        let data = common::random_data(n, *d);
        let w = common::random_dag(*d, 0.3);
        let (_, time_us) = time_operation(|| {
            mse_loss(&data, &w).unwrap()
        });
        println!("MSE n={}, d={}: {} μs", n, d, time_us);
    }
}

#[test]
fn bench_sample_scaling_mse() {
    // Analyze MSE computation scaling with sample count
    let d = 10;
    for n in [50, 100, 500, 1000].iter() {
        let data = common::random_data(*n, d);
        let w = common::random_dag(d, 0.3);
        let (_, time_us) = time_operation(|| {
            mse_loss(&data, &w).unwrap()
        });
        println!("MSE n={}, d={}: {} μs", n, d, time_us);
    }
}
