/// Benchmark Suite: Low-Level Matrix Operations
///
/// Benchmarks individual matrix operation performance:
/// - Matrix exponential (largest bottleneck)
/// - Acyclicity constraint computation
/// - Acyclicity gradient computation
///
/// Run with: cargo bench --bench bench_matrix_ops -- --verbose
///
/// Expected baseline performance (reference CPU):
/// - matrix_exp_10x10: ~0.1ms
/// - matrix_exp_50x50: ~5-10ms
/// - matrix_exp_100x100: ~50-100ms
/// - acyclicity_constraint_20nodes: ~1-2ms
/// - acyclicity_gradient_20nodes: ~2-3ms
mod profiling_utils;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::Array2;
use notears::*;
use rand::Rng;

/// Benchmark matrix exponential at various sizes
/// This is the primary bottleneck in the algorithm (O(d³·log(d)))
fn bench_matrix_exponential(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_exponential");
    group.sample_size(20); // Reduce samples for expensive operations

    let dimensions = vec![5, 10, 20, 30, 50];

    for dim in dimensions {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}x{}", dim, dim)),
            &dim,
            |b, &dim| {
                // Generate small random matrix
                let mut rng = rand::thread_rng();
                let mut m = Array2::zeros((dim, dim));
                for i in 0..dim {
                    for j in 0..dim {
                        m[[i, j]] = rng.gen_range(-0.5..0.5);
                    }
                }

                b.iter(|| utils::matrix_exponential(black_box(&m)));
            },
        );
    }

    group.finish();
}

/// Benchmark acyclicity constraint computation
/// This depends on matrix exponential, so performance scales similarly
fn bench_acyclicity_constraint(c: &mut Criterion) {
    let mut group = c.benchmark_group("acyclicity_constraint");
    group.sample_size(30);

    let dimensions = vec![5, 10, 15, 20, 30, 50];

    for dim in dimensions {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}nodes", dim)),
            &dim,
            |b, &dim| {
                let w = profiling_utils::random_dag(dim, 0.3);

                b.iter(|| acyclicity::acyclicity_constraint(black_box(&w)));
            },
        );
    }

    group.finish();
}

/// Benchmark acyclicity gradient computation
/// Requires matrix exponential + gradient of element-wise operations
fn bench_acyclicity_gradient(c: &mut Criterion) {
    let mut group = c.benchmark_group("acyclicity_gradient");
    group.sample_size(20);

    let dimensions = vec![5, 10, 15, 20, 30];

    for dim in dimensions {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}nodes", dim)),
            &dim,
            |b, &dim| {
                let w = profiling_utils::random_dag(dim, 0.3);

                b.iter(|| acyclicity::acyclicity_gradient(black_box(&w)));
            },
        );
    }

    group.finish();
}

/// Benchmark MSE loss computation (data fidelity term)
/// Scales as O(n·d²) where n = samples, d = dimensions
fn bench_mse_loss(c: &mut Criterion) {
    // Test varying samples with fixed d=20
    {
        let mut group = c.benchmark_group("mse_loss_samples_fixed_d20");
        let d = 20;

        for n in [100, 500, 1000, 5000].iter() {
            group.bench_with_input(
                BenchmarkId::from_parameter(format!("{}samples", n)),
                n,
                |b, &n| {
                    let data = profiling_utils::random_data(n, d);
                    let w = profiling_utils::random_dag(d, 0.2);

                    b.iter(|| scoring::mse_loss(black_box(&data), black_box(&w)));
                },
            );
        }
        group.finish();
    }

    // Test varying dimensions with fixed n=1000
    {
        let mut group = c.benchmark_group("mse_loss_dims_fixed_n1000");
        let n = 1000;

        for d in [5, 10, 20, 30, 50].iter() {
            group.bench_with_input(
                BenchmarkId::from_parameter(format!("{}dims", d)),
                d,
                |b, &d| {
                    let data = profiling_utils::random_data(n, d);
                    let w = profiling_utils::random_dag(d, 0.2);

                    b.iter(|| scoring::mse_loss(black_box(&data), black_box(&w)));
                },
            );
        }
        group.finish();
    }
}

/// Benchmark L1 penalty (sparsity regularization)
/// Scales as O(d²) - very cheap operation
fn bench_l1_penalty(c: &mut Criterion) {
    let mut group = c.benchmark_group("l1_penalty");

    let dimensions = vec![10, 50, 100, 200, 500];

    for dim in dimensions {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}x{}", dim, dim)),
            &dim,
            |b, &dim| {
                let w = profiling_utils::random_dag(dim, 0.5);

                b.iter(|| scoring::l1_penalty(black_box(&w)));
            },
        );
    }

    group.finish();
}

/// Benchmark data standardization (preprocessing)
/// Scales as O(n·d)
fn bench_standardize_data(c: &mut Criterion) {
    // Test varying sample sizes
    {
        let mut group = c.benchmark_group("standardize_data_samples");
        let d = 20;

        for n in [100, 500, 1000, 5000].iter() {
            group.bench_with_input(
                BenchmarkId::from_parameter(format!("{}samples", n)),
                n,
                |b, &n| {
                    let data = profiling_utils::random_data(n, d);

                    b.iter(|| utils::standardize_data(black_box(&data)));
                },
            );
        }
        group.finish();
    }

    // Test varying dimensions
    {
        let mut group = c.benchmark_group("standardize_data_dimensions");
        let n = 1000;

        for d in [5, 10, 20, 50, 100].iter() {
            group.bench_with_input(
                BenchmarkId::from_parameter(format!("{}dims", d)),
                d,
                |b, &d| {
                    let data = profiling_utils::random_data(n, d);

                    b.iter(|| utils::standardize_data(black_box(&data)));
                },
            );
        }
        group.finish();
    }
}

criterion_group!(
    matrix_ops,
    bench_matrix_exponential,
    bench_acyclicity_constraint,
    bench_acyclicity_gradient,
    bench_mse_loss,
    bench_l1_penalty,
    bench_standardize_data
);

criterion_main!(matrix_ops);
