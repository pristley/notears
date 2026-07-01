/// Criterion benchmarks for NOTEARS algorithm
///
/// Run with: cargo bench --release
/// HTML reports in: target/criterion/
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::Array2;
use notears::*;
use rand::Rng;

/// Generate synthetic data with known DAG structure
fn generate_synthetic_dag(n_samples: usize, n_nodes: usize, _density: f64) -> Array2<f64> {
    let mut rng = rand::thread_rng();
    let mut data = Array2::zeros((n_samples, n_nodes));

    // Generate random lower triangular weight matrix (ensures acyclicity)
    for i in 0..n_samples {
        for j in 0..n_nodes {
            data[[i, j]] = rng.gen::<f64>() * 2.0 - 1.0;
        }
    }

    data
}

fn benchmark_acyclicity_constraint(c: &mut Criterion) {
    let mut group = c.benchmark_group("acyclicity_constraint");

    for dim in [5, 10, 20, 50].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(dim), dim, |b, &dim| {
            let w = Array2::zeros((dim, dim));
            b.iter(|| acyclicity::acyclicity_constraint(black_box(&w)));
        });
    }

    group.finish();
}

fn benchmark_acyclicity_gradient(c: &mut Criterion) {
    let mut group = c.benchmark_group("acyclicity_gradient");

    for dim in [5, 10, 20].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(dim), dim, |b, &dim| {
            let w = Array2::zeros((dim, dim));
            b.iter(|| acyclicity::acyclicity_gradient(black_box(&w)));
        });
    }

    group.finish();
}

fn benchmark_matrix_exponential(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_exponential");

    for dim in [5, 10, 20, 50].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(dim), dim, |b, &dim| {
            let w = Array2::zeros((dim, dim));
            b.iter(|| utils::matrix_exponential(black_box(&w)));
        });
    }

    group.finish();
}

fn benchmark_mse_loss(c: &mut Criterion) {
    let mut group = c.benchmark_group("mse_loss");

    for n_samples in [100, 500, 1000].iter() {
        let dim = 20;
        group.bench_with_input(
            BenchmarkId::new("samples", n_samples),
            n_samples,
            |b, &n_samples| {
                let data = generate_synthetic_dag(n_samples, dim, 0.1);
                let w = Array2::zeros((dim, dim));
                b.iter(|| scoring::mse_loss(black_box(&data), black_box(&w)));
            },
        );
    }

    group.finish();
}

fn benchmark_standardize_data(c: &mut Criterion) {
    let mut group = c.benchmark_group("standardize_data");

    for n_samples in [100, 500, 1000].iter() {
        let dim = 20;
        group.bench_with_input(
            BenchmarkId::new("samples", n_samples),
            n_samples,
            |b, &n_samples| {
                let data = generate_synthetic_dag(n_samples, dim, 0.1);
                b.iter(|| utils::standardize_data(black_box(&data)));
            },
        );
    }

    group.finish();
}

fn benchmark_full_solve(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_solve");
    group.sample_size(10); // Reduce samples for long-running tests

    let dims = [5, 10];
    for dim in dims.iter() {
        group.bench_with_input(BenchmarkId::from_parameter(dim), dim, |b, &dim| {
            let data = generate_synthetic_dag(200, dim, 0.1);
            let std_data = utils::standardize_data(&data).unwrap();
            let config = OptimizationConfig::new(100, 50, 10, 1e-6, 1.0, 0.25, 0.3).unwrap();
            let loss_config = RegularizationConfig::new(0.1, false).unwrap();

            b.iter(|| {
                optimization::solve_with_config(
                    black_box(&std_data),
                    black_box(config.clone()),
                    black_box(loss_config.clone()),
                )
            });
        });
    }

    group.finish();
}

fn benchmark_gradient_computation(c: &mut Criterion) {
    let mut group = c.benchmark_group("gradient_computation");

    for dim in [10, 20, 50].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(dim), dim, |b, &dim| {
            let data = generate_synthetic_dag(100, dim, 0.1);
            let w = Array2::zeros((dim, dim));
            let config = RegularizationConfig::default();

            b.iter(|| {
                scoring::total_loss_gradient(black_box(&data), black_box(&w), black_box(&config))
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_acyclicity_constraint,
    benchmark_acyclicity_gradient,
    benchmark_matrix_exponential,
    benchmark_mse_loss,
    benchmark_standardize_data,
    benchmark_gradient_computation,
    benchmark_full_solve
);
criterion_main!(benches);
