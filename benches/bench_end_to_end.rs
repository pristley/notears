/// Benchmark Suite: End-to-End Optimization
///
/// Benchmarks complete DAG learning runs (full NOTEARS algorithm):
/// - Small problems (10-20 nodes): Quick baselines
/// - Medium problems (30-50 nodes): Typical use case
/// - Larger problems (100+ nodes): Scalability analysis
///
/// Run with: cargo bench --bench bench_end_to_end -- --verbose
/// Create comparison baseline: cargo bench --bench bench_end_to_end -- --save-baseline initial
/// Compare against baseline: cargo bench --bench bench_end_to_end -- --baseline initial
///
/// Expected performance (from NOTEARS paper):
/// | Problem Size   | Bottleneck        | Expected Time | Safety Margin |
/// |----------------|-------------------|---------------|---------------|
/// | d=20, n=1000   | L-BFGS iterations | 1-2 sec       | 3×            |
/// | d=50, n=1000   | Matrix exp        | 5-10 sec      | 5×            |
/// | d=100, n=1000  | Matrix exp        | 30-60 sec     | 10×           |

mod profiling_utils;

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use notears::*;
use std::time::Instant;

/// Small-scale optimization benchmark (quick baseline)
/// Good for regression testing in CI
fn bench_small_scale(c: &mut Criterion) {
    let mut group = c.benchmark_group("small_scale");
    group.sample_size(3);
    group.measurement_time(std::time::Duration::from_secs(30));
    
    // d=10, n=500 - runs in ~100-500ms
    group.bench_function("d10_n500_lambda0p1", |b| {
        let (_w_true, data) = profiling_utils::generate_erdos_renyi_dag(10, 15, 500);
        let standardized = utils::standardize_data(&data).unwrap();
        
        b.iter(|| {
            let opt_config = types::OptimizationConfig::default();
            let reg_config = types::RegularizationConfig::new(0.1, false).unwrap();
            optimization::solve_with_config(black_box(&standardized), opt_config, reg_config)
        });
    });
    
    // d=15, n=500
    group.bench_function("d15_n500_lambda0p1", |b| {
        let (_w_true, data) = profiling_utils::generate_erdos_renyi_dag(15, 25, 500);
        let standardized = utils::standardize_data(&data).unwrap();
        
        b.iter(|| {
            let opt_config = types::OptimizationConfig::default();
            let reg_config = types::RegularizationConfig::new(0.1, false).unwrap();
            optimization::solve_with_config(black_box(&standardized), opt_config, reg_config)
        });
    });
    
    group.finish();
}

/// Medium-scale optimization benchmark (typical use case)
fn bench_medium_scale(c: &mut Criterion) {
    let mut group = c.benchmark_group("medium_scale");
    group.sample_size(2);
    group.measurement_time(std::time::Duration::from_secs(60));
    
    // d=20, n=1000 - runs in ~1-2 seconds
    group.bench_function("d20_n1000_lambda0p1", |b| {
        let (_w_true, data) = profiling_utils::generate_erdos_renyi_dag(20, 30, 1000);
        let standardized = utils::standardize_data(&data).unwrap();
        
        b.iter(|| {
            let opt_config = types::OptimizationConfig::default();
            let reg_config = types::RegularizationConfig::new(0.1, false).unwrap();
            optimization::solve_with_config(black_box(&standardized), opt_config, reg_config)
        });
    });
    
    // d=30, n=1000 - runs in ~3-5 seconds
    group.bench_function("d30_n1000_lambda0p1", |b| {
        let (_w_true, data) = profiling_utils::generate_erdos_renyi_dag(30, 45, 1000);
        let standardized = utils::standardize_data(&data).unwrap();
        
        b.iter(|| {
            let opt_config = types::OptimizationConfig::default();
            let reg_config = types::RegularizationConfig::new(0.1, false).unwrap();
            optimization::solve_with_config(black_box(&standardized), opt_config, reg_config)
        });
    });
    
    group.finish();
}

/// Scaling study: impact of number of dimensions
fn bench_dimension_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("dimension_scaling");
    group.sample_size(1);
    group.measurement_time(std::time::Duration::from_secs(120));
    
    let dimensions = vec![10, 20, 30, 40, 50];
    let n_samples = 1000;
    let lambda = 0.1;
    
    for d in dimensions {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("d{}n{}", d, n_samples)),
            &d,
            |b, &d| {
                let n_edges = (d as f64 * 1.5) as usize;
                let (_w_true, data) = profiling_utils::generate_erdos_renyi_dag(d, n_edges, n_samples);
                let standardized = utils::standardize_data(&data).unwrap();
                
                b.iter(|| {
                    let opt_config = types::OptimizationConfig::default();
                    let reg_config = types::RegularizationConfig::new(lambda, false).unwrap();
                    optimization::solve_with_config(black_box(&standardized), opt_config, reg_config)
                });
            },
        );
    }
    
    group.finish();
}

/// Scaling study: impact of number of samples
fn bench_sample_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("sample_scaling");
    group.sample_size(2);
    group.measurement_time(std::time::Duration::from_secs(60));
    
    let d = 20;
    let sample_sizes = vec![100, 500, 1000, 2000];
    let lambda = 0.1;
    
    for n in sample_sizes {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("d{}n{}", d, n)),
            &n,
            |b, &n| {
                let n_edges = 30;
                let (_w_true, data) = profiling_utils::generate_erdos_renyi_dag(d, n_edges, n);
                let standardized = utils::standardize_data(&data).unwrap();
                
                b.iter(|| {
                    let opt_config = types::OptimizationConfig::default();
                    let reg_config = types::RegularizationConfig::new(lambda, false).unwrap();
                    optimization::solve_with_config(black_box(&standardized), opt_config, reg_config)
                });
            },
        );
    }
    
    group.finish();
}

/// Impact of regularization strength on convergence speed
fn bench_lambda_sensitivity(c: &mut Criterion) {
    let mut group = c.benchmark_group("lambda_sensitivity");
    group.sample_size(2);
    group.measurement_time(std::time::Duration::from_secs(60));
    
    let d = 20;
    let n = 1000;
    let lambda_values = vec![0.01, 0.05, 0.1, 0.3, 0.5];
    
    for lambda in lambda_values {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("d{}n{}l{}", d, n, lambda)),
            &lambda,
            |b, &lambda| {
                let n_edges = 30;
                let (_w_true, data) = profiling_utils::generate_erdos_renyi_dag(d, n_edges, n);
                let standardized = utils::standardize_data(&data).unwrap();
                
                b.iter(|| {
                    let opt_config = types::OptimizationConfig::default();
                    let reg_config = types::RegularizationConfig::new(lambda, false).unwrap();
                    optimization::solve_with_config(black_box(&standardized), opt_config, reg_config)
                });
            },
        );
    }
    
    group.finish();
}

/// Profiling helper: run solve with instrumentation
/// Prints out timing breakdown for analysis
fn profile_optimization_verbose(d: usize, n: usize, lambda: f64) {
    println!("\n=== Profiling NOTEARS Optimization ===");
    println!("Problem: d={}, n={}, lambda={}", d, n, lambda);
    
    let (_w_true, data) = profiling_utils::generate_erdos_renyi_dag(d, (d as f64 * 1.5) as usize, n);
    
    let start_total = Instant::now();
    let standardized = utils::standardize_data(&data).unwrap();
    let standardize_time = start_total.elapsed().as_secs_f64();
    println!("Standardization: {:.3}s", standardize_time);
    
    let start_solve = Instant::now();
    let opt_config = types::OptimizationConfig::default();
    let reg_config = types::RegularizationConfig::new(lambda, false).unwrap();
    let result = optimization::solve_with_config(&standardized, opt_config, reg_config).unwrap();
    let solve_time = start_solve.elapsed().as_secs_f64();
    
    let total_time = start_total.elapsed().as_secs_f64();
    
    println!("\n=== Optimization Results ===");
    println!("Total time: {:.3}s", total_time);
    println!("Solve time: {:.3}s", solve_time);
    println!("Iterations: {}", result.iterations);
    println!("Constraint violation: {:.6e}", result.constraint_violation);
    println!("Final score: {:.6e}", result.final_score);
    println!("Time per iteration: {:.3}s", solve_time / result.iterations as f64);
}

/// Verbose profiling benchmark (for manual analysis)
fn bench_profiling_verbose(c: &mut Criterion) {
    // Create a black-box benchmark that runs profiling
    // This doesn't contribute to criterion stats but helps manual analysis
    c.bench_function("profiling_d20_n1000", |_b| {
        profile_optimization_verbose(20, 1000, 0.1);
    });
}

criterion_group!(
    end_to_end,
    bench_small_scale,
    bench_medium_scale,
    bench_dimension_scaling,
    bench_sample_scaling,
    bench_lambda_sensitivity,
    bench_profiling_verbose
);

criterion_main!(end_to_end);
