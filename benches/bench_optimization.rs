/// Benchmark Suite: Optimization Operations
///
/// Benchmarks intermediate-level components used during optimization:
/// - Loss computation (scoring functions)
/// - Gradient computations
/// - Individual optimization iterations
///
/// Run with: cargo bench --bench bench_optimization -- --verbose
///
/// Expected baseline performance:
/// - Full iteration (d=20, n=1000): ~50-100ms
/// - Full iteration (d=50, n=1000): ~500-1000ms
///
/// Note: These are individual operation benchmarks, not full solves

mod profiling_utils;

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use notears::*;

/// Benchmark a single optimization iteration (without outer loop)
/// This measures: gradient computation + loss + constraint check
fn bench_optimization_inner_iteration(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimization_iteration");
    group.sample_size(10); // Small sample size for expensive operations
    
    // Test different problem sizes
    let test_cases = vec![
        (10, 500),   // 10 nodes, 500 samples
        (20, 1000),  // 20 nodes, 1000 samples
        (30, 1000),  // 30 nodes, 1000 samples
    ];
    
    for (d, n) in test_cases {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("d{}n{}", d, n)),
            &(d, n),
            |b, &(d, n)| {
                let data = profiling_utils::random_data(n, d);
                let standardized = utils::standardize_data(&data).unwrap();
                let w = profiling_utils::random_dag(d, 0.2);
                
                b.iter(|| {
                    // Simulate one iteration:
                    // 1. Compute loss
                    let loss = scoring::mse_loss(black_box(&standardized), black_box(&w)).unwrap();
                    
                    // 2. Compute constraint
                    let h = acyclicity::acyclicity_constraint(black_box(&w)).unwrap();
                    
                    // 3. Compute gradient of constraint
                    let grad_h = acyclicity::acyclicity_gradient(black_box(&w)).unwrap();
                    
                    // 4. L1 penalty
                    let l1 = scoring::l1_penalty(black_box(&w));
                    
                    (loss, h, grad_h, l1)
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark loss + gradient computation (used in L-BFGS)
fn bench_loss_gradient_pair(c: &mut Criterion) {
    let mut group = c.benchmark_group("loss_gradient_pair");
    group.sample_size(15);
    
    let test_cases = vec![
        (10, 500),
        (20, 1000),
        (30, 1000),
        (50, 1000),
    ];
    
    for (d, n) in test_cases {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("d{}n{}", d, n)),
            &(d, n),
            |b, &(d, n)| {
                let data = profiling_utils::random_data(n, d);
                let standardized = utils::standardize_data(&data).unwrap();
                let w = profiling_utils::random_dag(d, 0.2);
                
                b.iter(|| {
                    let loss = scoring::mse_loss(black_box(&standardized), black_box(&w)).unwrap();
                    let constraint_grad = acyclicity::acyclicity_gradient(black_box(&w)).unwrap();
                    (loss, constraint_grad)
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark acyclicity penalty computation at various ρ values
/// This simulates the augmented Lagrangian term: (ρ/2) * h(W)²
fn bench_augmented_lagrangian_penalty(c: &mut Criterion) {
    let mut group = c.benchmark_group("augmented_lagrangian_penalty");
    group.sample_size(30);
    
    let dimensions = vec![10, 20, 30, 50];
    let rho_values = vec![1.0, 10.0, 100.0];
    
    for d in dimensions {
        for rho in &rho_values {
            group.bench_with_input(
                BenchmarkId::from_parameter(format!("d{}rho{}", d, rho)),
                &(d, *rho),
                |b, &(d, rho)| {
                    let w = profiling_utils::random_dag(d, 0.3);
                    
                    b.iter(|| {
                        let h = acyclicity::acyclicity_constraint(black_box(&w)).unwrap();
                        let penalty = (rho / 2.0) * h * h;
                        penalty
                    });
                },
            );
        }
    }
    
    group.finish();
}

/// Benchmark weighted loss composition
/// This is what gets computed in each L-BFGS iteration
fn bench_composed_loss(c: &mut Criterion) {
    let mut group = c.benchmark_group("composed_loss");
    group.sample_size(15);
    
    let test_cases = vec![
        (10, 500, 0.1),   // lambda=0.1 (weak sparsity)
        (20, 1000, 0.1),
        (20, 1000, 0.5),  // lambda=0.5 (medium sparsity)
        (30, 1000, 0.3),  // lambda=0.3 (balanced)
    ];
    
    for (d, n, lambda) in test_cases {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("d{}n{}l{}", d, n, lambda)),
            &(d, n, lambda),
            |b, &(d, n, lambda)| {
                let data = profiling_utils::random_data(n, d);
                let standardized = utils::standardize_data(&data).unwrap();
                let w = profiling_utils::random_dag(d, 0.2);
                
                b.iter(|| {
                    // F(W) = MSE(W) + λ*L1(W)
                    let mse = scoring::mse_loss(black_box(&standardized), black_box(&w)).unwrap();
                    let l1 = scoring::l1_penalty(black_box(&w));
                    let total_loss = mse + lambda * l1;
                    total_loss
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark constraint relaxation sequence
/// Simulates increasing ρ in the augmented Lagrangian (key hyperparameter)
fn bench_constraint_progression(c: &mut Criterion) {
    let mut group = c.benchmark_group("constraint_progression");
    group.sample_size(20);
    
    let d = 20;
    let rho_sequence = vec![1.0, 10.0, 100.0, 1000.0];
    
    group.bench_function("rho_progression_d20", |b| {
        let w = profiling_utils::random_dag(d, 0.3);
        
        b.iter(|| {
            let mut penalties = Vec::new();
            
            for rho in &rho_sequence {
                let h = acyclicity::acyclicity_constraint(black_box(&w)).unwrap();
                let penalty = (rho / 2.0) * h * h;
                penalties.push(penalty);
            }
            
            penalties
        });
    });
    
    group.finish();
}

criterion_group!(
    optimization_ops,
    bench_optimization_inner_iteration,
    bench_loss_gradient_pair,
    bench_augmented_lagrangian_penalty,
    bench_composed_loss,
    bench_constraint_progression
);

criterion_main!(optimization_ops);
