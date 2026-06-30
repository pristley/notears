# Deployment & Best Practices Summary

Comprehensive guide for deploying and managing NOTEARS in production environments.

## 📚 Documentation Structure

### Quick Start & Fundamentals
- **[README.md](README.md)** - Installation, quick examples, performance targets
  - Minimal example
  - Advanced configuration example
  - Performance benchmarks
  - CI/CD status badges

### In-Depth Guides
- **[API Reference](docs/API.md)** - Complete type and function documentation
  - Type definitions (WeightMatrix, DataMatrix, OptimizationResult)
  - Function signatures and examples
  - Error types and handling
  - Common patterns and best practices

- **[Configuration Guide](docs/CONFIGURATION.md)** - Tuning by data regime
  - Three optimization regimes (underdetermined, balanced, overdetermined)
  - Lambda selection strategies (grid search, cross-validation, BIC)
  - Hyperparameter tuning (ρ, tolerance, progress_rate)
  - Reproducibility checklist

- **[Troubleshooting Guide](TROUBLESHOOTING.md)** - Common issues and solutions
  - Installation and compilation issues
  - Data quality problems
  - Convergence failures
  - Numerical stability issues
  - Performance optimization
  - FAQ section

### Tutorial Notebooks
- **[01_quick_start.ipynb](examples/01_quick_start.ipynb)** - Basic workflow
  - Data generation
  - Standardization
  - Structure learning
  - Result evaluation

- **[02_configuration_best_practices.ipynb](examples/02_configuration_best_practices.ipynb)** - Configuration by regime
  - Regime-specific configurations
  - Lambda selection strategies
  - Reproducibility practices
  - Quick reference table

### Performance & Benchmarking
- **[BENCHMARKING.md](BENCHMARKING.md)** - Performance profiling suite
  - Three-tier benchmark structure
  - Flame graph profiling
  - Memory profiling (heaptrack)
  - Baseline regression testing
  - CI integration

## 🔧 Configuration Management

### Three Optimization Regimes

#### 1. Underdetermined (n < d)
```rust
OptimizationConfig {
    max_outer_iterations: 20,
    max_lbfgs_iterations: 200,
    lbfgs_memory: 10,
    constraint_tolerance: 1e-7,
    penalty_rho_init: 1.0,      // Higher penalty
    progress_rate: 0.25,
    edge_threshold: 0.3,
}
```
- Use λ = 0.1-0.5 (strong regularization)
- Higher ρ_init for faster acyclicity enforcement
- More L-BFGS iterations for inner optimization

#### 2. Overdetermined (n >> d)
```rust
OptimizationConfig {
    max_outer_iterations: 15,
    max_lbfgs_iterations: 100,
    lbfgs_memory: 20,
    constraint_tolerance: 1e-8,
    penalty_rho_init: 0.1,      // Lower penalty
    progress_rate: 0.1,         // Stricter progress
    edge_threshold: 0.3,
}
```
- Use λ = 0.01-0.1 (weak regularization)
- Lower ρ_init for careful fine-tuning
- Stricter progress criterion (0.1 vs 0.25)

#### 3. Balanced (default)
```rust
OptimizationConfig::default()
```
- Use λ = 0.05-0.2 (moderate regularization)
- Standard settings work well
- Use when n ≈ 2-5 × d

## 📋 Version Management

### Semantic Versioning
- **MAJOR**: Incompatible API changes
- **MINOR**: Backwards-compatible features
- **PATCH**: Backwards-compatible bug fixes

### Current Version
- **Release**: 0.1.0 (2026-06-30)
- **MSRV**: Rust 1.56+
- **Status**: Production-ready

### Compatibility
- ✅ Stable Rust channel
- ✅ Beta channel
- ✅ Nightly channel (tested regularly)
- ✅ Platform-independent (pure Rust)

## 🔄 CI/CD Pipeline

### GitHub Actions Workflows

#### 1. Tests (`.github/workflows/tests.yml`)
- **Triggers**: Push to main, pull requests
- **Matrix**: Stable, Beta, Nightly Rust
- **Checks**:
  - Full test suite
  - Rustfmt (formatting)
  - Clippy (linting)
  - MSRV verification (1.56)

#### 2. Benchmarks (`.github/workflows/benchmarks.yml`)
- **Triggers**: Push to main
- **Suite**:
  - Matrix operations benchmarks
  - Optimization operation benchmarks
  - End-to-end benchmarks
  - Performance regression detection
  - Flamegraph profiling (on main)

#### 3. Documentation (`.github/workflows/docs.yml`)
- **Triggers**: Push to main
- **Features**:
  - Build rustdoc
  - Test documentation examples
  - Deploy to GitHub Pages
  - Integration with docs.rs

### Local Testing

```bash
# Quick tests
cargo test

# All tests with logging
RUST_LOG=debug cargo test -- --nocapture

# Benchmarks
cargo bench --bench bench_end_to_end

# Create baseline
cargo bench -- --save-baseline initial

# Compare against baseline
cargo bench -- --baseline initial
```

## 🚀 Deployment Checklist

- [ ] **Version Management**
  - [ ] Update version in Cargo.toml
  - [ ] Update CHANGELOG.md
  - [ ] Create git tag
  - [ ] Publish to crates.io: `cargo publish`

- [ ] **Testing**
  - [ ] Run full test suite: `cargo test`
  - [ ] Check formatting: `cargo fmt -- --check`
  - [ ] Run linter: `cargo clippy -- -D warnings`
  - [ ] Verify MSRV: `cargo +1.56 build`
  - [ ] Run benchmarks: `cargo bench --release`

- [ ] **Documentation**
  - [ ] Update README.md
  - [ ] Update CHANGELOG.md
  - [ ] Verify all doc examples compile: `cargo test --doc`
  - [ ] Build full docs: `cargo doc --no-deps --open`
  - [ ] Check docs.rs rendering

- [ ] **Performance**
  - [ ] Create performance baseline
  - [ ] Compare against previous release
  - [ ] Profile hotspots if needed
  - [ ] Document performance targets

- [ ] **Quality Assurance**
  - [ ] Code review
  - [ ] Integration testing
  - [ ] Real-world dataset testing
  - [ ] Backward compatibility check

## 🐛 Common Deployment Issues

### Issue: Build fails on CI but works locally

**Solution:**
- Ensure Rust version matches: `rustc --version`
- Check for platform-specific code
- Verify all dependencies are specified
- Run with `--verbose` for details

### Issue: Performance degrades in production

**Solution:**
- Ensure release build: `cargo build --release`
- Profile with flamegraph: `cargo flamegraph --bench`
- Check memory usage: `heaptrack`
- Verify data preprocessing (standardization)

### Issue: Results not reproducible

**Solution:**
- Document full configuration (see Configuration Guide)
- Fix random seed if needed
- Log all hyperparameters
- Version control data preprocessing steps

## 📊 Production Integration Examples

### Example 1: Data Pipeline Integration
```rust
use notears::optimization::solve;
use notears::utils::standardize_data;

fn learn_structure_from_pipeline(raw_data: &Array2<f64>) 
    -> Result<DAGStructure, Box<dyn std::error::Error>> 
{
    // 1. Preprocess
    let preprocessed = preprocess_data(raw_data)?;
    let standardized = standardize_data(&preprocessed)?;
    
    // 2. Solve
    let result = solve(&standardized, 0.1)?;
    
    // 3. Validate
    if !result.is_acyclic() {
        return Err("Not a valid DAG".into());
    }
    
    // 4. Extract structure
    let structure = DAGStructure::from_result(&result)?;
    Ok(structure)
}
```

### Example 2: Multi-lambda Evaluation
```rust
fn evaluate_across_lambdas(data: &Array2<f64>) 
    -> Result<Vec<(f64, OptimizationResult)>, Box<dyn std::error::Error>> 
{
    let standardized = standardize_data(data)?;
    let lambdas = [0.01, 0.05, 0.1, 0.2, 0.5];
    
    let mut results = Vec::new();
    for lambda in &lambdas {
        let result = solve(&standardized, lambda)?;
        results.push((*lambda, result));
    }
    
    Ok(results)
}
```

### Example 3: Ensemble Learning
```rust
fn ensemble_dag_learning(data: &Array2<f64>, n_runs: usize) 
    -> Result<EnsembleResult, Box<dyn std::error::Error>> 
{
    let standardized = standardize_data(data)?;
    let mut all_results = Vec::new();
    
    for _ in 0..n_runs {
        let result = solve(&standardized, 0.1)?;
        all_results.push(result);
    }
    
    // Combine results via voting or averaging
    let ensemble = combine_results(all_results)?;
    Ok(ensemble)
}
```

## 📞 Support & Resources

- **GitHub Issues**: https://github.com/pristley/notears/issues
- **Documentation**: https://docs.rs/notears
- **Benchmarks**: See BENCHMARKING.md for performance profiling
- **Examples**: Tutorial notebooks in examples/

## 🔗 Related Resources

- **Original Paper**: [DAGs with NO TEARS](https://arxiv.org/abs/1803.01422)
- **Matrix Exponential**: [Higham (2008)](https://arxiv.org/abs/0804.4150)
- **Augmented Lagrangian**: [Boyd & Parikh (2011)](https://web.stanford.edu/~boyd/admm_book.html)
- **Rust Best Practices**: [Rust Book](https://doc.rust-lang.org/book/)

## License

Licensed under MIT License — see LICENSE file for details.
