# NOTEARS: Rust Implementation

[![Crates.io](https://img.shields.io/crates/v/notears.svg)](https://crates.io/crates/notears)
[![Docs.rs](https://docs.rs/notears/badge.svg)](https://docs.rs/notears)
[![CI/CD](https://github.com/pristley/notears/workflows/CI/badge.svg)](https://github.com/pristley/notears/actions)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

Production-grade Rust implementation of the **NO TEARS** algorithm for learning directed acyclic graph (DAG) structures from observational data using continuous optimization.

**Paper:** Zheng et al. (2018) — [DAGs with NO TEARS](https://arxiv.org/abs/1803.01422)

## Overview

NOTEARS learns sparse DAG structures by solving:

$$\min_{W} \frac{1}{2n}\|X - XW\|_F^2 + \lambda\|W\|_1 \quad \text{subject to} \quad h(W) = \text{tr}(e^{W \odot W}) - d = 0$$

where:
- $W$ is the $d \times d$ weight matrix defining the DAG
- $h(W)$ is the differentiable acyclicity constraint
- $\odot$ denotes element-wise product (Hadamard)
- $e^{(\cdot)}$ is the matrix exponential

### Key Features

✅ **Differentiable Acyclicity Constraint** — Enable gradient-based optimization  
✅ **O(d³) per-iteration Complexity** — Via efficient matrix exponential  
✅ **L-BFGS + Augmented Lagrangian** — State-of-the-art constrained optimization  
✅ **Production-Grade Error Handling** — Comprehensive validation & descriptive errors  
✅ **Numerical Stability** — Across varying data regimes (n, d, λ)  
✅ **Comprehensive Benchmarks** — Performance profiling suite included  

## Quick Start

### Installation

Add to your `Cargo.toml`:
```toml
[dependencies]
notears = "0.1"
ndarray = "0.15"
```

### Minimal Example

```rust
use notears::optimization::solve;
use notears::utils::standardize_data;
use ndarray::Array2;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load your observational data (n samples × d variables)
    let data = Array2::zeros((1000, 20));
    
    // Standardize (recommended)
    let standardized = standardize_data(&data)?;
    
    // Learn DAG structure (default config, λ=0.1)
    let result = solve(&standardized, 0.1)?;
    
    // Extract learned structure
    let w_estimated = result.weight_matrix;
    let edges = result.edges();
    let acyclicity = result.constraint_violation;
    
    println!("Learned {} edges", edges.len());
    println!("Constraint violation: {:.2e}", acyclicity);
    
    Ok(())
}
```

### Advanced Usage with Custom Configuration

```rust
use notears::types::{OptimizationConfig, RegularizationConfig};
use notears::optimization::solve_with_config;
use ndarray::Array2;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data = Array2::zeros((1000, 50));
    let standardized = notears::utils::standardize_data(&data)?;
    
    // Custom configuration for large-scale problems
    let opt_config = OptimizationConfig {
        max_outer_iterations: 20,
        max_lbfgs_iterations: 200,
        lbfgs_memory: 15,
        constraint_tolerance: 1e-8,
        penalty_rho_init: 0.1,
        progress_rate: 0.1,
        edge_threshold: 0.3,
    };
    
    let reg_config = RegularizationConfig::new(0.05, false)?;
    let result = solve_with_config(&standardized, opt_config, reg_config)?;
    
    Ok(())
}
```

## Documentation

### 📚 Comprehensive Documentation Suite (~29,000 words)

**Start here:** [NOTEARS Documentation Master Index](docs/NOTEARS_Documentation_Master_Index.md) — Navigation guide with 5 reading paths by role

#### For Different Audiences:

- **[🚀 Quick Reference Guide](docs/NOTEARS_Quick_Reference_Guide.md)** — Practical cheat sheet
  - Algorithm comparison, hyperparameter tuning, troubleshooting (5 common issues), validation checklist, 10 pitfalls
  - *Best for: Practitioners needing fast answers*

- **[🛠️ Rust Implementation Guide](docs/NOTEARS_Rust_Implementation_Guide.md)** — Complete technical reference
  - 7-phase implementation roadmap, mathematical foundations, code examples, production checklist
  - *Best for: Software engineers implementing NOTEARS*

- **[📊 Algorithm Analysis & Comparison](docs/NOTEARS_Algorithm_Analysis_and_Comparison.md)** — Deep dive for researchers
  - Detailed comparison vs. PC/GES/LiNGAM/GOBNILP, 8-dimensional evaluation rubric, 4 real-world case studies
  - *Best for: Data scientists and researchers*

#### Technical References:

- **[API Reference](docs/API.md)** — Complete type and function documentation with examples
- **[Configuration Guide](docs/CONFIGURATION.md)** — Tuning for different data regimes (underdetermined, balanced, overdetermined)
- **[Troubleshooting Guide](TROUBLESHOOTING.md)** — Common issues, diagnostics, and solutions (20+ topics)
- **[Benchmarking Suite](BENCHMARKING.md)** — Performance profiling, flamegraph, regression testing
- **[Deployment Guide](DEPLOYMENT.md)** — Production setup, version management, CI/CD
- **[Tutorial Notebooks](examples/)** — Jupyter notebooks with worked examples and best practices

## Examples

### Example 1: Synthetic Data with Known Structure
```rust
// Generate data from known DAG
let w_true = create_dag(10, 0.3);  // 10 nodes, 30% density
let data = sample_from_dag(&w_true, 1000)?;

// Learn structure
let result = solve(&data, 0.1)?;

// Evaluate: compare learned structure to ground truth
let accuracy = evaluate_structure(&w_true, &result.weight_matrix);
println!("Structure accuracy: {:.2%}", accuracy);
```

### Example 2: Real-World Data Analysis
```rust
// Load real data (e.g., from CSV)
let data = load_data_from_file("data.csv")?;

// Standardize
let standardized = standardize_data(&data)?;

// Learn with lambda selection via cross-validation
let lambda = select_lambda(&standardized)?;
let result = solve(&standardized, lambda)?;

// Visualize DAG
visualize_dag(&result.weight_matrix, "learned_dag.svg")?;
```

### Example 3: Sensitivity Analysis
```rust
let data = Array2::zeros((1000, 20));
let standardized = standardize_data(&data)?;

// Vary regularization strength
for lambda in [0.01, 0.05, 0.1, 0.2, 0.5] {
    let result = solve(&standardized, lambda)?;
    println!("λ={}: {} edges", lambda, result.edges().len());
}
```

## Performance Targets

| Problem | Rust (target) | Paper (reference) | Safety Margin |
|---------|---|---|---|
| d=20, n=1000 | 1-2 sec | 1-2 sec | 3× |
| d=50, n=1000 | 5-10 sec | 5-10 sec | 5× |
| d=100, n=1000 | 30-60 sec | 30-60 sec | 10× |

See [BENCHMARKING.md](BENCHMARKING.md) for full performance analysis.

## Building from Source

```bash
# Clone repository
git clone https://github.com/pristley/notears
cd notears

# Build library
cargo build --release

# Run tests
cargo test --release

# Run benchmarks
cargo bench --bench bench_end_to_end

# Generate documentation
cargo doc --open
```

## Minimum Supported Rust Version (MSRV)

NOTEARS requires **Rust 1.56+** and works with:
- ✅ Stable channel
- ✅ Beta channel  
- ✅ Nightly channel (tested on latest)

Older Rust versions may work but are not officially supported.

## Project Structure

```
notears/
├── src/                           # Core library
│   ├── lib.rs                    # Library root
│   ├── types.rs                  # Type definitions & configuration
│   ├── optimization.rs           # L-BFGS + Augmented Lagrangian solver
│   ├── acyclicity.rs             # Differentiable acyclicity constraint
│   ├── scoring.rs                # Loss functions & gradients
│   └── utils.rs                  # Matrix operations & utilities
├── tests/                         # Integration test suite
│   ├── test_acyclicity.rs        # Constraint tests
│   ├── test_optimization.rs      # Solver tests
│   ├── test_scoring.rs           # Loss function tests
│   ├── test_integration.rs       # End-to-end workflows
│   └── common.rs                 # Test utilities
├── benches/                       # Performance benchmarks
│   ├── bench_matrix_ops.rs       # Low-level operations (matrix exp, etc.)
│   ├── bench_optimization.rs     # Intermediate solver components
│   ├── bench_end_to_end.rs       # Full algorithm end-to-end
│   └── profiling_utils.rs        # Benchmark utilities & data generation
├── examples/                      # Tutorial Jupyter notebooks
│   ├── 01_quick_start.ipynb      # Getting started guide
│   └── 02_configuration_best_practices.ipynb  # Configuration by regime
├── docs/                          # Detailed documentation
│   ├── API.md                    # Function & type reference
│   └── CONFIGURATION.md          # Hyperparameter tuning guide
├── .github/workflows/             # GitHub Actions CI/CD
│   ├── tests.yml                 # Multi-version testing (1.56+, stable, beta, nightly)
│   ├── benchmarks.yml            # Performance benchmarking with regression detection
│   └── docs.yml                  # Documentation generation & deployment
├── README.md                      # Project overview (you are here)
├── CHANGELOG.md                   # Version history & release notes
├── BENCHMARKING.md                # Performance profiling suite guide
├── TROUBLESHOOTING.md             # Common issues & debugging
├── DEPLOYMENT.md                  # Production deployment guide
├── Cargo.toml                     # Project manifest (MSRV: 1.56+)
└── LICENSE                        # MIT license
```

## Configuration Guide

Three preset configurations for common data regimes:

### Small n, Large d (Underdetermined)
```rust
OptimizationConfig {
    max_outer_iterations: 20,
    max_lbfgs_iterations: 200,
    lbfgs_memory: 10,
    constraint_tolerance: 1e-7,
    penalty_rho_init: 1.0,      // Higher penalty for faster DAG feasibility
    progress_rate: 0.25,
    edge_threshold: 0.3,
}
```

### Large n, Small d (Overdetermined)
```rust
OptimizationConfig {
    max_outer_iterations: 15,
    max_lbfgs_iterations: 100,
    lbfgs_memory: 20,
    constraint_tolerance: 1e-8,
    penalty_rho_init: 0.1,      // Lower penalty for fine-tuning
    progress_rate: 0.1,         // Stricter progress criterion
    edge_threshold: 0.3,
}
```

### Balanced (Default)
```rust
OptimizationConfig::default()  // See types.rs for values
```

## Testing

```bash
# Unit tests
cargo test --lib

# Integration tests
cargo test --test '*'

# All tests with logging
RUST_LOG=debug cargo test -- --nocapture

# Specific test
cargo test test_acyclicity_constraint -- --nocapture
```

## Performance Profiling

```bash
# Run benchmarks
cargo bench --bench bench_matrix_ops
cargo bench --bench bench_optimization
cargo bench --bench bench_end_to_end

# Generate baseline for regression testing
cargo bench -- --save-baseline initial

# Compare against baseline
cargo bench -- --baseline initial

# Flame graph profiling (Linux)
cargo flamegraph --bench bench_end_to_end
```

See [BENCHMARKING.md](BENCHMARKING.md) for detailed profiling guide.

## CI/CD Status

[![Tests](https://github.com/pristley/notears/workflows/Tests/badge.svg)](https://github.com/pristley/notears/actions)
[![Benchmarks](https://github.com/pristley/notears/workflows/Benchmarks/badge.svg)](https://github.com/pristley/notears/actions)
[![Code Coverage](https://codecov.io/gh/pristley/notears/badge.svg)](https://codecov.io/gh/pristley/notears)
[![Documentation](https://github.com/pristley/notears/workflows/Docs/badge.svg)](https://docs.rs/notears)

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Add tests for new functionality
4. Ensure all tests pass: `cargo test`
5. Run benchmarks: `cargo bench`
6. Submit a pull request

## References

- **NOTEARS Paper:** Zheng et al. (2018) — [DAGs with NO TEARS: Continuous Optimization for Learning Acyclic Graphs](https://arxiv.org/abs/1803.01422)
- **Matrix Exponential:** Higham (2008) — [Functions of Matrices: Theory and Computation](https://arxiv.org/abs/0804.4150)
- **Augmented Lagrangian:** Boyd & Parikh (2011) — [Distributed Optimization and Statistical Learning](https://web.stanford.edu/~boyd/admm_book.html)

## License

Licensed under the MIT License — see [LICENSE](LICENSE) file for details.

## Citation

If you use NOTEARS in your research, please cite:

```bibtex
@inproceedings{zheng2018dags,
  title={DAGs with NO TEARS: Continuous Optimization for Learning Acyclic Graphs},
  author={Zheng, Xun and Aragam, Bryon and Ravikumar, Pradeep K and Xing, Eric P},
  booktitle={Advances in Neural Information Processing Systems},
  pages={9472--9483},
  year={2018}
}
```

## Acknowledgments

- Original algorithm by Zheng et al. (2018)
- Built with [ndarray](https://docs.rs/ndarray), [nalgebra](https://docs.rs/nalgebra), and [rayon](https://docs.rs/rayon)
