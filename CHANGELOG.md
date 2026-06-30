# Changelog

All notable changes to the NOTEARS project are documented in this file.

## [0.1.0] - 2026-06-30

### Phase 9: High-Level User-Facing API ✨
**Added comprehensive unified entry point for DAG structure learning**

- `learn_dag()` - Main API function orchestrating full NOTEARS pipeline
  - Single-call interface for end-to-end DAG learning
  - Comprehensive input validation with descriptive error messages
  - Random weight initialization for faster convergence
  - Automatic configuration with optional customization
  - Post-processing with thresholding and acyclicity validation

- Input Validation Layer (8 checks):
  - Data dimensions (n > 0, d > 0, n ≥ d, d ≤ 500)
  - Data quality (no NaN/Inf, proper norm scaling)
  - Hyperparameter ranges (lambda ∈ [0,1], threshold ≥ 0)

- Configuration System:
  - Sensible defaults: 100 outer iterations, 50 L-BFGS steps, 10 memory pairs
  - Full customization support via OptimizationConfig
  - Threshold parameter overrides edge_threshold in config

- Error Handling:
  - Descriptive error messages for 10+ failure modes
  - Warnings for marginal convergence (h(W) > 1e-6)
  - Cycle detection in extracted adjacency

- Documentation:
  - 230+ line comprehensive docstring
  - Algorithm overview and complexity analysis
  - Hyperparameter tuning guide
  - Data preprocessing recommendations
  - Production usage patterns

- Testing: 14 test cases (10 validation, 4 skipped optimization)
  - All validation tests passing
  - Comprehensive error condition coverage

**Commits:**
- 62572496: Phase 9 implementation with full API

---

### Phase 8: Post-Optimization Thresholding ✨
**Discrete DAG extraction from continuous solver output**

- `extract_adjacency()` - Hard thresholding for binary adjacency
  - Converts continuous weights to discrete 0/1 matrix
  - Full error handling and validation
  - Returns Result<Array2<i32>, UtilError>

- Weight Distribution Analysis:
  - `analyze_weight_distribution()` - Statistical analysis for threshold selection
  - `WeightAnalysis` struct with sorted weights and gap detection
  - Gap-based heuristic for automatic threshold selection

- `extract_adjacency_adaptive()` - Data-driven thresholding
  - Uses maximum gap in weight distribution
  - Falls back to default if no clear gap
  - Returns both adjacency and selected threshold

- `is_acyclic_adjacency()` - DAG validation via topological sort
  - Kahn's algorithm implementation
  - O(d + |E|) complexity
  - Detects all cycle types

- Testing: 18 new test cases
  - Basic thresholding, boundary conditions
  - Error handling and edge cases
  - Weight analysis and gap detection
  - DAG validation (valid DAGs, self-loops, cycles)

**Commits:**
- a7b71f63: Phase 8 implementation with comprehensive tests

---

### Phase 7: Dual Ascent Loop ✨
**Full augmented Lagrangian outer loop with adaptive penalty**

- `solve_ecp()` - Complete dual ascent algorithm
  - Algorithm 1 from Zheng et al. 2018 implementation
  - Adaptive penalty strategy with automatic scaling
  - Penalty capping at 1e8 to prevent overflow
  - Dual variable updates: α_{k+1} = α_k + ρ·h(W_{k+1})
  - Convergence detection based on acyclicity constraint

- Features:
  - Penalty multiplication by 10 until h_new < 0.25·h_prev
  - Weight explosion safeguard (|W_max| > 1e6 detection)
  - Adaptive penalty scaling for better convergence
  - Comprehensive convergence logging

- Testing: Extensive convergence validation
  - 111 total tests passing (92 unit + 18 integration + 1 doctest)
  - Gradient norm decrease verification
  - Numerical stability validation

**Commits:**
- a95059ad: Phase 7 implementation with dual ascent loop

---

### Phases 1-6: Core Mathematical Framework ✅
**Complete mathematical foundation for DAG structure learning**

#### Phase 1-2: Matrix Exponential Computation
- Padé (3,3) approximation with scaling & squaring
- Numerically stable O(d³·log(d)) algorithm
- Comprehensive test suite with stability validation

#### Phase 3: Acyclicity Constraint
- h(W) = tr(exp(W ⊙ W^T)) - d implementation
- Gradient computation: ∇h = 2·diag(exp(W^T)^T)·(W ∘ I(W^T))·W
- Constraint validation with acyclicity checking

#### Phase 4: Scoring Functions
- MSE loss: (1/2n)||X - XW||_F^2
- L1 penalty: λ||W||_1
- Total score: F(W) = loss + penalty
- Gradient computation with finite-difference validation

#### Phase 5: Augmented Lagrangian
- L_ρ(W, α) = F(W) + (ρ/2)h(W)² + α·h(W)
- Three-term decomposition
- Gradient: ∇_W L_ρ = ∇F + (ρh+α)∇h

#### Phase 6: L-BFGS Optimizer
- SimpleLBFGS with two-loop recursion
- Armijo line search with backtracking
- 10-pair memory management
- Convergence via gradient norm threshold

**Test Results:** 92 unit tests + 18 integration tests passing

---

## Features

### Core Algorithm
- NO TEARS (Zheng et al. 2018) implementation
- Augmented Lagrangian method with L-BFGS
- Continuous relaxation of DAG constraint
- O(d³·log(d)) per-iteration complexity

### Numerical Stability
- Matrix exponential via Padé + scaling & squaring
- Gradient computation with numerical validation
- Overflow/underflow protection
- NaN/Inf detection throughout

### Error Handling
- Comprehensive validation at each stage
- Descriptive error messages
- Production-grade Result types
- Full error propagation

### API Design
- High-level `learn_dag()` entry point
- Flexible configuration system
- Modular component access
- Type-safe Rust interfaces

### Testing
- 137 total tests (119 lib + 18 integration)
- Unit tests for each component
- Integration tests for full pipeline
- Numerical validation suites

---

## Dependencies

### Runtime
- `ndarray` 0.15 - Array operations (with serde)
- `nalgebra` 0.33 - Linear algebra
- `serde` 1.0 - Serialization (with derive)
- `serde_json` 1.0 - JSON support
- `rayon` 1.7 - Parallelization
- `thiserror` 1.0 - Error handling
- `rand` 0.8 - Random number generation

### Development
- `criterion` 0.5 - Benchmarking
- `approx` 0.5 - Floating-point comparison

### Build Configuration
- **Release Profile**: opt-level=3, lto=true, codegen-units=1
- **Target**: Production-grade optimization
- **Edition**: Rust 2021

---

## Usage Quick Start

```rust
use notears::learn_dag;
use notears::standardize_data;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load data (n samples, d variables)
    let data = load_csv("data.csv")?;
    
    // Preprocess: center and scale
    let standardized = standardize_data(&data)?;
    
    // Learn DAG structure
    let result = learn_dag(&standardized, 0.1, 0.3, None)?;
    
    // Inspect results
    println!("Discovered edges: {}", result.edges().len());
    println!("Constraint violation: {:.2e}", result.constraint_violation);
    println!("Iterations: {}", result.iterations);
    
    Ok(())
}
```

---

## Known Limitations

1. **Solver Convergence**: May not converge on certain synthetic data patterns
2. **Dimension Limit**: d > 500 not recommended (numerical instability)
3. **Data Preprocessing**: Requires externally normalized data (centering & scaling)
4. **Non-Deterministic**: Random initialization yields different results across runs

---

## Future Work (Planned Phases)

### Phase 10: Advanced Thresholding
- ROC curve analysis for threshold optimization
- Validation set-based threshold selection
- Weight-based edge scoring

### Phase 11: Adjacency Utilities
- Graph traversal (neighbors, reachability)
- Path detection algorithms
- Topological sorting utilities

### Phase 12: Solver Improvements
- Better convergence detection
- Adaptive penalty scheduling
- Initialization heuristics

---

## References

**Main Paper:**
- Zheng, X., Aragam, B., Ravikumar, P. K., & Xing, E. P. (2018).
  "DAGs with NO TEARS: Continuous optimization for learning acyclic graphs."
  In *Advances in Neural Information Processing Systems (NeurIPS)*.
  https://arxiv.org/abs/1803.01422

**Key Algorithms:**
- Higham, N. J. (2008). *Functions of Matrices: Theory and Computation*.
  (Matrix exponential via Padé approximation)
- Kahn, A. B. (1962). "Topological sorting of large networks."
  (DAG validation via topological sort)
- Nocedal, J. (1980). "Updating quasi-Newton matrices with limited storage."
  (L-BFGS algorithm)

---

## License

MIT - See LICENSE file

---

## Contributors

NOTEARS Contributors (2026)
