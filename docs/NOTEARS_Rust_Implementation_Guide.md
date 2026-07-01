# NOTEARS Rust Implementation Guide
## Complete Technical Reference for Building DAG Learning Systems

---

## Executive Summary

NOTEARS (NO TEARS - NO Test REstrictions) is a breakthrough method for learning directed acyclic graph (DAG) structures from observational data. This guide provides software engineers with everything needed to implement NOTEARS in Rust from scratch.

**Key Innovation**: Instead of searching over exponential graph space (like GES, PC algorithms), NOTEARS uses a differentiable acyclicity constraint h(W) = tr(exp(W⊙W)) - d and continuous optimization (augmented Lagrangian + L-BFGS). This enables:
- Fast computation: O(n·d³) per iteration (vs. exponential for alternatives)
- Large-scale problems: handles 500+ variables on CPU
- Better solutions: achieves lower loss than greedy approaches
- Theoretical guarantees: recovers true DAG under identifiability assumptions

**Reference**: Zheng et al. (2018). "DAGs with NO TEARS: Continuous Optimization for Learning Acyclic Graph Structures"

---

## PART I: MATHEMATICAL FOUNDATIONS

### Section 1: Problem Formulation

#### 1.1 Structural Equation Model (SEM)

NOTEARS assumes linear SEM:
```
X_j = ∑_k W_jk X_k + ε_j,  for all j ∈ {1,...,d}

In matrix form: X = X·W^T + ε
```

Where:
- **X**: n×d data matrix (n samples, d variables)
- **W**: d×d weight matrix (W[i,j] = edge weight i→j)
- **ε**: n×d noise (independent, mean zero, finite variance)
- **ε ~ N(0, σ²I)** typically assumed (Gaussian noise)

**Acyclicity**: W must correspond to acyclic DAG (no cycles allowed)

**Identifiability**: Under assumptions on ε, the DAG structure is identifiable from observations

---

#### 1.2 Acyclicity Constraint (The Innovation)

Standard DAG: no cycles means W^k = 0 for some k

**Key insight**: Use trace of matrix exponential
```
h(W) = tr(exp(W ⊙ W)) - d

Property: h(W) = 0 ⟺ W acyclic
```

Where:
- **⊙** = Hadamard product (element-wise multiplication)
- **exp()** = matrix exponential
- **tr()** = trace (sum of diagonal elements)
- **d** = number of variables

**Why this works**:
- If W has cycle of length k: eigenvalues of W⊙W include positive values
- exp(·) amplifies eigenvalues → trace increases
- Only acyclic W has h(W) = 0
- h(W) is differentiable → can use gradient-based optimization

---

#### 1.3 Optimization Problem

```
minimize: L(W) + λ||W||_1
subject to: h(W) = 0
            (W_ii = 0 diagonal constraint)

Where:
- L(W) = loss function (e.g., MSE, or likelihood)
- λ > 0 = regularization strength
- ||W||_1 = sum of absolute values (induces sparsity)
```

**Augmented Lagrangian** reformulation:
```
minimize: L(W) + λ||W||_1 + ρ/2 * h(W)^2 + γ * h(W)

ρ > 0 = penalty parameter (increases over iterations)
γ = Lagrange multiplier (adjusted via dual updates)

Key: at convergence, h(W) → 0, so constraint satisfied
```

---

### Section 2: The Algorithm

#### 2.1 Three-Loop Structure

```
Outer loop (augmented Lagrangian):
├─ For each ρ value in {ρ_1, ρ_2, ..., ρ_T}:
│  │
│  └─ Middle loop (penalty increase):
│     ├─ Solve primal subproblem with L-BFGS
│     │
│     └─ Inner loop (L-BFGS iterations):
│        ├─ Gradient of augmented Lagrangian
│        ├─ Line search
│        └─ Hessian approximation (L-BFGS memory)
│
└─ Dual update (γ, ρ)
```

---

#### 2.2 Pseudocode

```
Algorithm: NOTEARS

Input:
  X (n × d data matrix)
  λ (regularization strength)
  ω (penalty growth rate)
  ρ_init (initial penalty)

Output:
  W (d × d weight matrix)

1. Standardize X (zero mean, unit variance)

2. Initialize: W ← 0, γ ← 0, ρ ← ρ_init

3. For outer_iter = 1 to MAX_OUTER_ITERS:
   
   4. For inner_iter = 1 to MAX_LBFGS_ITERS:
      a. Compute gradient:
         ∇f = ∇L(W) + λ·sign(W) + ρ·h(W)·∇h(W) + γ·∇h(W)
      
      b. L-BFGS update: W ← W - step_size·(B^{-1}·∇f)
         where B = approximate Hessian (limited memory)
   
   5. Dual update:
      γ ← γ + ρ·h(W)
      ρ ← ρ·ω  (increase penalty)
   
   6. Check convergence:
      if h(W) < h_tol and ||W_new - W_old|| < w_tol:
         return W
      endif
   
7. Return W

Conversions:
  - W[i,j] > threshold → edge i→j exists
  - Adjacency matrix: A[i,j] = 1{|W[i,j]| > θ}
```

---

## PART II: RUST IMPLEMENTATION GUIDE (7 Phases)

### Phase 1: Project Setup & Dependencies

**Create project**:
```bash
cargo new notears
cd notears
```

**Cargo.toml** (required dependencies):
```toml
[dependencies]
ndarray = "0.15"                    # Matrix operations
ndarray-linalg = { version="0.14", features=["openblas"] }
nalgebra = "0.32"                  # Alternative: linear algebra
ndarray-rand = "0.14"              # Random matrices
rand = "0.8"                       # Random number generation
serde = { version="1.0", features=["derive"] }  # Serialization
serde_json = "1.0"

[dev-dependencies]
criterion = "0.5"                  # Benchmarking
approx = "0.5"                     # Approximate equality
ndarray-stats = "0.13"             # Statistical operations
```

**Project structure**:
```
notears/
├── Cargo.toml
├── src/
│   ├── lib.rs
│   ├── types.rs           # Data structures
│   ├── scoring.rs         # Loss function L(W)
│   ├── acyclicity.rs      # h(W) and ∇h(W)
│   ├── optimization.rs    # L-BFGS & augmented Lagrangian
│   └── utils.rs           # Utilities (standardize, etc.)
├── tests/
│   └── integration_tests.rs
└── benches/
    └── benchmarks.rs
```

---

### Phase 2: Core Data Structures

**src/types.rs**:
```rust
use ndarray::Array2;

pub type WeightMatrix = Array2<f64>;
pub type DataMatrix = Array2<f64>;

#[derive(Clone, Debug)]
pub struct OptimizationConfig {
    pub max_outer_iterations: usize,
    pub max_lbfgs_iterations: usize,
    pub constraint_tolerance: f64,
    pub variable_tolerance: f64,
    pub penalty_growth_rate: f64,
    pub initial_penalty: f64,
}

#[derive(Clone, Debug)]
pub struct RegularizationConfig {
    pub lambda: f64,
    pub use_bias: bool,
}

#[derive(Clone, Debug)]
pub struct OptimizationResult {
    pub weight_matrix: WeightMatrix,
    pub adjacency_matrix: WeightMatrix,
    pub converged: bool,
    pub iterations: usize,
    pub final_score: f64,
    pub constraint_violation: f64,
}
```

---

### Phase 3: Acyclicity Constraint Implementation

**src/acyclicity.rs** - Computing h(W) and ∇h(W):

```rust
use ndarray::{Array2, Array1};
use ndarray_linalg::Norm;

/// Compute h(W) = tr(exp(W ⊙ W)) - d
pub fn acyclicity_constraint(w: &Array2<f64>) -> Result<f64, String> {
    let d = w.nrows();
    if w.ncols() != d {
        return Err("W must be square".to_string());
    }
    
    // Hadamard product: W ⊙ W (element-wise multiplication)
    let w_hadamard = w * w;
    
    // Matrix exponential: exp(W ⊙ W)
    let exp_w = matrix_exponential(&w_hadamard)?;
    
    // Trace: sum of diagonal
    let trace: f64 = exp_w.diag().sum();
    
    // Return h(W) = tr(...) - d
    Ok(trace - d as f64)
}

/// Compute ∇h(W) = exp(W ⊙ W)^T ⊙ 2*W
pub fn acyclicity_gradient(w: &Array2<f64>) -> Result<Array2<f64>, String> {
    let d = w.nrows();
    
    // W ⊙ W (Hadamard product)
    let w_hadamard = w * w;
    
    // exp(W ⊙ W)
    let exp_w = matrix_exponential(&w_hadamard)?;
    
    // (exp(W ⊙ W))^T (transpose)
    let exp_w_t = exp_w.t();
    
    // 2*W
    let two_w = 2.0 * w;
    
    // Hadamard product: exp_w_t ⊙ 2*W
    Ok(&exp_w_t * &two_w)
}

/// Matrix exponential using scaling-and-squaring + Padé approximation
/// Numerically stable for matrices with eigenvalues up to ~60
fn matrix_exponential(a: &Array2<f64>) -> Result<Array2<f64>, String> {
    // [Implementation uses standard algorithm with scaling]
    // Typically use ndarray_linalg or nalgebra built-in
    
    // Simplified: use eigendecomposition if available
    // For production, use specialized library
    todo!("Implement via ndarray_linalg or custom Padé approximation")
}
```

---

### Phase 4: Loss Function & Scoring

**src/scoring.rs**:

```rust
use ndarray::Array2;

/// Mean Squared Error: L(W) = ||X - X@W^T||_F^2 / (2*n)
pub fn mse_loss(x: &Array2<f64>, w: &Array2<f64>) -> Result<f64, String> {
    let n = x.nrows();
    let xwt = x.dot(&w.t());
    let residual = x - &xwt;
    let loss = residual.mapv(|x| x * x).sum() / (2.0 * n as f64);
    Ok(loss)
}

/// Gradient of MSE: ∇_W L(W) = -X^T @ (X - X@W^T) / n
pub fn mse_loss_gradient(x: &Array2<f64>, w: &Array2<f64>) -> Result<Array2<f64>, String> {
    let n = x.nrows() as f64;
    let xwt = x.dot(&w.t());
    let residual = x - &xwt;
    let grad = -x.t().dot(&residual) / n;
    Ok(grad)
}

/// L1 penalty for sparsity
pub fn l1_penalty(w: &Array2<f64>) -> f64 {
    w.mapv(|x| x.abs()).sum()
}

/// Proximal gradient step for L1 (soft thresholding)
pub fn soft_threshold(w: &Array2<f64>, lambda: f64) -> Array2<f64> {
    w.mapv(|x| {
        if x > lambda {
            x - lambda
        } else if x < -lambda {
            x + lambda
        } else {
            0.0
        }
    })
}
```

---

### Phase 5: L-BFGS Optimization

**src/optimization.rs** - Core L-BFGS implementation:

```rust
use ndarray::{Array2, Array1};

pub struct LBFGSOptimizer {
    max_iterations: usize,
    memory_size: usize,
    tolerance: f64,
}

impl LBFGSOptimizer {
    pub fn new(max_iterations: usize) -> Self {
        LBFGSOptimizer {
            max_iterations,
            memory_size: 20,
            tolerance: 1e-6,
        }
    }
    
    /// Minimize objective = loss + λ||W||_1 + ρ/2*h(W)^2 + γ*h(W)
    pub fn optimize(
        &self,
        w_init: &Array2<f64>,
        lambda: f64,
        rho: f64,
        gamma: f64,
        loss_fn: impl Fn(&Array2<f64>) -> Result<f64, String>,
        grad_fn: impl Fn(&Array2<f64>) -> Result<Array2<f64>, String>,
        h_fn: impl Fn(&Array2<f64>) -> Result<f64, String>,
        h_grad_fn: impl Fn(&Array2<f64>) -> Result<Array2<f64>, String>,
    ) -> Result<(Array2<f64>, usize), String> {
        
        let mut w = w_init.clone();
        let mut s_history = vec![];  // Step vectors
        let mut y_history = vec![];  // Gradient differences
        
        for iter in 0..self.max_iterations {
            // Compute gradient of augmented Lagrangian
            let grad_loss = grad_fn(&w)?;
            let h_val = h_fn(&w)?;
            let grad_h = h_grad_fn(&w)?;
            let grad_l1 = &w.mapv(|x| x.signum());
            
            let grad_aug = &grad_loss 
                + &(grad_l1 * lambda)
                + &(&grad_h * (rho * h_val + gamma));
            
            // Compute direction using L-BFGS approximation
            let direction = self.compute_direction(&grad_aug, &s_history, &y_history);
            
            // Line search with backtracking
            let (w_new, step_size) = self.line_search(
                &w, &direction, lambda, rho, gamma,
                &loss_fn, &grad_fn, &h_fn, &h_grad_fn
            )?;
            
            // Update history
            let s = &w_new - &w;
            let grad_new = grad_fn(&w_new)?;
            let y = &grad_new - &grad_loss;
            
            s_history.push(s.clone());
            y_history.push(y.clone());
            
            // Keep only last M vectors
            if s_history.len() > self.memory_size {
                s_history.remove(0);
                y_history.remove(0);
            }
            
            // Check convergence
            if grad_aug.norm_l2() < self.tolerance {
                return Ok((w_new, iter + 1));
            }
            
            w = w_new;
        }
        
        Ok((w, self.max_iterations))
    }
    
    fn compute_direction(
        &self,
        gradient: &Array2<f64>,
        s_history: &[Array2<f64>],
        y_history: &[Array2<f64>],
    ) -> Array2<f64> {
        // [L-BFGS two-loop recursion]
        // Approximate inverse Hessian applied to gradient
        gradient.clone()  // Simplified; full implementation uses two-loop recursion
    }
    
    fn line_search(
        &self,
        w: &Array2<f64>,
        direction: &Array2<f64>,
        // ... parameters ...
    ) -> Result<(Array2<f64>, f64), String> {
        // Backtracking line search with Armijo condition
        let mut alpha = 1.0;
        for _ in 0..20 {
            let w_new = w + &(direction * alpha);
            // Check Armijo condition
            // If satisfied: return (w_new, alpha)
            alpha *= 0.5;
        }
        Err("Line search failed".to_string())
    }
}
```

---

### Phase 6: Augmented Lagrangian Solver

**Full optimization loop**:

```rust
pub fn solve_notears(
    x: &Array2<f64>,
    lambda: f64,
    config: &OptimizationConfig,
) -> Result<OptimizationResult, String> {
    let d = x.ncols();
    let mut w = Array2::<f64>::zeros((d, d));
    let mut gamma = 0.0;
    let mut rho = config.initial_penalty;
    
    let optimizer = LBFGSOptimizer::new(config.max_lbfgs_iterations);
    
    for outer_iter in 0..config.max_outer_iterations {
        // Solve primal subproblem
        let (w_new, _) = optimizer.optimize(
            &w, lambda, rho, gamma,
            |w| mse_loss(x, w),
            |w| mse_loss_gradient(x, w),
            |w| acyclicity_constraint(w),
            |w| acyclicity_gradient(w),
        )?;
        
        // Compute constraint violation
        let h_val = acyclicity_constraint(&w_new)?;
        
        // Dual update
        gamma += rho * h_val;
        rho *= config.penalty_growth_rate;
        
        // Check convergence
        if h_val.abs() < config.constraint_tolerance {
            let adj_matrix = (&w_new.mapv(|x| x.abs()) .mapv(|x| if x > 0.05 { 1.0 } else { 0.0 }));
            let score = mse_loss(x, &w_new)?;
            
            return Ok(OptimizationResult {
                weight_matrix: w_new,
                adjacency_matrix: adj_matrix,
                converged: true,
                iterations: outer_iter + 1,
                final_score: score,
                constraint_violation: h_val,
            });
        }
        
        w = w_new;
    }
    
    Err(format!("Failed to converge in {} iterations", config.max_outer_iterations))
}
```

---

### Phase 7: Testing & Validation

**tests/integration_tests.rs**:

```rust
#[cfg(test)]
mod tests {
    use notears::*;
    
    #[test]
    fn test_acyclicity_zero_matrix() {
        let w = Array2::<f64>::zeros((3, 3));
        let h = acyclicity_constraint(&w).unwrap();
        assert!(h.abs() < 1e-10);  // Zero matrix is acyclic
    }
    
    #[test]
    fn test_acyclicity_cyclic_matrix() {
        // Create matrix with 2→3→2 cycle
        let mut w = Array2::<f64>::zeros((3, 3));
        w[[1, 2]] = 1.0;  // 2→3
        w[[2, 1]] = 1.0;  // 3→2
        let h = acyclicity_constraint(&w).unwrap();
        assert!(h > 0.1);  // Should have positive h
    }
    
    #[test]
    fn test_mse_loss() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let w = Array2::<f64>::zeros((2, 2));
        let loss = mse_loss(&x, &w).unwrap();
        assert!(loss > 0.0);
    }
    
    #[test]
    fn test_full_pipeline() {
        // Generate synthetic DAG data
        let true_w = generate_random_dag(5, 0.3);
        let x = generate_data_from_sem(100, &true_w);
        
        // Run NOTEARS
        let config = OptimizationConfig::default();
        let result = solve_notears(&x, 0.1, &config).unwrap();
        
        // Verify basic properties
        assert!(result.converged);
        assert!(result.constraint_violation < 1e-4);
        assert!(result.iterations < 100);
    }
}
```

---

## PART III: Production Considerations

### Deployment Checklist

- [ ] **Algorithm correctness verified**: Unit tests pass, h(W) converges
- [ ] **Numerical stability**: No NaN/Inf under realistic data ranges
- [ ] **Performance benchmarked**: Meets latency SLA (e.g., 5s for d=100)
- [ ] **Data validation**: Handles edge cases (missing values, outliers, duplicates)
- [ ] **Monitoring**: Logs h(W) convergence, execution time, memory usage
- [ ] **Error handling**: Graceful fallback if optimization fails
- [ ] **Documentation**: Examples, configuration guide, troubleshooting
- [ ] **Backwards compatibility**: API stable for clients

### Recommended Monitoring Metrics

1. **Algorithm health**:
   - Final h(W) value (should be < 1e-4)
   - Number of outer iterations to convergence
   - Execution time per run

2. **Data quality**:
   - Sample size vs. variable count ratio
   - Data standardization check (mean, std)
   - Missing value rate

3. **Result quality**:
   - Sparsity of learned DAG
   - Stability across multiple runs
   - Accuracy vs. ground truth (if available)

---

## PART IV: Common Implementation Issues & Solutions

### Issue 1: Matrix Exponential Overflow

**Problem**: exp(W⊙W) produces Inf for large eigenvalues

**Solution**: 
- Use scaling-and-squaring algorithm
- Library: `ndarray_linalg::Norm` or `nalgebra`
- Scaling ensures eigenvalues in safe range before exponential

### Issue 2: L-BFGS Not Converging

**Problem**: Line search fails, direction not descent

**Solution**:
- Check gradient computation (numerical gradient vs. analytical)
- Verify learning rate initialization
- Reduce max step size in line search

### Issue 3: Poor Serialization of Results

**Problem**: Large matrices slow to save

**Solution**:
- Use sparse matrix format (COO, CSR) if graph sparse
- Compress: only store W[i,j] > threshold
- Use binary format (bincode) instead of JSON

---

## References & Further Reading

**Key Papers**:
1. Zheng et al. (2018). "DAGs with NO TEARS" - Main NOTEARS paper
2. Zheng et al. (2021). "DAGs with NO TEARS: A Continuous Optimization Approach" - Extended version
3. Shimizu et al. (2006). "LiNGAM: A new approach to causal inference" - Alternative method

**Libraries**:
- ndarray (Rust linear algebra)
- ndarray_linalg (advanced operations)
- nalgebra (alternative linear algebra)

**Resources**:
- https://github.com/xunzheng/notears - Original Python implementation
- https://causalml.readthedocs.io/ - Causal ML library documentation
- Causal inference textbooks (Pearl, Bareinboim, Peters)

---

**Version**: 1.0  
**Last Updated**: July 2026  
**Estimated Implementation Time**: 15-25 hours for complete, production-ready implementation
