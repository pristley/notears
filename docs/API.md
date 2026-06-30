# API Reference

Complete documentation of NOTEARS public types and functions.

## Core Types

### `WeightMatrix`
Type alias for $d \times d$ weight matrix representing the DAG structure.

```rust
pub type WeightMatrix = Array2<f64>;
```

**Properties:**
- Square matrix (d×d)
- Lower triangular when acyclic (not required as input)
- Element $W_{ij}$ represents edge weight from node $j$ to node $i$

---

### `DataMatrix`
Type alias for $n \times d$ data matrix with $n$ samples and $d$ variables.

```rust
pub type DataMatrix = Array2<f64>;
```

**Properties:**
- $n$ rows (samples)
- $d$ columns (variables)
- Typically standardized before solving

---

### `OptimizationConfig`

Configuration for the augmented Lagrangian and L-BFGS optimization.

```rust
pub struct OptimizationConfig {
    pub max_outer_iterations: usize,    // Augmented Lagrangian iterations
    pub max_lbfgs_iterations: usize,    // Inner L-BFGS steps
    pub lbfgs_memory: usize,             // L-BFGS history size (10-20)
    pub constraint_tolerance: f64,       // Convergence: h(W) < tol
    pub penalty_rho_init: f64,           // Initial penalty parameter ρ
    pub progress_rate: f64,              // Rate for adaptive ρ (0.1-0.25)
    pub edge_threshold: f64,             // Threshold for edge detection
}
```

**Default values:**
```rust
OptimizationConfig::default() = OptimizationConfig {
    max_outer_iterations: 100,
    max_lbfgs_iterations: 50,
    lbfgs_memory: 10,
    constraint_tolerance: 1e-8,
    penalty_rho_init: 1.0,
    progress_rate: 0.25,
    edge_threshold: 0.3,
}
```

**Data Regime Presets:**

#### Small n, Large d
```rust
OptimizationConfig {
    max_outer_iterations: 20,
    max_lbfgs_iterations: 200,
    lbfgs_memory: 10,
    constraint_tolerance: 1e-7,
    penalty_rho_init: 1.0,        // Higher penalty for faster feasibility
    progress_rate: 0.25,
    edge_threshold: 0.3,
}
```

#### Large n, Small d  
```rust
OptimizationConfig {
    max_outer_iterations: 15,
    max_lbfgs_iterations: 100,
    lbfgs_memory: 20,
    constraint_tolerance: 1e-8,
    penalty_rho_init: 0.1,        // Lower penalty for fine-tuning
    progress_rate: 0.1,           // Stricter progress
    edge_threshold: 0.3,
}
```

---

### `RegularizationConfig`

Regularization settings for the loss function.

```rust
pub struct RegularizationConfig {
    pub lambda: f64,               // L1 regularization weight (∈ [0,1])
    pub use_l2: bool,              // L2 regularization (Tikhonov)
}
```

**Construction:**
```rust
let config = RegularizationConfig::new(0.1, false)?;  // λ=0.1, no L2
```

**Validation:**
- `lambda ∈ [0, 1]` — unbounded λ not recommended
- Returns `ConfigError` if invalid

---

### `OptimizationResult`

Output from optimization solver.

```rust
pub struct OptimizationResult {
    pub weight_matrix: WeightMatrix,     // Learned W (d×d)
    pub constraint_violation: f64,       // Final h(W)
    pub iterations: usize,               // Outer loop iterations
    pub final_score: f64,                // F(W) + λ||W||₁
    pub adjacency_matrix: WeightMatrix,  // Binary: 1 if |W_ij| > threshold
}
```

**Methods:**

#### `edges(&self) -> Vec<(usize, usize, f64)>`
Extract edges as (source, target, weight) triplets.

```rust
let edges = result.edges();
for (i, j, weight) in edges {
    println!("Edge {}→{}: weight={}", i, j, weight);
}
```

#### `is_acyclic(&self) -> bool`
Check if constraint is satisfied (within tolerance).

```rust
if result.is_acyclic() {
    println!("Valid DAG learned");
}
```

---

## Main API Functions

### `optimization::solve()`

Simplest entry point: solve with default configuration.

```rust
pub fn solve(data: &DataMatrix, lambda: f64) 
    -> Result<OptimizationResult, OptimizationError>
```

**Parameters:**
- `data`: Standardized data matrix (n×d)
- `lambda`: L1 regularization strength ∈ [0, 1]

**Returns:**
- `Ok(OptimizationResult)` on success
- `Err(OptimizationError)` if convergence fails or input invalid

**Example:**
```rust
use notears::optimization::solve;
use notears::utils::standardize_data;

let data = Array2::zeros((1000, 20));
let standardized = standardize_data(&data)?;
let result = solve(&standardized, 0.1)?;
```

---

### `optimization::solve_with_config()`

Advanced entry point: full configuration control.

```rust
pub fn solve_with_config(
    data: &DataMatrix,
    opt_config: OptimizationConfig,
    reg_config: RegularizationConfig,
) -> Result<OptimizationResult, OptimizationError>
```

**Example:**
```rust
let opt_config = OptimizationConfig {
    max_outer_iterations: 20,
    constraint_tolerance: 1e-9,
    ..OptimizationConfig::default()
};
let reg_config = RegularizationConfig::new(0.05, false)?;

let result = solve_with_config(&standardized, opt_config, reg_config)?;
```

---

## Utility Functions

### `utils::standardize_data()`

Standardize data to zero mean, unit variance.

```rust
pub fn standardize_data(data: &DataMatrix) 
    -> Result<DataMatrix, UtilError>
```

**Recommended:** Always standardize before solving.

```rust
let data_raw = load_data_from_file("data.csv")?;
let data_standardized = standardize_data(&data_raw)?;
let result = solve(&data_standardized, 0.1)?;
```

---

### `utils::matrix_exponential()`

Compute matrix exponential using Padé approximation with scaling-squaring.

```rust
pub fn matrix_exponential(weight_matrix: &WeightMatrix) 
    -> Result<WeightMatrix, UtilError>
```

**Properties:**
- Numerically stable for well-conditioned matrices
- O(d³·log d) complexity
- Relative error < 1e-14 for typical matrices

---

### `acyclicity::acyclicity_constraint()`

Compute acyclicity constraint $h(W) = \text{tr}(e^{W \odot W}) - d$.

```rust
pub fn acyclicity_constraint(weight_matrix: &WeightMatrix) 
    -> Result<f64, AcyclicityError>
```

**Properties:**
- $h(W) = 0$ ⟺ W is acyclic
- $h(W) > 0$ for cyclic W
- Differentiable everywhere

**Example:**
```rust
let h = acyclicity_constraint(&w)?;
println!("Constraint violation: {:.2e}", h);
if h < 1e-8 {
    println!("Valid DAG");
}
```

---

### `acyclicity::acyclicity_gradient()`

Compute gradient $\nabla h(W) = e^{W \odot W}^T \odot 2W$.

```rust
pub fn acyclicity_gradient(weight_matrix: &WeightMatrix) 
    -> Result<WeightMatrix, AcyclicityError>
```

**Formula:** $[\nabla h]_{ij} = 2 W_{ij} [e^{W \odot W}]_{ji}^T$

Used internally in L-BFGS optimization.

---

### `acyclicity::acyclicity_with_gradient()`

Efficient: compute both constraint and gradient in one call.

```rust
pub fn acyclicity_with_gradient(weight_matrix: &WeightMatrix)
    -> Result<(f64, WeightMatrix), AcyclicityError>
```

Saves recomputation of matrix exponential vs calling both separately.

---

### `scoring::mse_loss()`

Compute mean squared error: $\ell(W) = \frac{1}{2n}\|X - XW\|_F^2$.

```rust
pub fn mse_loss(data: &DataMatrix, weight_matrix: &WeightMatrix)
    -> Result<f64, ScoringError>
```

**Data fidelity term** in loss function.

---

### `scoring::l1_penalty()`

Compute L1 penalty: $\|W\|_1 = \sum_{ij} |W_{ij}|$.

```rust
pub fn l1_penalty(weight_matrix: &WeightMatrix) -> f64
```

**Sparsity term** in loss function.

---

### `scoring::total_loss()`

Compute complete loss: $F(W) = \ell(W) + \lambda \|W\|_1$.

```rust
pub fn total_loss(
    data: &DataMatrix,
    weight_matrix: &WeightMatrix,
    config: &RegularizationConfig,
) -> Result<f64, ScoringError>
```

---

## Error Types

### `OptimizationError`

```rust
pub enum OptimizationError {
    #[error("Configuration error: {0}")]
    Config(#[from] ConfigError),
    
    #[error("Acyclicity error: {0}")]
    Acyclicity(#[from] AcyclicityError),
    
    #[error("Scoring error: {0}")]
    Scoring(#[from] ScoringError),
    
    #[error("Failed to converge after {max_iterations} iterations. Final h(W)={h_value}")]
    ConvergenceFailed { max_iterations: usize, h_value: f64 },
}
```

### `ConfigError`

Configuration validation failures:
- `InvalidMaxOuterIterations` — must be > 0
- `InvalidConstraintTolerance` — must be ∈ [1e-10, 1e-4]
- `InvalidLambda` — must be ∈ [0, 1]
- `InvalidEdgeThreshold` — must be > 0

### `AcyclicityError`

- `NonSquareMatrix` — W must be square
- `NumericalError` — matrix exponential failed

### `ScoringError`

- `DimensionMismatch` — data dimensions don't match W
- `EmptyData` — no samples or variables
- `NonSquareWeight` — W must be square

---

## Working with Results

### Extract Edges

```rust
let edges = result.edges();
// Vec<(usize, usize, f64)> = [(source, target, weight), ...]

for (i, j, weight) in edges {
    if weight.abs() > 0.1 {
        println!("Strong edge {}→{}: {:.3}", i, j, weight);
    }
}
```

### Check Convergence

```rust
if result.constraint_violation < 1e-8 {
    println!("✓ Converged: valid DAG");
} else if result.constraint_violation < 1e-6 {
    println!("⚠ Marginal convergence: check acyclicity");
} else {
    println!("✗ Failed: h(W) = {:.2e}", result.constraint_violation);
}
```

### Adjacency Matrix

```rust
// Binary matrix: 1 if |W_ij| > threshold
let adj = &result.adjacency_matrix;

// Convert to edge list
let mut edge_list = Vec::new();
for i in 0..d {
    for j in 0..d {
        if adj[[i,j]] > 0.5 {
            edge_list.push((i, j));
        }
    }
}
```

### Save/Load Results

With `serde` feature:
```rust
use serde_json;

// Save
let json = serde_json::to_string(&result.weight_matrix)?;
std::fs::write("weights.json", json)?;

// Load
let json = std::fs::read_to_string("weights.json")?;
let w: WeightMatrix = serde_json::from_str(&json)?;
```

---

## Common Patterns

### Automatic Configuration Selection

```rust
fn solve_adaptive(data: &DataMatrix, lambda: f64) 
    -> Result<OptimizationResult, Box<dyn std::error::Error>> 
{
    let (n, d) = data.dim();
    
    let config = if n < d {
        // Underdetermined: small n, large d
        OptimizationConfig {
            penalty_rho_init: 1.0,
            ..Default::default()
        }
    } else {
        // Overdetermined: large n, small d
        OptimizationConfig {
            penalty_rho_init: 0.1,
            max_lbfgs_iterations: 100,
            ..Default::default()
        }
    };
    
    let reg_config = RegularizationConfig::new(lambda, false)?;
    Ok(solve_with_config(data, config, reg_config)?)
}
```

### Lambda Selection via Grid Search

```rust
fn select_lambda(data: &DataMatrix) 
    -> Result<f64, Box<dyn std::error::Error>> 
{
    let lambdas = vec![0.01, 0.05, 0.1, 0.2, 0.5];
    let mut best_lambda = 0.1;
    let mut best_score = f64::INFINITY;
    
    for lambda in lambdas {
        let result = solve(data, lambda)?;
        if result.final_score < best_score {
            best_score = result.final_score;
            best_lambda = lambda;
        }
    }
    
    Ok(best_lambda)
}
```

### Batch Processing Multiple Datasets

```rust
fn process_multiple(datasets: Vec<DataMatrix>, lambda: f64)
    -> Result<Vec<OptimizationResult>, Box<dyn std::error::Error>>
{
    let mut results = Vec::new();
    
    for data in datasets {
        let standardized = standardize_data(&data)?;
        let result = solve(&standardized, lambda)?;
        results.push(result);
    }
    
    Ok(results)
}
```

---

## Performance Considerations

### Memory Usage
- $O(n \cdot d)$ for data matrix
- $O(d^2)$ for weight matrix and gradients
- $O(\text{lbfgs\_memory} \cdot d^2)$ for L-BFGS history

### Time Complexity
- Per iteration: $O(d^3 \log d)$ (matrix exponential dominates)
- Total: $O(\text{iterations} \cdot d^3 \log d)$

### Optimization Tips
1. **Standardize data** — improves convergence
2. **Start with λ = 0.1** — tune if needed
3. **Use appropriate config** — depends on n vs d
4. **Monitor iterations** — if > max_outer_iterations, increase tolerance or lambda

---

## See Also

- [Configuration Guide](CONFIGURATION.md) — Detailed tuning advice
- [Benchmarking](../BENCHMARKING.md) — Performance profiling
- [Troubleshooting](../TROUBLESHOOTING.md) — Common issues
- [Examples](../examples/) — Jupyter notebooks
