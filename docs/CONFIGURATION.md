# Configuration Guide

Detailed guidance for tuning NOTEARS to different data regimes and optimization challenges.

## Quick Reference

| Regime | n vs d | Config | λ Range | Use Case |
|--------|--------|--------|---------|----------|
| **Small n, Large d** | n < d | `config_small_n` | 0.1-0.5 | Gene expression, high-dim signal |
| **Large n, Small d** | n > d | `config_large_n` | 0.01-0.2 | Sensor data, well-sampled |
| **Balanced** | n ≈ d | Default | 0.05-0.2 | Most applications |

## Configuration by Data Regime

### 1. Small n, Large d (Underdetermined)

**Characteristics:**
- Fewer samples than variables: n < d
- High-dimensional, limited data (typical in genomics, finance)
- Many possible DAGs consistent with data

**Challenges:**
- Risk of overfitting
- Ill-conditioned optimization
- Need strong regularization

**Recommended Config:**
```rust
let config = OptimizationConfig {
    max_outer_iterations: 20,        // Fewer outer iterations
    max_lbfgs_iterations: 200,       // More L-BFGS for fine optimization
    lbfgs_memory: 10,                // Smaller memory footprint
    constraint_tolerance: 1e-7,      // Slightly relaxed tolerance
    penalty_rho_init: 1.0,           // HIGH initial penalty
    progress_rate: 0.25,             // Standard progress check
    edge_threshold: 0.3,
};
```

**Regularization:**
```rust
// Use moderate to strong L1 regularization
let lambdas_to_try = vec![0.1, 0.2, 0.3, 0.5];

// Avoid very small λ (overfitting)
// Avoid very large λ (all zeros solution)
```

**Tips:**
1. Increase `penalty_rho_init` (1.0 or higher) — enforces acyclicity faster
2. Use higher `max_lbfgs_iterations` (200+) — allow inner optimization to converge
3. Consider ensemble methods — average over multiple runs
4. Verify sparsity of solution — if dense, increase λ

**Example:**
```rust
use notears::types::{OptimizationConfig, RegularizationConfig};
use notears::optimization::solve_with_config;

fn solve_underdetermined(data: &DataMatrix) 
    -> Result<OptimizationResult, Box<dyn std::error::Error>> 
{
    // Data has n < d
    let (n, d) = data.dim();
    assert!(n < d, "Not underdetermined");
    
    let standardized = standardize_data(data)?;
    
    let opt_config = OptimizationConfig {
        max_outer_iterations: 20,
        max_lbfgs_iterations: 200,
        lbfgs_memory: 10,
        constraint_tolerance: 1e-7,
        penalty_rho_init: 1.0,
        progress_rate: 0.25,
        edge_threshold: 0.3,
    };
    
    let reg_config = RegularizationConfig::new(0.2, false)?;
    
    solve_with_config(&standardized, opt_config, reg_config)
}
```

---

### 2. Large n, Small d (Overdetermined)

**Characteristics:**
- Many samples relative to variables: n >> d
- Well-determined system
- Multiple equations per variable
- Good signal-to-noise ratio

**Challenges:**
- May converge prematurely
- Can find spurious edges if λ too small
- Need to balance data fidelity vs sparsity

**Recommended Config:**
```rust
let config = OptimizationConfig {
    max_outer_iterations: 15,        // Fewer outer iterations needed
    max_lbfgs_iterations: 100,       // Sufficient L-BFGS steps
    lbfgs_memory: 20,                // Larger memory for better quasi-Newton
    constraint_tolerance: 1e-8,      // Strict convergence
    penalty_rho_init: 0.1,           // LOW initial penalty
    progress_rate: 0.1,              // STRICT progress: 10% improvement
    edge_threshold: 0.3,
};
```

**Regularization:**
```rust
// Use conservative L1 regularization
let lambdas_to_try = vec![0.01, 0.05, 0.1, 0.15];

// Can use weaker regularization (more edges recovered)
// Lower λ = more sensitive to noise, but better for discovery
```

**Tips:**
1. Decrease `penalty_rho_init` (0.1 or lower) — avoid forcing premature feasibility
2. Use `progress_rate: 0.1` — stricter stopping criterion
3. Increase `lbfgs_memory` (15-20) — leverage good conditioning
4. Lower `constraint_tolerance` (1e-9) — can afford stricter tolerance
5. Consider validation set — test structure on held-out data

**Example:**
```rust
fn solve_overdetermined(data: &DataMatrix, lambda: f64) 
    -> Result<OptimizationResult, Box<dyn std::error::Error>> 
{
    let (n, d) = data.dim();
    assert!(n > d, "Not overdetermined");
    
    let standardized = standardize_data(data)?;
    
    let opt_config = OptimizationConfig {
        max_outer_iterations: 15,
        max_lbfgs_iterations: 100,
        lbfgs_memory: 20,
        constraint_tolerance: 1e-8,
        penalty_rho_init: 0.1,
        progress_rate: 0.1,
        edge_threshold: 0.3,
    };
    
    let reg_config = RegularizationConfig::new(lambda, false)?;
    
    solve_with_config(&standardized, opt_config, reg_config)
}
```

---

### 3. Balanced (Default)

**Characteristics:**
- Moderate sample size: n ≈ 2-5 × d
- Most real-world applications
- Good balance of constraints and observations

**Recommended Config:**
```rust
OptimizationConfig::default()  // Ready-to-use defaults
```

Which is:
```rust
OptimizationConfig {
    max_outer_iterations: 100,
    max_lbfgs_iterations: 50,
    lbfgs_memory: 10,
    constraint_tolerance: 1e-8,
    penalty_rho_init: 1.0,
    progress_rate: 0.25,
    edge_threshold: 0.3,
}
```

---

## Regularization Parameter λ

**Goal:** Balance data fidelity vs sparsity

$$F(W) = \underbrace{\frac{1}{2n}\|X - XW\|_F^2}_{\text{data fidelity}} + \lambda \underbrace{\|W\|_1}_{\text{sparsity}}$$

### λ Selection Strategies

#### 1. Grid Search
```rust
fn select_lambda_grid(data: &DataMatrix) 
    -> Result<f64, Box<dyn std::error::Error>> 
{
    let standardized = standardize_data(data)?;
    let lambdas = vec![0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5];
    
    let mut results = Vec::new();
    for lambda in lambdas {
        let result = solve(&standardized, lambda)?;
        results.push((lambda, result.final_score, result.edges().len()));
    }
    
    // Print results for inspection
    for (lambda, score, n_edges) in &results {
        println!("λ={:.4}: F(W)={:.6e}, edges={}", lambda, score, n_edges);
    }
    
    // Return λ with minimum score
    let (best_lambda, _, _) = results.into_iter()
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap();
    
    Ok(best_lambda)
}
```

#### 2. Cross-Validation (Proper Approach)
```rust
fn select_lambda_cv(data: &DataMatrix, n_folds: usize) 
    -> Result<f64, Box<dyn std::error::Error>> 
{
    let standardized = standardize_data(data)?;
    let (n, d) = standardized.dim();
    let fold_size = n / n_folds;
    
    let lambdas = vec![0.01, 0.05, 0.1, 0.2, 0.5];
    let mut cv_scores = vec![0.0; lambdas.len()];
    
    for fold in 0..n_folds {
        let train_start = fold * fold_size;
        let train_end = if fold == n_folds - 1 { n } else { (fold + 1) * fold_size };
        
        // Split data into train/test
        let train = standardized.slice(s![0..train_start, ..])
            .vstack(&standardized.slice(s![train_end.., ..]))?;
        let test = standardized.slice(s![train_start..train_end, ..]);
        
        for (i, &lambda) in lambdas.iter().enumerate() {
            let result = solve(&train, lambda)?;
            
            // Compute test loss
            let test_loss = scoring::mse_loss(&test, &result.weight_matrix)?;
            cv_scores[i] += test_loss / n_folds as f64;
        }
    }
    
    // Return best λ
    let best_idx = cv_scores.iter()
        .enumerate()
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap().0;
    
    Ok(lambdas[best_idx])
}
```

#### 3. BIC / AIC (Information Criteria)
```rust
fn select_lambda_bic(data: &DataMatrix) 
    -> Result<f64, Box<dyn std::error::Error>> 
{
    let standardized = standardize_data(data)?;
    let (n, d) = standardized.dim();
    let lambdas = vec![0.01, 0.05, 0.1, 0.2, 0.5];
    
    let mut bic_scores = Vec::new();
    
    for lambda in &lambdas {
        let result = solve(&standardized, lambda)?;
        
        let mse = scoring::mse_loss(&standardized, &result.weight_matrix)?;
        let k = result.edges().len();  // Number of edges
        
        // BIC = n*log(MSE) + k*log(n)
        let bic = n as f64 * mse.ln() + k as f64 * (n as f64).ln();
        
        bic_scores.push((lambda, bic, k));
    }
    
    // Print for inspection
    for (lambda, bic, k) in &bic_scores {
        println!("λ={:.4}: BIC={:.2}, edges={}", lambda, bic, k);
    }
    
    let (best_lambda, _, _) = bic_scores.into_iter()
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap();
    
    Ok(*best_lambda)
}
```

### Guidance by Problem Type

| Problem | λ Range | Notes |
|---------|---------|-------|
| Dense DAG expected | 0.01-0.05 | Few regularization penalty |
| Sparse DAG (typical) | 0.05-0.2 | Standard sparsity |
| Very sparse DAG | 0.2-0.5 | Strong regularization |
| High-dimensional (n<d) | 0.1-0.5 | Must prevent overfitting |
| Well-determined (n>d) | 0.01-0.1 | Can afford lower λ |

---

## Hyperparameter Tuning

### `penalty_rho_init`

Initial penalty parameter in augmented Lagrangian.

**Role:** Weights acyclicity constraint vs data fidelity

**Higher ρ** → Faster acyclicity enforcement, but may sacrifice fit
**Lower ρ** → Better initial fit, but slower convergence to feasibility

**Guidance:**
- **Underdetermined** (n<d): Use ρ=1.0 or higher
- **Overdetermined** (n>d): Use ρ=0.1 or lower
- **Default**: ρ=1.0

```rust
// Experiment with ρ
for rho_init in [0.1, 0.5, 1.0, 5.0, 10.0] {
    let mut config = OptimizationConfig::default();
    config.penalty_rho_init = rho_init;
    
    let reg_config = RegularizationConfig::new(0.1, false)?;
    let result = solve_with_config(&data, config, reg_config)?;
    
    println!("ρ={}: iterations={}, h(W)={:.2e}", 
             rho_init, result.iterations, result.constraint_violation);
}
```

### `constraint_tolerance`

Convergence threshold for acyclicity constraint $h(W)$.

**Smaller tolerance** → More iterations, better acyclicity
**Larger tolerance** → Faster, may have cycles

**Guidance:**
- **Strict** (scientific): 1e-9 to 1e-8
- **Standard** (default): 1e-8
- **Relaxed**: 1e-6 (for quick prototyping)

### `progress_rate`

Adaptive penalty parameter: increase ρ if progress < `progress_rate`.

**Guidance:**
- **Lenient** (0.25-0.5): Allow slow progress
- **Standard** (0.25): Default
- **Strict** (0.05-0.1): Only for overdetermined problems

---

## Common Issues & Solutions

### Issue: "Failed to converge after N iterations"

**Causes:**
1. λ too small → too many edges → can't satisfy acyclicity
2. λ too large → W ≈ 0 → trivial solution
3. ρ initialization wrong for data regime

**Solutions:**
```rust
// Increase λ
let result = solve(&data, 0.3)?;  // from 0.1

// OR adjust ρ
let mut config = OptimizationConfig::default();
config.penalty_rho_init = 5.0;  // higher penalty

// OR increase iterations
config.max_outer_iterations = 200;
```

### Issue: "h(W) converged but still has cycles"

**Causes:**
- Numerical precision issues
- Very tight edge threshold masking cycles

**Solution:**
```rust
// Increase edge threshold
let mut config = OptimizationConfig::default();
config.edge_threshold = 0.5;  // from 0.3

// Verify acyclicity manually
let edges = result.edges();
verify_acyclic(&edges)?;
```

### Issue: "Solution is all zeros"

**Causes:**
- λ too large → strong regularization suppresses all edges
- Data near singular → high noise

**Solution:**
```rust
// Decrease λ
let result = solve(&data, 0.01)?;  // from 0.1

// OR check data quality
let standardized = standardize_data(&data)?;
// Verify no NaN/Inf, check condition number
```

### Issue: "Solution is too dense"

**Causes:**
- λ too small
- ρ too large (forces feasibility but not sparsity)

**Solution:**
```rust
// Increase λ
let result = solve(&data, 0.5)?;

// OR adjust configuration
let mut config = OptimizationConfig::default();
config.penalty_rho_init = 0.5;  // lower initial penalty
config.progress_rate = 0.1;     // stricter progress
```

---

## Reproducibility Checklist

For consistent results:

- [ ] Fix random seed if applicable
- [ ] Document data preprocessing (standardization, outlier removal)
- [ ] Record `OptimizationConfig` used
- [ ] Record `RegularizationConfig` (λ, use_l2)
- [ ] Note NOTEARS version (e.g., v0.1.0)
- [ ] Specify Rust version (e.g., 1.70+)

**Example saving configuration:**
```rust
use serde_json;

let config = serde_json::json!({
    "notears_version": "0.1.0",
    "rust_version": "1.70.0",
    "optimization_config": {
        "max_outer_iterations": 100,
        "max_lbfgs_iterations": 50,
        "lbfgs_memory": 10,
        "constraint_tolerance": 1e-8,
        "penalty_rho_init": 1.0,
        "progress_rate": 0.25,
        "edge_threshold": 0.3,
    },
    "regularization_config": {
        "lambda": 0.1,
        "use_l2": false,
    },
    "data_info": {
        "n_samples": 1000,
        "n_variables": 20,
    }
});

std::fs::write("experiment_config.json", config.to_string_pretty()?)?;
```

---

## References

- [API Reference](API.md)
- [Benchmarking Guide](../BENCHMARKING.md)
- [Troubleshooting](../TROUBLESHOOTING.md)
- NOTEARS Paper: https://arxiv.org/abs/1803.01422
