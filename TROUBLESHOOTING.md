# Troubleshooting Guide

Solutions for common issues when using NOTEARS.

## Installation & Compilation

### "Could not find `notears` in the list of known crates"

**Cause:** Package not in crates.io or wrong version specified

**Solution:**
```toml
# Use published version
[dependencies]
notears = "0.1"

# Or build from source
[dependencies]
notears = { path = "../notears" }  # local path
```

---

### "Failed to resolve: use of undeclared crate `ndarray`"

**Cause:** Missing dependency

**Solution:**
```toml
[dependencies]
notears = "0.1"
ndarray = "0.15"  # Required by NOTEARS
```

---

### "Minimum supported Rust version (1.56) not met"

**Cause:** Using Rust version older than 1.56

**Solution:**
```bash
# Upgrade Rust
rustup update

# Check version
rustc --version
```

---

## Data Issues

### "Dimension mismatch: data has 20 variables but weight matrix is 30×30"

**Cause:** Weight matrix dimension doesn't match data

**Solution:**
```rust
// Ensure consistency
let (n, d) = data.dim();
let w = Array2::zeros((d, d));  // Match data dimension

// When loading pre-trained weights
assert_eq!(w.dim().0, d, "Weight matrix dimension mismatch");
```

---

### "NaN or Inf detected in computation"

**Causes:**
1. Data contains NaN or Inf
2. Numerical overflow in matrix exponential
3. Singular data matrix

**Solution:**
```rust
// Check data quality
fn validate_data(data: &Array2<f64>) -> Result<(), String> {
    if data.iter().any(|x| !x.is_finite()) {
        return Err("Data contains NaN or Inf".to_string());
    }
    
    // Standardize (also removes constant columns)
    let standardized = standardize_data(data)
        .map_err(|e| e.to_string())?;
    
    if standardized.iter().any(|x| !x.is_finite()) {
        return Err("Data still has NaN/Inf after standardization".to_string());
    }
    
    Ok(())
}

// Use validation
validate_data(&data)?;
let standardized = standardize_data(&data)?;
```

**Prevention:**
```rust
// Remove outliers (before standardization)
fn remove_outliers(data: &Array2<f64>, n_std: f64) -> Array2<f64> {
    // Keep rows within n_std standard deviations
    // ... implementation ...
}

// Remove constant columns
fn remove_constant_columns(data: &Array2<f64>) -> Array2<f64> {
    // Filter out columns with zero variance
    // ... implementation ...
}
```

---

### "Data matrix must have at least 1 sample and 1 variable"

**Cause:** Empty data array

**Solution:**
```rust
let (n, d) = data.dim();
assert!(n > 0 && d > 0, "Data must be non-empty");

// Check before solving
if n == 0 || d == 0 {
    return Err("Cannot solve with empty data".into());
}
```

---

## Convergence Issues

### "Failed to converge after 100 iterations. Final h(W)=0.02"

**Cause:** Acyclicity constraint not satisfied (h(W) > tolerance)

**Possible causes:**
1. λ too small (too many edges)
2. Configuration not suitable for data
3. Data quality issues

**Solutions:**

**Option 1: Increase regularization**
```rust
// Try stronger L1 penalty
let result = solve(&data, 0.5)?;  // from 0.1
```

**Option 2: Adjust configuration**
```rust
let mut config = OptimizationConfig::default();
config.max_outer_iterations = 200;  // More iterations
config.penalty_rho_init = 5.0;       // Higher initial penalty
config.constraint_tolerance = 1e-6;  // Relax tolerance slightly

let reg_config = RegularizationConfig::new(0.1, false)?;
let result = solve_with_config(&data, config, reg_config)?;
```

**Option 3: Use data-regime-specific config**
```rust
let (n, d) = data.dim();

let config = if n < d {
    // Underdetermined
    OptimizationConfig {
        penalty_rho_init: 2.0,
        max_outer_iterations: 30,
        ..Default::default()
    }
} else {
    // Overdetermined
    OptimizationConfig {
        penalty_rho_init: 0.1,
        constraint_tolerance: 1e-9,
        progress_rate: 0.05,
        ..Default::default()
    }
};

let reg_config = RegularizationConfig::new(0.1, false)?;
let result = solve_with_config(&data, config, reg_config)?;
```

---

### "Convergence is very slow (100+ iterations)"

**Causes:**
1. λ borderline between two regimes
2. ρ_init not appropriate
3. Data poorly scaled

**Solutions:**

**1. Check λ value**
```rust
// Try a range of λ values
for lambda in [0.05, 0.1, 0.15, 0.2, 0.3] {
    let result = solve(&data, lambda)?;
    println!("λ={:.2}: iterations={}, time={:.2}s",
             lambda, result.iterations, elapsed);
}

// Avoid boundary values
```

**2. Adjust penalty parameter**
```rust
let mut config = OptimizationConfig::default();

// If converging too slowly:
config.penalty_rho_init = 2.0;      // Faster acyclicity enforcement
config.progress_rate = 0.5;         // More lenient progress

// If spending too long in inner loop:
config.max_lbfgs_iterations = 20;   // Fewer L-BFGS steps per outer iter
```

**3. Verify data scaling**
```rust
// Check column statistics
let (n, d) = data.dim();
for j in 0..d {
    let col = data.column(j);
    let mean: f64 = col.mean().unwrap_or(0.0);
    let var: f64 = /* compute variance */ ;
    println!("Variable {}: mean={:.3}, std={:.3}", j, mean, var.sqrt());
}

// Re-standardize
let standardized = standardize_data(&data)?;
```

---

### "Solution looks degenerate (W all zeros or all same values)"

**Cause:** Strong regularization or numerical issue

**Solutions:**

**If W ≈ 0 everywhere:**
```rust
// λ too large
let result = solve(&data, 0.01)?;  // Decrease λ

// OR check data quality
let mut sum = 0.0;
for x in data.iter() {
    sum += x * x;
}
let rms = (sum / (n * d) as f64).sqrt();
println!("Data RMS: {}", rms);  // Should be ~1 after standardization
```

**If W has repeated structure:**
```rust
// Possible data issue (repeated columns/correlations)
// Use PCA to remove redundancy
// Or increase edge_threshold
let mut config = OptimizationConfig::default();
config.edge_threshold = 0.5;

let result = solve_with_config(&data, config, RegularizationConfig::new(0.1, false)?)?;
```

---

### "Error: constraint_tolerance must be in [1e-10, 1e-4]"

**Cause:** Invalid tolerance value in configuration

**Solution:**
```rust
// Valid range: [1e-10, 1e-4]
let config = OptimizationConfig {
    constraint_tolerance: 1e-8,  // ✓ Valid (between 1e-10 and 1e-4)
    ..Default::default()
};

// Invalid:
// constraint_tolerance: 1e-12,  // Too strict
// constraint_tolerance: 1e-3,   // Too relaxed
```

---

## Numerical Issues

### "Warning: acyclicity constraint h(W) = 50.00 is very large"

**Cause:** Very cyclic or ill-conditioned problem

**Solutions:**

```rust
// 1. Increase penalty parameter
let mut config = OptimizationConfig::default();
config.penalty_rho_init = 10.0;

// 2. Stronger regularization
let result = solve(&data, 0.5)?;  // Increase λ

// 3. Relax tolerance slightly
config.constraint_tolerance = 1e-5;

// 4. Check matrix exponential conditioning
let w = &result.weight_matrix;
let h = acyclicity_constraint(w)?;
println!("Constraint h(W) = {:.2e}", h);

if h > 1.0 {
    eprintln!("Warning: Large cycles in solution");
    // May indicate:
    // - Too few iterations
    // - λ too small
    // - Data quality issues
}
```

---

### "Error: Non-square weight matrix"

**Cause:** Attempted to compute matrix exponential on non-square matrix

**Solution:**
```rust
// Ensure W is always square
let (d1, d2) = w.dim();
assert_eq!(d1, d2, "Weight matrix must be square");

// When creating from data
let (n, d) = data.dim();
let w = Array2::zeros((d, d));  // Square!
```

---

### "Matrix exponential computation failed"

**Causes:**
1. Matrix has extremely large/small values (ill-conditioned)
2. Matrix dimension > 500
3. Numerical underflow/overflow

**Solutions:**

```rust
// 1. Check condition number
fn check_conditioning(w: &Array2<f64>) -> f64 {
    // Estimate condition number (naive)
    let norm_w = w.iter().map(|x| x*x).sum::<f64>().sqrt();
    norm_w  // Very rough estimate
}

// 2. Scale matrix
let w_scaled = w / w.iter().map(|x| x.abs()).fold(0.0, f64::max);

// 3. Check dimension
let (d, _) = w.dim();
if d > 500 {
    return Err("Matrix dimension > 500 not supported".into());
}

// 4. Ensure values are reasonable
let max_val = w.iter().map(|x| x.abs()).fold(0.0, f64::max);
if max_val > 1e6 {
    eprintln!("Warning: Very large weight values (max={})", max_val);
    // May indicate numerical instability
}
```

---

## Performance Issues

### "Benchmark taking much longer than expected"

**Causes:**
1. CPU frequency scaling / thermal throttling
2. Other processes using CPU
3. Debug build (unoptimized)

**Solutions:**

```bash
# Build with optimizations
cargo bench --release

# Disable frequency scaling (Linux)
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Pin to specific CPU cores
taskset -c 0,1 cargo bench --bench bench_end_to_end

# Check system load
top -b -n 1 | head -20
```

---

### "Memory usage is too high"

**Causes:**
1. Large data matrix (n×d)
2. Large L-BFGS history
3. Multiple solves in sequence without cleanup

**Solutions:**

```rust
// 1. Reduce L-BFGS memory
let mut config = OptimizationConfig::default();
config.lbfgs_memory = 5;  // from 10

// 2. Process data in batches
for data_batch in data_chunks {
    let standardized = standardize_data(&data_batch)?;
    let result = solve(&standardized, 0.1)?;
    // Process result and drop
}

// 3. Use smaller dimensions
let (n, d) = data.dim();
if d > 100 {
    // Consider feature selection or dimensionality reduction
}

// 4. Check memory usage
use std::alloc::GlobalAlloc;
// or use system monitoring
```

---

## Runtime Errors

### "Parsing error when loading configuration"

**Cause:** Invalid JSON/YAML in config file

**Solution:**
```rust
use serde_json;

fn load_config(path: &str) -> Result<OptimizationConfig, Box<dyn std::error::Error>> {
    let json = std::fs::read_to_string(path)?;
    let config: OptimizationConfig = serde_json::from_str(&json)
        .map_err(|e| {
            eprintln!("Config parse error: {}", e);
            e.into()
        })?;
    Ok(config)
}

// Example valid JSON:
let config_json = r#"{
    "max_outer_iterations": 100,
    "max_lbfgs_iterations": 50,
    "lbfgs_memory": 10,
    "constraint_tolerance": 1e-8,
    "penalty_rho_init": 1.0,
    "progress_rate": 0.25,
    "edge_threshold": 0.3
}"#;
```

---

### "Thread panic: index out of bounds"

**Cause:** Dimension mismatch or corrupted data

**Solution:**
```rust
// Add bounds checking
fn safe_matrix_access(w: &Array2<f64>, i: usize, j: usize) -> Result<f64, String> {
    let (d, _) = w.dim();
    if i >= d || j >= d {
        return Err(format!("Index ({},{}) out of bounds for {}×{} matrix", i, j, d, d));
    }
    Ok(w[[i, j]])
}

// Validate before operations
let (n, d) = data.dim();
let (w_d, w_d2) = w.dim();
assert_eq!(d, w_d, "Dimension mismatch");
assert_eq!(w_d, w_d2, "Weight matrix not square");
```

---

## FAQ

### Q: Should I standardize data?
**A:** Yes, highly recommended. It:
- Improves numerical stability
- Makes hyperparameters more transferable
- Enables fair comparison across variables
- Speeds up convergence

```rust
let standardized = standardize_data(&data)?;
let result = solve(&standardized, 0.1)?;
```

---

### Q: What λ should I use?
**A:** Start with 0.1 and tune:
- λ too small → too many edges, may not converge
- λ too large → too few edges, all zeros
- Use grid search or CV for data-driven selection

---

### Q: Can I parallelize NOTEARS?
**A:** Yes, but:
- Internal operations use rayon for parallelism
- Can parallelize over multiple λ values
- Cannot easily parallelize single solve

```rust
use rayon::prelude::*;

let lambdas = vec![0.01, 0.05, 0.1, 0.2];
let results: Vec<_> = lambdas.par_iter()
    .map(|&lambda| solve(&data, lambda))
    .collect();
```

---

### Q: How much data do I need?
**A:** Minimum n >> d recommended
- General guideline: n ≥ 10d
- Can work with n < d (underdetermined) but need strong regularization

---

### Q: How do I know the solution is correct?
**A:** Check:
1. Acyclicity: h(W) < tolerance (< 1e-8)
2. Edges: reasonable number (not all zero, not too many)
3. Convergence: iterations < max_outer_iterations
4. Reproducibility: same config → same result

---

## Getting Help

1. Check [API Reference](docs/API.md) and [Configuration Guide](docs/CONFIGURATION.md)
2. Review [examples/](examples/) for working code
3. Run tests to verify installation: `cargo test`
4. Open issue on [GitHub](https://github.com/pristley/notears/issues)

Include when reporting issues:
- NOTEARS version: `cargo tree | grep notears`
- Rust version: `rustc --version`
- Reproducible code example
- Error message (full output)
- System info (OS, CPU)
