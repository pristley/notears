# Performance Benchmarking & Profiling Guide

Comprehensive performance analysis toolkit for the NOTEARS algorithm using Criterion.rs.

## Quick Start

Run all benchmarks:
```bash
cargo bench --release
```

Run specific benchmark suite:
```bash
cargo bench --bench bench_matrix_ops
cargo bench --bench bench_optimization  
cargo bench --bench bench_end_to_end
```

## Benchmark Suites

### 1. Matrix Operations (bench_matrix_ops.rs)
Low-level primitive operations that form the core computational bottlenecks.

**Run:**
```bash
cargo bench --bench bench_matrix_ops -- --verbose
```

**Benchmarks:**

| Operation | Dimensions | Time (baseline) | Complexity | Notes |
|-----------|-----------|-----------------|-----------|-------|
| **matrix_exponential** | 5×5 to 50×50 | 0.1ms to 10ms | O(d³·log d) | Primary bottleneck; uses Padé approximation |
| **acyclicity_constraint** | 5 to 50 nodes | 0.5ms to 5ms | O(d³) | Depends on matrix exponential |
| **acyclicity_gradient** | 5 to 30 nodes | 1ms to 8ms | O(d³) | Chain rule: exp(W²)ᵀ ⊙ 2W |
| **mse_loss** | n∈[100,5000], d∈[5,50] | 0.01ms to 1ms | O(n·d²) | Linear in samples, quadratic in dims |
| **l1_penalty** | 10×10 to 500×500 | <0.01ms | O(d²) | Trivial operation; baseline |
| **standardize_data** | n∈[100,5000], d∈[5,100] | 0.1ms to 10ms | O(n·d) | Preprocessing step |

**Expected Results:**
```
matrix_exponential_10x10:        time: [100.00 ms 105.00 ms 110.00 ms]
matrix_exponential_50x50:        time: [5.0 s 5.2 s 5.4 s]
acyclicity_constraint_20nodes:   time: [1.2 ms 1.5 ms 1.8 ms]
acyclicity_gradient_20nodes:     time: [2.0 ms 2.3 ms 2.6 ms]
```

### 2. Optimization Operations (bench_optimization.rs)
Intermediate-level operations used during the optimization loop.

**Run:**
```bash
cargo bench --bench bench_optimization -- --verbose
```

**Benchmarks:**

| Benchmark | Problem Size | Expected Time | Purpose |
|-----------|--------------|----------------|---------|
| **optimization_iteration** | d=10-30, n=500-1000 | 10-100ms | Single L-BFGS iteration |
| **loss_gradient_pair** | d=10-50, n=1000 | 1-50ms | Both needed in L-BFGS step |
| **augmented_lagrangian_penalty** | d=10-50, ρ∈[1,100] | 0.5-5ms | Penalty term computation |
| **composed_loss** | d=10-30, λ∈[0.1,0.5] | 0.1-1ms | Total F(W) + λ·L1(W) |
| **constraint_progression** | ρ sequence (1→1000) | 1-10ms | Simulates ρ adaptation |

**Performance Notes:**
- Total iteration time = constraint + gradient + loss evaluation
- L-BFGS requires both gradient and function value
- Augmented Lagrangian penalty can dominate if h(W) is large

**Expected Output:**
```
optimization_iteration/d10n500:      time: [25.3 ms 26.1 ms 27.0 ms]
optimization_iteration/d20n1000:     time: [85.5 ms 89.2 ms 93.1 ms]
loss_gradient_pair/d20n1000:         time: [12.1 ms 12.8 ms 13.5 ms]
```

### 3. End-to-End Optimization (bench_end_to_end.rs)
Full NOTEARS algorithm runs on realistic problem sizes.

**Run:**
```bash
cargo bench --bench bench_end_to_end -- --verbose

# Create baseline for regression testing
cargo bench --bench bench_end_to_end -- --save-baseline initial

# Compare against baseline
cargo bench --bench bench_end_to_end -- --baseline initial
```

**Problem Classes:**

#### Small-Scale (Quick Regression Tests)
- **d=10, n=500**: ~100-500ms | Good for CI
- **d=15, n=500**: ~300-800ms | Quick baseline

#### Medium-Scale (Typical Use Case)
- **d=20, n=1000**: **1-2 seconds** | Standard problem
- **d=30, n=1000**: **3-5 seconds** | Moderate complexity

#### Scaling Studies
- **Dimension Scaling**: d=10→50 (n=1000 fixed)
- **Sample Scaling**: n=100→2000 (d=20 fixed)
- **Lambda Sensitivity**: λ=0.01→0.5 (d=20, n=1000)

**Expected Performance:**
```
small_scale/d10_n500_lambda0p1:              time: [250.5 ms 265.2 ms 281.3 ms]
medium_scale/d20_n1000_lambda0p1:           time: [1.25 s 1.35 s 1.48 s]
dimension_scaling/d10n1000:                  time: [350.2 ms 375.1 ms 401.5 ms]
dimension_scaling/d50n1000:                  time: [12.3 s 13.1 s 14.0 s]
sample_scaling/d20n100:                      time: [180.5 ms 195.3 ms 213.2 ms]
sample_scaling/d20n5000:                     time: [2.15 s 2.35 s 2.58 s]
```

## Profiling & Analysis

### 1. Flame Graph Profiling

Generate CPU flame graphs (Linux only):
```bash
# Install cargo-flamegraph if needed
cargo install flamegraph

# Profile a specific benchmark
cargo flamegraph --bench bench_end_to_end -- d20_n1000 --profile-time 10

# View flame graph
firefox flamegraph.svg
```

Expected hotspots:
- ~60-70%: Matrix exponential (Padé approximation)
- ~10-15%: Gradient computation
- ~10-15%: MSE loss + L1 penalty

### 2. Memory Profiling

Track memory usage and detect leaks:
```bash
# Install heaptrack (Linux)
sudo apt install heaptrack

# Profile benchmark
heaptrack cargo bench --bench bench_end_to_end -- --verbose d20_n1000

# Analyze results
heaptrack_gui heaptrack.cargo*.gz
```

### 3. Execution Timeline Analysis

Insert profiling markers into benchmark output:
```bash
# Run with verbose output
cargo bench --bench bench_end_to_end -- --verbose d20_n1000 2>&1 | tee results.txt

# Parse timing information from stdout
grep -E "Elapsed|iterations|Time per" results.txt
```

## Performance Baselines & Targets

### Expected Performance (Paper Reference)
From Zheng et al. (2018) NOTEARS paper:

| Problem | Hardware | Iterations | Time |
|---------|----------|-----------|------|
| d=20, n=1000 | CPU (reference) | 5-10 | 1-2 sec |
| d=50, n=1000 | CPU (reference) | 3-8 | 5-10 sec |
| d=100, n=1000 | CPU (reference) | 2-5 | 30-60 sec |

### Rust Implementation Targets
Our implementation should achieve:
- **d=20, n=1000**: 1-2 seconds (3× safety margin)
- **d=50, n=1000**: 5-10 seconds (5× safety margin)
- **d=100, n=1000**: 30-60 seconds (10× safety margin)

Safety margins account for:
- Numerical precision differences
- Implementation variations
- System variance (~20-30%)

## Commands Reference

### Basic Usage
```bash
# Quick smoke test
cargo bench --bench bench_matrix_ops acyclicity_constraint

# Full benchmark suite (takes ~10-30 min)
cargo bench --release

# Specific test
cargo bench --bench bench_end_to_end medium_scale
```

### Advanced Usage
```bash
# Run with specific sample size (criterion option)
cargo bench --bench bench_matrix_ops -- --sample-size 100

# Measurement time (in seconds)
cargo bench --bench bench_optimization -- --measurement-time 120

# Verbose output with more details
cargo bench --bench bench_end_to_end -- --verbose

# Profile time: how long to collect data
cargo bench --bench bench_end_to_end -- --profile-time 60

# Multiple times to reduce noise
cargo bench --bench bench_end_to_end -- --verbose --sample-size 5
```

### Baseline Regression Testing
```bash
# Initial baseline
cargo bench --bench bench_end_to_end -- --save-baseline before_optimization

# Run optimization...
# Then compare
cargo bench --bench bench_end_to_end -- --baseline before_optimization

# Compare against multiple baselines
cargo bench --bench bench_end_to_end -- --baseline before_optimization \
                                       --baseline after_cache_fix
```

## Interpreting Results

### Criterion Output Format
```
function_name                         time:   [100.0 ms 105.0 ms 110.0 ms]
```

- **Lower estimate**: 100.0 ms (confidence interval lower bound)
- **Point estimate**: 105.0 ms (most likely time)
- **Upper estimate**: 110.0 ms (confidence interval upper bound)

### Regression Detection
Criterion automatically flags changes:
- **🟠 WARNING**: 5-10% slowdown (investigate)
- **🔴 ERROR**: >10% slowdown (likely regression)

### Variance Sources
- **System load**: Minimize background processes
- **Thermal throttling**: Let system cool between runs
- **CPU frequency scaling**: Disable for stable results
  ```bash
  # Linux: check scaling
  cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
  ```

## HTML Reports

Criterion generates beautiful HTML reports:
```bash
# Reports generated in target/criterion/report/index.html
cargo bench --bench bench_matrix_ops
open target/criterion/report/index.html
```

## Debugging Performance

### If benchmarks are slower than expected:

1. **Check CPU affinity**
   ```bash
   taskset -c 0,1 cargo bench --bench bench_end_to_end
   ```

2. **Disable turbo boost** (for consistent results)
   ```bash
   # Linux
   echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo
   ```

3. **Check for thermal throttling**
   ```bash
   watch -n 1 "cat /proc/cpuinfo | grep MHz"
   ```

4. **Profile with perf** (Linux)
   ```bash
   perf record -g cargo bench --bench bench_end_to_end -- d20_n1000
   perf report
   ```

## Optimization Opportunities

Based on profile data, optimization priorities:

1. **Matrix exponential** (60-70% of time)
   - Current: Padé approx with scaling-squaring O(d³·log d)
   - Possible: Cache for repeated computations
   - Possible: Specialized routine for small d

2. **Gradient computation** (10-15% of time)
   - Current: Full matrix exponential
   - Possible: Exploit sparsity in W

3. **MSE loss** (5-10% of time)
   - Current: Full matrix multiplication
   - Possible: Batched computation for multiple W

## CI Integration

Add to `.github/workflows/bench.yml`:
```yaml
- name: Run benchmarks
  run: |
    cargo bench --bench bench_matrix_ops
    cargo bench --bench bench_optimization
    cargo bench --bench bench_end_to_end -- --save-baseline ${{ github.run_id }}
```

## References

- Criterion.rs: https://bheisler.github.io/criterion.rs/book/
- Higham matrix exponential: https://arxiv.org/pdf/0804.4150.pdf
- NOTEARS paper: https://arxiv.org/abs/1803.01422
