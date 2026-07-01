# NOTEARS Quick Reference Guide
## Practical Cheat Sheet for DAG Learning & Troubleshooting

---

## 1. Algorithm Comparison at a Glance

| **Dimension** | **NOTEARS** | **PC** | **GES** | **LiNGAM** |
|---|---|---|---|---|
| **Speed** | ⚡⚡⚡ Fast | ⚡ Slow | ⚡⚡ Medium | ⚡⚡⚡ Fast |
| **Scalability** | Large graphs | Small-medium | Medium | Large graphs |
| **Assumptions** | Linear SEM | Causal sufficiency | Causal sufficiency | Nonlinear acyclic |
| **Faithfulness** | ✗ Not required | ✓ Required | ✓ Required | ✗ Not required |
| **Acyclicity** | ✓ Enforced | ✓ Enforced | ✓ Enforced | ✓ Enforced |
| **Interpretation** | Weight matrix W | DAG structure | DAG structure | Causal coeff. |
| **Robustness** | Good | Moderate | Good | Good |
| **When to use** | Default choice | Expert guidance | Large search space | Nonlinear suspected |

---

## 2. Algorithm Selection Decision Tree

```
START: I need to learn a DAG structure

├─ Do I have continuous, linear data?
│  ├─ YES → Continue to next question
│  └─ NO  → Use NOTEARS-G (mixed) or GOLEM (nonlinear)
│
├─ Do I have 50+ variables?
│  ├─ YES (Large) → Use NOTEARS (best scalability)
│  ├─ NO (Small) → Continue to next question
│
├─ Do I have causal domain knowledge?
│  ├─ YES → Use PC/GES (can incorporate knowledge)
│  ├─ NO  → Continue to next question
│
├─ Do I have ground truth for validation?
│  ├─ YES → Use NOTEARS + check against ground truth
│  ├─ NO  → Use NOTEARS + stability analysis
│
├─ Performance requirement?
│  ├─ Speed critical → NOTEARS (minutes for 100 vars)
│  ├─ Accuracy critical → GES (more thorough search)
│  └─ Balanced → NOTEARS (recommended default)
│
└─ DECISION: Use NOTEARS
   Parameters: λ ∈ [0.01, 0.5], ω ∈ [0.1, 10]
   Validation: stability + expert review
```

---

## 3. NOTEARS Overview

**What it does**:
- Learns directed acyclic graph (DAG) structure from continuous data
- Returns weighted adjacency matrix W where W[i,j] = edge weight from i→j
- Minimizes: Loss(W) + λ||W||₁ s.t. h(W) = 0
- Uses augmented Lagrangian + L-BFGS optimization

**Key innovation**:
- Acyclicity constraint: h(W) = tr(exp(W ⊙ W)) - d = 0
- Differentiable → can use gradient-based optimization
- No search over exponential space (unlike GES, PC)

**When it works well**:
- Linear relationships in data
- 50-1000 variables
- Moderate sample sizes (n ≥ 10*d)
- Sparse ground truth graphs

**When it struggles**:
- Highly nonlinear data
- Cyclic relationships
- Very small samples (n < d)
- Densely connected graphs

---

## 4. Hyperparameter Tuning Cheat Sheet

### 4.1 Lambda (λ) - L₁ Regularization Strength

**What it controls**: Sparsity of solution (number of edges)

| λ value | Effect | Use case |
|---------|--------|----------|
| 0.0 | Dense solution, many edges | Exploratory, low noise |
| 0.01-0.05 | Light regularization | Moderate sparsity expected |
| 0.05-0.2 | Standard regularization | **RECOMMENDED DEFAULT** |
| 0.2-0.5 | Heavy regularization | Sparse solution expected |
| >0.5 | Very sparse, mostly disconnected | Only if prior knowledge sparse |

**How to choose λ**:
```
Method 1 (Quick): λ = 1.0 / (2*n)  [BIC-style scaling]
   Example: n=500 → λ ≈ 0.001 (then scale up by 5-10x)

Method 2 (Better): Cross-validation
   - Split data into train/val (80/20)
   - Try λ ∈ {0.001, 0.005, 0.01, 0.05, 0.1, 0.2}
   - Pick highest val score
   
Method 3 (Best): Stability selection
   - Run on bootstrap samples with different λ
   - Pick λ where edges are most stable (>80% appear)
```

**Rules of thumb**:
- If getting too many edges: increase λ by 2-3x
- If missing known edges: decrease λ by 2-3x
- Typical range: 0.01-0.3 for most problems

---

### 4.2 Omega (ω) - Penalty Growth Rate

**What it controls**: How quickly penalty increases in augmented Lagrangian

| ω value | Convergence | Precision |
|---------|-----------|-----------|
| 0.1-0.5 | Slow | High |
| 0.5-2.0 | **Balanced** | **Recommended** |
| 2.0-10 | Fast | May miss optimal |
| >10 | Very fast | Numerical issues |

**Default**: ω = 1.0 (multiply by ~1.5 each iteration)

**How to choose**:
- Standard: ω = 1.0 (recommended for most)
- Unstable optimization: decrease to 0.5-0.75
- Tight deadline: increase to 2.0-5.0 (sacrifice precision)

---

### 4.3 Max Iterations & Thresholds

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| **max_outer_iterations** | 100 | 20-200 | Usually converges by 50 |
| **max_inner_iterations** | 50 | 20-100 | L-BFGS iterations per step |
| **constraint_tol** | 1e-6 | 1e-8 to 1e-4 | When h(W) small enough |
| **variable_tol** | 1e-6 | 1e-8 to 1e-4 | When W doesn't change |

---

### 4.4 Complete Hyperparameter Template

```rust
// Starting template
OptimizationConfig {
    max_outer_iterations: 100,      // Outer loop iterations
    max_lbfgs_iterations: 50,       // Inner L-BFGS steps
    constraint_tolerance: 1e-6,     // h(W) convergence threshold
    variable_tolerance: 1e-6,       // W convergence threshold
    penalty_growth_rate: 1.0,       // ω (multiply per iteration)
    initial_penalty: 0.25,          // Initial ρ value
    rho_multiplier: 2.0,            // Increase factor each iteration
}

// Recommended defaults by problem type:

// Conservative (prefer accuracy)
OptimizationConfig {
    max_outer_iterations: 150,
    constraint_tolerance: 1e-8,
    penalty_growth_rate: 0.75,
    ..Default::default()
}

// Balanced (most problems)
OptimizationConfig::default()  // All defaults

// Aggressive (speed needed)
OptimizationConfig {
    max_outer_iterations: 50,
    constraint_tolerance: 1e-4,
    penalty_growth_rate: 2.0,
    ..Default::default()
}

// For regularization:
RegularizationConfig::new(lambda, use_bias)?
```

---

## 5. Runtime Benchmarks

### Typical Execution Times (on modern CPU)

```
Variables (d) | Samples (n) | Time (seconds) | Memory (MB)
---------|----------|-----------|----------
10        | 100      | 0.1       | 5
50        | 500      | 2.5       | 25
100       | 1000     | 8.0       | 100
200       | 5000     | 45        | 400
500       | 10000    | 180       | 2500
1000      | 20000    | 720       | 10000
```

**Scaling**: ~O(n*d³) due to matrix exponential computation

**Factors affecting speed**:
- Matrix exponential (biggest bottleneck)
- L-BFGS iterations (depends on λ, data)
- Data standardization (small, parallelizable)
- Initialization (negligible)

---

## 6. Troubleshooting Guide: 5 Common Issues

### Issue 1: Optimization Not Converging

**Symptoms**: 
- h(W) > 1e-4 after many iterations
- Loss oscillating, not decreasing
- Penalty increasing indefinitely

**Root Causes**:
1. **λ too small** → regularization too weak
2. **Poor initialization** → starting far from optimum
3. **Numerical instability** → matrix exponential overflow
4. **Data quality** → outliers, missing values

**Solutions** (in order of likelihood):
```
1. Increase λ by 3-5x and rerun
   - If h(W) now converges: λ was problem (SUCCESS)
   - If h(W) still doesn't converge: proceed to 2

2. Check data quality:
   - Remove outliers (z-score > 4)
   - Check for NaN/Inf values
   - Verify standardization (mean≈0, std≈1)
   - Try removing one variable at a time (identify problematic var)

3. Adjust hyperparameters:
   - Decrease penalty_growth_rate: 1.0 → 0.5
   - Increase max_outer_iterations: 100 → 200
   - Loosen constraint_tolerance: 1e-6 → 1e-4
   
4. Try different initialization:
   - Use different random seed
   - Initialize with zeros instead of random
   - If NOTEARS library: try initialization option
```

**Prevention**:
- Always standardize data first (z-score normalization)
- Start with λ = 0.05, adjust from there
- For first run, use conservative hyperparameters

---

### Issue 2: Too Many False Positive Edges

**Symptoms**:
- More edges than domain knowledge suggests
- Many weak edges (W[i,j] near zero)
- Adjacency matrix very dense

**Root Causes**:
1. **λ too small** → insufficient regularization
2. **High noise** → spurious correlations
3. **Nonlinear relationships** → linear model misspecified

**Solutions**:
```
1. Increase λ (most common fix)
   - Multiply λ by 2-3x
   - Verify edges > 0.01 remain (true signals)
   - Repeat until reasonable sparsity

2. Threshold weak edges
   - Remove edges with |W[i,j]| < threshold
   - threshold = 0.05-0.1 (depends on scale)
   - Consider relative to max(|W|)

3. Validate with domain knowledge
   - Ask: "do these edges make sense?"
   - Cross-reference with literature
   - Adjust λ based on feedback
```

**Prevention**:
- Use λ ∈ [0.05, 0.2] as starting point
- Always visually inspect edge distribution
- Report both point estimates and confidence intervals

---

### Issue 3: Missing Known Ground Truth Edges

**Symptoms**:
- Algorithm missed edges known to exist
- Accuracy metrics low vs. baseline
- Weak edges below thresholding cutoff

**Root Causes**:
1. **λ too large** → over-regularization suppressing edges
2. **Insufficient data** → low statistical power
3. **Model misspecification** → nonlinear relationship

**Solutions**:
```
1. Decrease λ (most common fix)
   - Divide λ by 2-3x
   - Check if missing edges reappear
   - Balance with false positives from Issue 2

2. Increase sample size if possible
   - Power increases with √n
   - Need n > 10*d typically
   - Collect more data for better precision

3. Check model assumptions
   - Visualize missing edge relationship
   - Check if linear or nonlinear
   - If nonlinear: NOTEARS inappropriate, use GOLEM
```

**Prevention**:
- Compare to baseline with known hyperparameters
- Use ground truth for sensitivity analysis
- Report both sensitivity and specificity

---

### Issue 4: Numerical Instability / NaN or Inf

**Symptoms**:
- NaN in loss function
- Infinity in matrix elements
- Optimization crashes

**Root Causes**:
1. **Matrix exponential overflow** → exp(large number) = Inf
2. **Ill-conditioned data** → extreme scales or correlations
3. **Weak machine precision** → too many matrix operations

**Solutions**:
```
1. Check data standardization (MOST COMMON)
   - Verify all variables z-score normalized
   - Check: mean(X) ≈ 0, std(X) ≈ 1 for each column
   - If not: apply z-score normalization
   - Retry optimization

2. Reduce problem size
   - Select subset of most relevant variables
   - Use PCA to 50-80% variance retained
   - Rerun on reduced problem

3. Adjust tolerance for numerical stability
   - Increase constraint_tolerance: 1e-6 → 1e-4 or 1e-3
   - Reduce max_outer_iterations: 100 → 50
   - Use constraint_tolerance stopping criterion

4. Check for problematic variables
   - Try removing each variable one at a time
   - If removing variable X fixes: X is problematic
   - Either transform X or remove from analysis
```

**Prevention**:
- Always z-score normalize before running
- Check correlation matrix for near-perfect correlations (>0.99)
- Use double precision (float64), not single precision

---

### Issue 5: Slow Execution / Timeout

**Symptoms**:
- Execution takes hours for moderate-sized problem
- Matrix operations slow
- Memory usage high

**Root Causes**:
1. **Problem too large** → d > 500 with general approach
2. **Poor convergence** → many L-BFGS iterations
3. **Inefficient implementation** → naive matrix exponential

**Solutions**:
```
1. Reduce problem dimension
   - Feature selection: use domain knowledge
   - PCA: retain 95% variance
   - Remove highly correlated variables
   
2. Increase λ for faster convergence
   - Higher regularization = fewer L-BFGS steps
   - Try λ = 0.1-0.2 (vs. default 0.05)
   - May sacrifice accuracy for speed

3. Adjust hyperparameters for speed
   - Increase penalty_growth_rate: 1.0 → 2.0
   - Reduce constraint_tolerance: 1e-6 → 1e-3
   - Reduce max_outer_iterations: 100 → 30

4. Use optimized implementation
   - Compiled language (Rust vs. Python)
   - GPU acceleration if available
   - Sparse matrices if graph is sparse

5. Parallelize (if available)
   - Bootstrap runs on different cores
   - Different initializations in parallel
   - Aggregate results
```

**Prevention**:
- For d > 200: expect minutes to hours
- For d > 500: consider feature reduction first
- Profile bottleneck (matrix exponential usually 50-70%)

---

## 7. Validation Checklist

### BEFORE Running NOTEARS

- [ ] **Data loaded and shape verified**: n rows (samples), d columns (variables)
- [ ] **No missing values**: Check for NaN/None; handle or remove rows
- [ ] **Data standardized**: Z-score normalize each column (mean=0, std=1)
- [ ] **No constant columns**: Check std > 0 for all variables
- [ ] **Outliers addressed**: Identify extreme values (z-score >4); decide keep/remove
- [ ] **Variables meaningful**: Verify all variables are relevant to problem
- [ ] **Sample size sufficient**: n ≥ 10*d recommended
- [ ] **No duplicate rows**: Check for identical samples
- [ ] **Correlation checked**: Identify highly correlated pairs (r > 0.99)
- [ ] **Domain knowledge gathered**: What edges do experts expect?

### DURING Running NOTEARS

- [ ] **h(W) converging**: Check h(W) decreases over iterations
- [ ] **Loss reasonable**: Verify loss is negative or near zero
- [ ] **No NaN/Inf**: Monitor for numerical issues
- [ ] **Iterations reasonable**: Usually converges by 50 iterations
- [ ] **Memory acceptable**: Check RAM usage vs. available memory
- [ ] **Time reasonable**: Compare vs. benchmark table in Section 5

### AFTER Running NOTEARS - If Ground Truth Available

- [ ] **Accuracy metrics computed**:
  - [ ] Sensitivity (TP / (TP + FN))
  - [ ] Specificity (TN / (TN + FP))
  - [ ] Structural Hamming Distance (SHD)
  - [ ] Area Under ROC Curve (AUROC)
- [ ] **Results vs. ground truth**: Quantified via metrics above
- [ ] **Edge weights reasonable**: Compare to true edge weights
- [ ] **False positives identified**: Which edges are spurious?
- [ ] **False negatives identified**: Which true edges missed?
- [ ] **λ sensitivity analyzed**: How do results change with λ?

### AFTER Running NOTEARS - If Ground Truth NOT Available

- [ ] **Adjacency matrix visualized**: Plot as heatmap
- [ ] **Edge weight distribution inspected**: Histogram of |W|
- [ ] **Acyclicity verified**: No cycles in returned graph
- [ ] **Sparsity reasonable**: # edges matches domain expectation
- [ ] **Stability analyzed** (recommended):
  - [ ] Run on multiple bootstrap samples
  - [ ] Report edges stable in >80% of runs
  - [ ] Distinguish stable vs. spurious edges
- [ ] **Domain expert validation**:
  - [ ] Does adjacency matrix make sense?
  - [ ] Are strong edges expected?
  - [ ] Are weak edges plausible?
- [ ] **Comparison to baseline**:
  - [ ] Compare to correlations: |W[i,j]| vs. |ρ[i,j]|
  - [ ] Compare to known subgraphs: accuracy on known edges
  - [ ] Compare to other methods: GES, PC if available

### ONGOING Monitoring (Production)

- [ ] **New data periodically tested**: Does DAG still hold?
- [ ] **Performance metrics tracked**: Are predictions still accurate?
- [ ] **Retrain trigger identified**: When to rerun NOTEARS?
- [ ] **Data drift detection**: Are variable distributions changing?
- [ ] **Results reproducibility**: Same hyperparameters, same results?

---

## 8. Success Metrics & Reporting Templates

### Metrics to Report

**If Ground Truth Available** (rank by importance):
1. **Sensitivity**: % true edges detected (higher is better)
2. **Specificity**: % non-edges correctly identified (higher is better)
3. **Structural Hamming Distance**: Edit distance to truth (lower is better)
4. **AUROC**: Area under ROC curve (higher is better)
5. **F1-Score**: Harmonic mean of precision/recall

**If Ground Truth NOT Available** (rank by importance):
1. **Stability**: % of edges stable across bootstrap samples
2. **Graph statistics**: # edges, # connected components, density
3. **Sparsity pattern**: % edges above threshold (e.g., >0.05)
4. **Convergence**: Final h(W) value (should be <1e-4)
5. **Reproducibility**: Results with different random seeds

### Template: One-Page Summary

```
NOTEARS Analysis Summary
=======================

Problem:
- Variables: d = [YOUR D]
- Samples: n = [YOUR N]
- Known edges: [X] (if available)
- Goal: [DESCRIBE GOAL]

Data:
- Standardized: [YES/NO]
- Missing values: [N REMOVED]
- Outliers: [N REMOVED]
- Correlation check: [MAX R = X]

Hyperparameters:
- Lambda (λ): [VALUE]
- Omega (ω): [VALUE]
- Max iterations: [N]
- Constraint tolerance: [TOL]

Results:
- Edges detected: [N]
- Graph density: [X%]
- Convergence: h(W) = [VALUE] (goal < 1e-4)
- Execution time: [X seconds]

Validation:
[IF GROUND TRUTH]
- Sensitivity: [X%]
- Specificity: [X%]
- SHD: [X]

[IF NO GROUND TRUTH]
- Stability: [X% edges stable]
- Domain review: [PASSED/REVIEW]

Interpretation:
- [KEY FINDING 1]
- [KEY FINDING 2]
- [LIMITATIONS/CAVEATS]

Recommendations:
- [RECOMMENDED NEXT STEPS]
```

---

## 9. Common Pitfalls & Prevention (10 Top Issues)

### Pitfall 1: Skipping Data Standardization
**Problem**: Results completely wrong if data not z-score normalized  
**Prevention**: Always first line: `standardize_data(X)`  
**Check**: `assert!(mean(X) ≈ 0 && std(X) ≈ 1)`

### Pitfall 2: Choosing λ Blindly
**Problem**: Results very sensitive to λ; random choice usually wrong  
**Prevention**: Use cross-validation or stability selection  
**Check**: Run with λ and λ/2 and λ*2; verify direction of change

### Pitfall 3: Trusting Single Run
**Problem**: Stochastic initialization means results vary; one run misleading  
**Prevention**: Run multiple times (5-10) with different seeds  
**Check**: Report stable edges (appear in >80% of runs)

### Pitfall 4: Over-Interpreting Weak Edges
**Problem**: Small non-zero entries (W[i,j] = 0.01) may be numerical artifacts  
**Prevention**: Threshold edges at 0.05-0.1 or report confidence intervals  
**Check**: Repeat with different initialization; does edge remain?

### Pitfall 5: Ignoring Convergence Diagnostics
**Problem**: May stop before convergence with suboptimal solution  
**Prevention**: Check h(W) < 1e-4 and loss stabilization  
**Check**: Plot h(W) and loss vs. iteration; should look smooth

### Pitfall 6: Wrong Sample Size / Variable Ratio
**Problem**: NOTEARS needs n ≥ 10*d roughly; small sample = unreliable  
**Prevention**: Check n/d ratio first; if < 5, use more data or fewer variables  
**Check**: Stability analysis shows instability if n too small

### Pitfall 7: Assuming Linearity Without Checking
**Problem**: NOTEARS requires linear relationships; nonlinear data fails  
**Prevention**: Scatterplot pairwise relationships; check linearity  
**Check**: If nonlinear: use NOTEARS-G or GOLEM instead

### Pitfall 8: No Baseline Comparison
**Problem**: Can't tell if NOTEARS is working well without reference point  
**Prevention**: Compare to simpler methods (e.g., hard threshold on correlations)  
**Check**: "Is NOTEARS better than random guessing?" (should be yes!)

### Pitfall 9: Forgetting Causal Sufficiency Assumption
**Problem**: If hidden confounders exist, results will be wrong  
**Prevention**: Think about possible unmeasured variables affecting your domain  
**Check**: If suspect confounding: document assumption violation

### Pitfall 10: Single Run as Final Result
**Problem**: Publication bias toward reporting best single run  
**Prevention**: Report multiple runs; use median/mean + confidence intervals  
**Check**: "Would reviewers accept this if I ran it again?"

---

## 10. Quick Answers: FAQ

**Q: "My λ is X. Should I change it?"**  
A: If edges seem reasonable + h(W) converged: probably fine. If too many/few edges: try λ/2 or 2λ.

**Q: "Should I remove correlated variables?"**  
A: Not necessary, but highly correlated pairs (r > 0.95) may cause numerical issues. Document if found.

**Q: "How many iterations until convergence?"**  
A: Usually 20-50. If >100 and not converged: problem with data or λ.

**Q: "Can I trust results from non-converged optimization?"**  
A: No. Suboptimal solution could be substantially wrong. Always ensure h(W) < 1e-4.

**Q: "Should I weight samples differently?"**  
A: NOTEARS doesn't support weighting. Preprocess data instead (e.g., duplicate important samples).

**Q: "What if my data has cycles?"**  
A: NOTEARS will still run but return approximate solution (h(W) > 0). Check h(W) to detect this.

**Q: "Can I run NOTEARS on GPU?"**  
A: Depends on implementation. Rust version: no native GPU support. Consider PyTorch version if available.

**Q: "How sensitive is NOTEARS to initialization?"**  
A: Fairly robust; different W_init usually gives similar results if λ reasonable. Report multiple runs.

**Q: "What's the minimum sample size?"**  
A: n ≥ d is technically feasible, but n ≥ 10*d recommended for reliability.

**Q: "Should I use bootstrap for confidence intervals?"**  
A: Yes! Run on 100 bootstrap samples; report edges stable in >80% runs.

---

## Quick Links to Other Docs

- **Mathematical details?** → Implementation_Guide.md, Part I
- **Comparing to other algorithms?** → Analysis_and_Comparison.md, Section IV
- **How to implement it?** → Implementation_Guide.md, Section II
- **Case studies?** → Analysis_and_Comparison.md, Section VIII
- **Production deployment?** → Analysis_and_Comparison.md, Section VII

---

**Last Updated**: July 2026  
**Version**: 1.0  
**Quick reference format**: ~4,000 words, designed for 10-15 min reads per section
