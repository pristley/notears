/// Shared profiling utilities for benchmarking suite
///
/// Provides helper functions for:
/// - Data generation (DAGs, synthetic data)
/// - Timing instrumentation
/// - Performance metrics collection
use ndarray::Array2;
use rand::Rng;
use std::time::Instant;

/// Generate a random acyclic weight matrix (DAG in lower-triangular form)
///
/// Creates a lower-triangular weight matrix to guarantee acyclicity,
/// with random edge weights and configurable density.
pub fn random_dag(d: usize, edge_density: f64) -> Array2<f64> {
    let mut rng = rand::thread_rng();
    let mut w = Array2::<f64>::zeros((d, d));

    // Lower triangular ensures acyclicity
    for i in 0..d {
        for j in 0..i {
            if rng.gen::<f64>() < edge_density {
                w[[i, j]] = rng.gen::<f64>() * 2.0 - 1.0; // weight ∈ [-1, 1]
            }
        }
    }

    w
}

/// Generate random standard normal data matrix
pub fn random_data(n: usize, d: usize) -> Array2<f64> {
    let mut rng = rand::thread_rng();
    let mut data = Array2::<f64>::zeros((n, d));

    for i in 0..n {
        for j in 0..d {
            data[[i, j]] = rand_normal(&mut rng);
        }
    }

    data
}

/// Generate data from linear structural equation model (SEM)
///
/// Creates data from: X_i = sum_j W_{ij} * X_j + noise_i
/// where W is acyclic and noise is standard normal.
pub fn data_from_sem(n: usize, d: usize, w: &Array2<f64>, noise_scale: f64) -> Array2<f64> {
    let mut rng = rand::thread_rng();
    let mut data = Array2::<f64>::zeros((n, d));

    // Generate exogenous noise
    for i in 0..n {
        for j in 0..d {
            data[[i, j]] = rand_normal(&mut rng) * noise_scale;
        }
    }

    // Apply linear transformation via lower-triangular W
    for i in 0..n {
        for node in 0..d {
            for parent in 0..node {
                data[[i, node]] += w[[node, parent]] * data[[i, parent]];
            }
        }
    }

    data
}

/// Generate Erdős–Rényi random DAG
///
/// Generates n_nodes nodes with random edges (probability edge_prob).
/// Ensures acyclicity via topological ordering.
pub fn generate_erdos_renyi_dag(
    n_nodes: usize,
    n_edges_expected: usize,
    n_samples: usize,
) -> (Array2<f64>, Array2<f64>) {
    let edge_prob = (n_edges_expected as f64) / ((n_nodes * n_nodes) as f64);
    let edge_prob = edge_prob.min(1.0).max(0.0);

    let w = random_dag(n_nodes, edge_prob);
    let data = data_from_sem(n_samples, n_nodes, &w, 1.0);

    (w, data)
}

/// Box-Muller transform for generating standard normal random variables
fn rand_normal<R: Rng>(rng: &mut R) -> f64 {
    let u1 = rng.gen::<f64>();
    let u2 = rng.gen::<f64>();
    let r = (-2.0 * u1.ln()).sqrt();
    let theta = 2.0 * std::f64::consts::PI * u2;
    r * theta.cos()
}

/// Simple timing instrumentation for profiling
pub struct TimingGuard {
    start: Instant,
    label: String,
}

impl TimingGuard {
    /// Create new timing guard with label
    pub fn new(label: &str) -> Self {
        TimingGuard {
            start: Instant::now(),
            label: label.to_string(),
        }
    }

    /// Return elapsed time and print to stdout
    pub fn finish(self) -> f64 {
        let elapsed = self.start.elapsed().as_secs_f64();
        println!("[{}] Elapsed: {:.3}s", self.label, elapsed);
        elapsed
    }
}

/// Performance metrics structure
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub total_time_s: f64,
    pub iterations: usize,
    pub time_per_iteration_ms: f64,
    pub throughput_iters_per_sec: f64,
}

impl PerformanceMetrics {
    /// Create metrics from timing data
    pub fn from_timing(total_time_s: f64, iterations: usize) -> Self {
        let time_per_iteration_ms = (total_time_s / iterations as f64) * 1000.0;
        let throughput_iters_per_sec = iterations as f64 / total_time_s;

        PerformanceMetrics {
            total_time_s,
            iterations,
            time_per_iteration_ms,
            throughput_iters_per_sec,
        }
    }

    /// Print metrics in tabular format
    pub fn print(&self) {
        println!(
            "  Total: {:.3}s | Per-iter: {:.2}ms | Throughput: {:.1} iter/s",
            self.total_time_s, self.time_per_iteration_ms, self.throughput_iters_per_sec
        );
    }
}
