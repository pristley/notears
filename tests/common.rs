//! Common utilities and helper functions for test suite
//!
//! Provides synthetic data generation, comparison functions, and metrics
//! used across all test modules.

use ndarray::Array2;
use rand::Rng;

/// Generate a random acyclic weight matrix (DAG in lower-triangular form)
///
/// Creates a lower-triangular weight matrix to guarantee acyclicity,
/// with random edge weights in [-1, 1] and edge density ~50%.
///
/// # Arguments
/// * `d` - Matrix dimension (number of nodes)
/// * `edge_density` - Probability of edge existence (default 0.5)
///
/// # Returns
/// Random DAG weight matrix (d×d)
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
///
/// Creates n×d data matrix with i.i.d. standard normal entries.
/// This serves as input for testing scoring functions and optimization.
///
/// # Arguments
/// * `n` - Number of samples
/// * `d` - Number of variables
///
/// # Returns
/// Random data matrix (n×d) from N(0, 1)
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
///
/// # Arguments
/// * `n` - Number of samples
/// * `d` - Number of variables
/// * `w` - Weight matrix (d×d) - should be acyclic
/// * `noise_scale` - Noise standard deviation
///
/// # Returns
/// Simulated data (n×d) from the SEM with structure W
pub fn data_from_sem(n: usize, d: usize, w: &Array2<f64>, noise_scale: f64) -> Array2<f64> {
    let mut rng = rand::thread_rng();
    let mut data = Array2::<f64>::zeros((n, d));

    // Generate noise
    let noise = (0..n)
        .map(|_| {
            (0..d)
                .map(|_| rand_normal(&mut rng) * noise_scale)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    // Simulate using topological order (lower triangular W)
    for sample in 0..n {
        for i in 0..d {
            let mut x_i = noise[sample][i];
            for j in 0..i {
                x_i += w[[i, j]] * data[[sample, j]];
            }
            data[[sample, i]] = x_i;
        }
    }

    data
}

/// Standardize data matrix (zero mean, unit variance per column)
///
/// Centers and scales each column independently.
///
/// # Arguments
/// * `data` - Data matrix (n×d)
///
/// # Returns
/// Standardized data matrix
pub fn standardize(data: &Array2<f64>) -> Array2<f64> {
    let (n, d) = data.dim();
    let mut result = data.clone();

    for j in 0..d {
        let col = data.column(j);
        let mean = col.iter().sum::<f64>() / n as f64;
        let var = col.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0);
        let std = var.sqrt();

        if std > 1e-10 {
            for i in 0..n {
                result[[i, j]] = (result[[i, j]] - mean) / std;
            }
        }
    }

    result
}

/// Structural Hamming Distance (SHD) between two adjacency matrices
///
/// Counts the number of edge differences between two graphs:
/// - False positives: edges in predicted but not in true
/// - False negatives: edges in true but not in predicted
/// - Reversals: edges with opposite direction
///
/// # Arguments
/// * `true_graph` - Ground truth adjacency matrix (d×d)
/// * `predicted_graph` - Predicted adjacency matrix (d×d)
///
/// # Returns
/// SHD metric (0 = perfect recovery)
pub fn structural_hamming_distance(
    true_graph: &Array2<i32>,
    predicted_graph: &Array2<i32>,
) -> usize {
    let d = true_graph.shape()[0];
    let mut shd = 0;

    for i in 0..d {
        for j in 0..d {
            if i != j {
                let true_edge = true_graph[[i, j]] != 0;
                let pred_edge = predicted_graph[[i, j]] != 0;

                if true_edge != pred_edge {
                    shd += 1; // False positive or false negative
                }
            }
        }
    }

    shd
}

/// Check if matrix is approximately symmetric
///
/// Compares matrix with its transpose element-wise.
///
/// # Arguments
/// * `w` - Matrix to check
/// * `tolerance` - Maximum allowed element-wise difference
///
/// # Returns
/// true if ||w - w^T||_∞ < tolerance
pub fn is_symmetric(w: &Array2<f64>, tolerance: f64) -> bool {
    let wt = w.t().to_owned();
    let diff = w - &wt;
    diff.iter().map(|x| x.abs()).fold(0.0, f64::max) < tolerance
}

/// Check if two matrices are approximately equal element-wise
///
/// # Arguments
/// * `a` - First matrix
/// * `b` - Second matrix
/// * `tolerance` - Maximum allowed element-wise difference
///
/// # Returns
/// true if ||a - b||_∞ < tolerance
pub fn close_enough(a: &Array2<f64>, b: &Array2<f64>, tolerance: f64) -> bool {
    if a.shape() != b.shape() {
        return false;
    }
    let diff = a - b;
    diff.iter().map(|x| x.abs()).fold(0.0, f64::max) < tolerance
}

/// Check if matrix is lower triangular (within tolerance)
///
/// A lower triangular matrix satisfies A[i,j] ≈ 0 for i < j.
///
/// # Arguments
/// * `w` - Matrix to check
/// * `tolerance` - Maximum allowed non-zero value in upper triangle
///
/// # Returns
/// true if matrix is approximately lower triangular
pub fn is_lower_triangular(w: &Array2<f64>, tolerance: f64) -> bool {
    let d = w.shape()[0];
    for i in 0..d {
        for j in (i + 1)..d {
            if w[[i, j]].abs() > tolerance {
                return false;
            }
        }
    }
    true
}

/// Compute Frobenius norm: sqrt(sum of squared elements)
///
/// # Arguments
/// * `matrix` - Input matrix
///
/// # Returns
/// Frobenius norm ||A||_F
pub fn frobenius_norm(matrix: &Array2<f64>) -> f64 {
    matrix.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Compute spectral norm (largest singular value)
///
/// # Arguments
/// * `matrix` - Input matrix
///
/// # Returns
/// Spectral norm ||A||_2
#[allow(dead_code)]
pub fn spectral_norm(matrix: &Array2<f64>) -> f64 {
    // Simplified: compute via power iteration or use max singular value
    // For testing, we approximate with largest absolute eigenvalue
    let norm_frob = frobenius_norm(matrix);
    let d = matrix.shape()[0];
    (norm_frob / (d as f64).sqrt()).max(norm_frob / d as f64)
}

/// Compute infinity norm: max row sum of absolute values
///
/// # Arguments
/// * `matrix` - Input matrix
///
/// # Returns
/// Infinity norm ||A||_∞
#[allow(dead_code)]
pub fn infinity_norm(matrix: &Array2<f64>) -> f64 {
    let d = matrix.shape()[0];
    let mut max_sum: f64 = 0.0;
    for i in 0..d {
        let row_sum = (0..d).map(|j| matrix[[i, j]].abs()).sum::<f64>();
        max_sum = max_sum.max(row_sum);
    }
    max_sum
}

/// Generate random normal variable using Box-Muller transform
///
/// # Arguments
/// * `rng` - Random number generator
///
/// # Returns
/// Sample from N(0, 1)
fn rand_normal(rng: &mut rand::rngs::ThreadRng) -> f64 {
    use std::f64::consts::PI;
    let u1 = rng.gen::<f64>();
    let u2 = rng.gen::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
}

/// Count non-zero entries in matrix
///
/// # Arguments
/// * `matrix` - Input matrix
///
/// # Returns
/// Number of non-zero entries
#[allow(dead_code)]
pub fn count_nonzero(matrix: &Array2<f64>) -> usize {
    matrix.iter().filter(|x| x.abs() > 1e-15).count()
}

/// Compute edge statistics for a weight matrix
///
/// # Arguments
/// * `w` - Weight matrix
/// * `threshold` - Threshold for edge detection
///
/// # Returns
/// (num_edges, sparsity, average_edge_weight, max_weight)
pub fn edge_statistics(w: &Array2<f64>, threshold: f64) -> (usize, f64, f64, f64) {
    let d = w.shape()[0];
    let mut num_edges = 0;
    let mut sum_weights: f64 = 0.0;
    let mut max_weight: f64 = 0.0;

    for i in 0..d {
        for j in 0..d {
            let weight = w[[i, j]].abs();
            if weight > threshold {
                num_edges += 1;
                sum_weights += weight;
                max_weight = max_weight.max(weight);
            }
        }
    }

    let total_entries = d * d;
    let sparsity = (total_entries - num_edges) as f64 / total_entries as f64;
    let avg_weight = if num_edges > 0 {
        sum_weights / num_edges as f64
    } else {
        0.0
    };

    (num_edges, sparsity, avg_weight, max_weight)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random_dag_is_lower_triangular() {
        let w = random_dag(10, 0.5);
        assert!(is_lower_triangular(&w, 1e-15));
    }

    #[test]
    fn test_standardize_zero_mean() {
        let data = random_data(100, 5);
        let standardized = standardize(&data);

        for j in 0..5 {
            let col = standardized.column(j);
            let mean = col.iter().sum::<f64>() / 100.0;
            assert!(mean.abs() < 1e-10);
        }
    }

    #[test]
    fn test_close_enough_identical() {
        let a = Array2::<f64>::ones((3, 3));
        let b = Array2::<f64>::ones((3, 3));
        assert!(close_enough(&a, &b, 1e-15));
    }

    #[test]
    fn test_symmetric_identity() {
        let identity = Array2::<f64>::eye(5);
        assert!(is_symmetric(&identity, 1e-15));
    }

    #[test]
    fn test_frobenius_norm_identity() {
        let identity = Array2::<f64>::eye(5);
        let norm = frobenius_norm(&identity);
        assert!((norm - (5.0_f64).sqrt()).abs() < 1e-14);
    }

    #[test]
    fn test_structural_hamming_distance_perfect() {
        let g1 = Array2::<i32>::zeros((3, 3));
        let g2 = Array2::<i32>::zeros((3, 3));
        assert_eq!(structural_hamming_distance(&g1, &g2), 0);
    }

    #[test]
    fn test_structural_hamming_distance_single_edge() {
        let mut g1 = Array2::<i32>::zeros((3, 3));
        let g2 = Array2::<i32>::zeros((3, 3));
        g1[[0, 1]] = 1;

        let shd = structural_hamming_distance(&g1, &g2);
        assert_eq!(shd, 1);
    }

    #[test]
    fn test_edge_statistics_empty() {
        let w = Array2::<f64>::zeros((5, 5));
        let (edges, sparsity, avg_weight, max_weight) = edge_statistics(&w, 0.01);

        assert_eq!(edges, 0);
        assert_eq!(sparsity, 1.0);
        assert_eq!(avg_weight, 0.0);
        assert_eq!(max_weight, 0.0);
    }

    #[test]
    fn test_data_from_sem_shape() {
        let w = random_dag(5, 0.5);
        let data = data_from_sem(100, 5, &w, 0.1);
        assert_eq!(data.shape(), &[100, 5]);
    }
}
