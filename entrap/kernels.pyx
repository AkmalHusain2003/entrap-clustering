# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# kernels.pyx - Implementasi kernel numerik yang dioptimalkan

import numpy as np
cimport numpy as np
from libc.math cimport sqrt, exp, tanh, log

np.import_array()


cpdef np.ndarray[np.float64_t, ndim=1] compute_cov_from_rows(
    np.ndarray[np.float64_t, ndim=2] neighbor_distances
):
    """
    Compute coefficient of variation for each row of distances.
    
    For each point, computes the coefficient of variation (std/mean) of its
    neighbor distances. Used to quantify local density heterogeneity.
    
    Parameters
    ----------
    neighbor_distances : ndarray, shape (n_points, n_neighbors)
        Distance matrix where each row contains distances to neighbors.
    
    Returns
    -------
    ndarray, shape (n_points,)
        Coefficient of variation for each point. Returns inf if mean is ~0.
    
    Notes
    -----
    Numerically stable computation with early zero-detection.
    """
    cdef Py_ssize_t n_points = neighbor_distances.shape[0]
    cdef Py_ssize_t m = neighbor_distances.shape[1]
    cdef np.ndarray[np.float64_t, ndim=1] covs = np.zeros(n_points, dtype=np.float64)
    
    cdef Py_ssize_t i, j
    cdef double row_sum, mean, variance_sum, diff, std
    
    for i in range(n_points):
        row_sum = 0.0
        for j in range(m):
            row_sum += neighbor_distances[i, j]
        mean = row_sum / m
        
        if mean > 1e-12:
            variance_sum = 0.0
            for j in range(m):
                diff = neighbor_distances[i, j] - mean
                variance_sum += diff * diff
            std = sqrt(variance_sum / m)
            covs[i] = std / mean
        else:
            covs[i] = np.inf
    
    return covs


cpdef np.ndarray[np.float64_t, ndim=1] compute_cluster_mean(
    np.ndarray[np.float64_t, ndim=2] cluster_points
):
    """
    Compute mean (centroid) of cluster points.
    
    Parameters
    ----------
    cluster_points : ndarray, shape (n_samples, n_features)
        Point coordinates.
    
    Returns
    -------
    ndarray, shape (n_features,)
        Cluster centroid.
    """
    cdef Py_ssize_t n = cluster_points.shape[0]
    cdef Py_ssize_t d = cluster_points.shape[1]
    cdef np.ndarray[np.float64_t, ndim=1] mean = np.zeros(d, dtype=np.float64)
    
    cdef Py_ssize_t i, j
    
    for i in range(n):
        for j in range(d):
            mean[j] += cluster_points[i, j]
    for j in range(d):
        mean[j] /= n
    
    return mean


cpdef np.ndarray[np.float64_t, ndim=2] compute_cluster_covariance(
    np.ndarray[np.float64_t, ndim=2] cluster_points,
    np.ndarray[np.float64_t, ndim=1] mean,
    double ridge_epsilon
):
    """
    Compute regularized covariance matrix of cluster.
    
    Applies ridge regularization (adds small constant to diagonal) to ensure
    invertibility and numerical stability.
    
    Parameters
    ----------
    cluster_points : ndarray, shape (n_samples, n_features)
        Point coordinates.
    mean : ndarray, shape (n_features,)
        Cluster mean.
    ridge_epsilon : float
        Regularization strength added to diagonal elements.
    
    Returns
    -------
    ndarray, shape (n_features, n_features)
        Regularized covariance matrix.
    """
    cdef Py_ssize_t n = cluster_points.shape[0]
    cdef Py_ssize_t d = cluster_points.shape[1]
    cdef np.ndarray[np.float64_t, ndim=2] cov = np.zeros((d, d), dtype=np.float64)
    
    cdef Py_ssize_t i, j, k
    cdef double[:] diff = np.zeros(d, dtype=np.float64)
    
    for i in range(n):
        for j in range(d):
            diff[j] = cluster_points[i, j] - mean[j]
        
        for j in range(d):
            for k in range(d):
                cov[j, k] += diff[j] * diff[k]
    
    for j in range(d):
        for k in range(d):
            cov[j, k] /= n
            if j == k:
                cov[j, k] += ridge_epsilon
    
    return cov


cpdef double compute_mahalanobis_sq(
    np.ndarray[np.float64_t, ndim=1] diff,
    np.ndarray[np.float64_t, ndim=2] Sigma_inv
):
    """
    Compute squared Mahalanobis distance.
    
    Computes (x - μ)^T Σ^{-1} (x - μ) efficiently without forming
    full intermediate matrices.
    
    Parameters
    ----------
    diff : ndarray, shape (n_features,)
        Deviation vector (x - μ).
    Sigma_inv : ndarray, shape (n_features, n_features)
        Inverse covariance matrix.
    
    Returns
    -------
    float
        Squared Mahalanobis distance.
    """
    cdef Py_ssize_t n = diff.shape[0]
    cdef double result = 0.0
    cdef double temp
    cdef Py_ssize_t i, j
    
    for i in range(n):
        temp = 0.0
        for j in range(n):
            temp += diff[j] * Sigma_inv[j, i]
        result += diff[i] * temp
    
    return result


cpdef double logistic_mapping(
    double cov_value,
    double cov_10,
    double cov_50,
    double cov_90,
    double q_min,
    double q_max,
    double alpha
):
    """
    Map coefficient of variation to adaptive quantile via sigmoid.
    
    Implements smooth, monotonic mapping from local density heterogeneity
    (measured by CoV) to neighborhood quantile parameter. Centered at CoV_50.
    
    Parameters
    ----------
    cov_value : float
        Coefficient of variation for current point.
    cov_10, cov_50, cov_90 : float
        Reference percentiles (10th, 50th, 90th) of CoV distribution.
    q_min, q_max : float
        Output quantile range [q_min, q_max].
    alpha : float, default=10.0
        Sigmoid steepness (higher = sharper transition).
    
    Returns
    -------
    float
        Adaptive quantile in [q_min, q_max].
    
    Notes
    -----
    Uses logistic function: q(CoV) = q_min + (q_max - q_min) * sigmoid(...).
    """
    cdef double delta = 1e-12
    cdef double a = alpha / (cov_90 - cov_10 + delta)
    cdef double b = cov_50
    cdef double exponent = -a * (cov_value - b)
    cdef double sigmoid = 1.0 / (1.0 + exp(exponent))
    cdef double q_adaptive = q_min + (q_max - q_min) * sigmoid
    
    # Clip to range
    if q_adaptive < q_min:
        q_adaptive = q_min
    elif q_adaptive > q_max:
        q_adaptive = q_max
    
    return q_adaptive


cpdef double bounded_delta_energy_cython(double delta_T, double eps):
    """
    Apply trust-region bounding via hyperbolic tangent.
    
    Caps magnitude of energy change to prevent optimization instability.
    Bounded change: δ̂T = ε tanh(δT / ε).
    
    Parameters
    ----------
    delta_T : float
        Raw energy change.
    eps : float
        Trust-region radius.
    
    Returns
    -------
    float
        Bounded energy change, magnitude ≤ eps.
    """
    return eps * tanh(delta_T / eps)