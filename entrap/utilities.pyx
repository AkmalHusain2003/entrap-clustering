# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# utilities.pyx - Implementasi fungsi-fungsi utilitas

import numpy as np
cimport numpy as np
from libc.math cimport log
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist, squareform, pdist
from scipy.stats import qmc

np.import_array()

# Metrik yang didukung
SUPPORTED_METRICS = {
    'euclidean', 'manhattan', 'cityblock', 'minkowski', 'chebyshev',
    'cosine', 'correlation', 'hamming', 'jaccard', 'canberra',
    'braycurtis', 'mahalanobis', 'seuclidean', 'sqeuclidean'
}


cpdef object validate_metric(object metric):
    """
    Validate distance metric parameter.
    
    Parameters
    ----------
    metric : str or callable
        Distance metric name or custom metric function.
    
    Returns
    -------
    str or callable
        Validated metric.
    
    Raises
    ------
    ValueError
        If metric is a string not in SUPPORTED_METRICS and not callable.
    """
    if callable(metric):
        return metric
    if metric not in SUPPORTED_METRICS:
        raise ValueError(f"Metric '{metric}' not supported.")
    return metric


cpdef double estimate_intrinsic_dimension_twenn(
    np.ndarray X,
    bint X_is_dist = False
):
    """
    Estimate intrinsic dimension using Two-NN method.
    
    Fits ratio of 2nd-to-1st nearest neighbor distances to determine
    intrinsic dimensionality of the data manifold. Based on:
    "Estimating the Intrinsic Dimension of Datasets by a Minimal 
    Neighborhood Information" (Facco et al., 2017).
    
    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features) or (n_samples, n_samples)
        Data points or precomputed distance matrix.
    X_is_dist : bool, default=False
        If True, X is treated as a symmetric distance matrix.
    
    Returns
    -------
    float
        Estimated intrinsic dimension, clipped to [1, min(n_features, n_samples)].
    
    Notes
    -----
    Requires at least 3 samples. Returns 1.0 for smaller datasets.
    """
    cdef Py_ssize_t N = X.shape[0]
    if N < 3:
        return 1.0
    
    cdef np.ndarray dist
    cdef np.ndarray mu
    cdef np.ndarray sort_idx
    cdef np.ndarray Femp
    cdef double r1, r2
    cdef Py_ssize_t i
    cdef double d_hat
    
    if X_is_dist:
        dist = X
    else:
        dist = squareform(pdist(X, metric='euclidean'))
    
    mu = np.zeros(N, dtype=np.float64)
    for i in range(N):
        sort_idx = np.argsort(dist[i, :])
        r1 = dist[i, sort_idx[1]]
        r2 = dist[i, sort_idx[2]]
        mu[i] = r2 / r1 if r1 > 1e-12 else 1.0
    
    sort_idx = np.argsort(mu)
    Femp = np.arange(N, dtype=np.float64) / N
    
    lr = LinearRegression(fit_intercept=False)
    log_mu = np.log(mu[sort_idx] + 1e-12).reshape(-1, 1)
    neg_log_1_minus_F = -np.log(1 - Femp + 1e-12).reshape(-1, 1)
    lr.fit(log_mu, neg_log_1_minus_F)
    d_hat = lr.coef_[0][0]
    
    # Clip to valid range
    if X_is_dist:
        return max(1.0, min(d_hat, <double>N))
    else:
        return max(1.0, min(d_hat, <double>X.shape[1]))


cpdef tuple compute_cov_distribution(np.ndarray[np.float64_t, ndim=1] per_point_covs):
    """
    Compute percentiles (10th, 50th, 90th) of coefficient of variation.
    
    Handles infinite values and ensures positive spread to avoid division
    by zero in subsequent quantile mappings.
    
    Parameters
    ----------
    per_point_covs : ndarray, shape (n_points,)
        Coefficient of variation values.
    
    Returns
    -------
    tuple of float
        (q10, q50, q90) - 10th, 50th, and 90th percentiles.
    
    Notes
    -----
    If q90 == q10 (no spread), adds epsilon to q90.
    Infinite values are excluded before percentile computation.
    """
    if len(per_point_covs) == 0:
        return (0.0, 0.5, 1.0)
    
    cdef np.ndarray finite_mask = np.isfinite(per_point_covs)
    cdef np.ndarray finite_array = per_point_covs[finite_mask]
    
    if len(finite_array) == 0:
        return (0.0, 0.5, 1.0)
    
    cdef double q10 = np.percentile(finite_array, 10)
    cdef double q50 = np.percentile(finite_array, 50)
    cdef double q90 = np.percentile(finite_array, 90)
    
    cdef double epsilon
    if abs(q90 - q10) < 1e-12:
        epsilon = max(1e-6, 0.01 * abs(q10) if q10 != 0.0 else 1e-6)
        q90 = q10 + epsilon
    
    return (q10, q50, q90)


cpdef tuple select_landmarks_lhs(
    np.ndarray points,
    int max_landmarks = 5000,
    int min_landmarks = 10,
    double variance_threshold = 0.95,
    str metric = 'euclidean',
    dict metric_params = {}
):
    """
    Select landmarks using Latin Hypercube Sampling (LHS).
    
    Reduces dataset size for expensive topological computations via LHS.
    Falls back to random sampling if LHS fails, and fills gaps if not
    enough unique points are selected.
    
    Parameters
    ----------
    points : ndarray, shape (n_samples, n_features)
        Input points.
    max_landmarks : int, default=5000
        Maximum number of landmarks to select.
    min_landmarks : int, default=10
        Minimum number of landmarks.
    variance_threshold : float, default=0.95
        Target fraction of variance to explain.
    metric : str, default='euclidean'
        Distance metric for finding nearest points.
    metric_params : dict
        Additional parameters for distance metric.
    
    Returns
    -------
    landmarks : ndarray, shape (n_landmarks, n_features)
        Selected landmark points.
    n_landmarks : int
        Number of landmarks actually selected.
    variance_captured : float
        Fraction of variance in original data explained by landmarks.
    
    Notes
    -----
    If n_samples <= max_landmarks, returns all points.
    Variance is estimated via PCA on full and landmark sets.
    """
    cdef Py_ssize_t n_points = points.shape[0]
    cdef Py_ssize_t n_features = points.shape[1]
    
    if n_points <= max_landmarks:
        return points, n_points, 1.0
    
    cdef int n_target = min(max_landmarks, n_points)
    cdef int n_actual = max(int(0.1 * n_points), min_landmarks)
    n_actual = min(n_actual, n_target)
    
    cdef np.ndarray landmarks
    cdef np.ndarray unique_indices
    cdef np.ndarray indices
    cdef double variance_captured
    
    try:
        sampler = qmc.LatinHypercube(d=n_features)
        sample = sampler.random(n=n_actual)
        
        l_bounds = np.min(points, axis=0)
        u_bounds = np.max(points, axis=0)
        
        margin = (u_bounds - l_bounds) * 0.01
        scaled_samples = qmc.scale(sample, l_bounds - margin, u_bounds + margin)
        
        if metric == 'euclidean':
            tree = cKDTree(points)
            _, indices = tree.query(scaled_samples, k=1, workers=-1)
        else:
            dists = cdist(scaled_samples, points, metric=metric, **metric_params)
            indices = np.argmin(dists, axis=1)
        
        unique_indices = np.unique(indices)
        landmarks = points[unique_indices]
        
        if len(landmarks) < n_actual // 2:
            remaining_needed = n_actual - len(landmarks)
            remaining_indices = np.setdiff1d(np.arange(n_points), unique_indices)
            if len(remaining_indices) > 0:
                fill_indices = np.random.choice(
                    remaining_indices,
                    size=min(len(remaining_indices), remaining_needed),
                    replace=False
                )
                landmarks = np.vstack([landmarks, points[fill_indices]])
    
    except Exception as e:
        import logging
        logging.warning(f"LHS selection failed: {e}")
        indices = np.random.choice(n_points, size=min(n_points, n_actual), replace=False)
        landmarks = points[indices]
    
    try:
        if len(landmarks) < n_features:
            variance_captured = 1.0
        else:
            pca_full = PCA(n_components=min(n_points, n_features))
            pca_full.fit(points)
            total_var = np.sum(pca_full.explained_variance_)
            
            pca_sub = PCA(n_components=min(len(landmarks), n_features))
            pca_sub.fit(landmarks)
            sub_var = np.sum(pca_sub.explained_variance_)
            
            variance_captured = min(1.0, sub_var / (total_var + 1e-12))
    except:
        variance_captured = 1.0
    
    return landmarks, len(landmarks), variance_captured