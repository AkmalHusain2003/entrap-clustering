# cython: language_level=3
# kernels.pxd - Header file untuk fungsi-fungsi kernel numerik

cimport numpy as np

# Deklarasi fungsi untuk komputasi coefficient of variation
cpdef np.ndarray[np.float64_t, ndim=1] compute_cov_from_rows(
    np.ndarray[np.float64_t, ndim=2] neighbor_distances
)

# Deklarasi fungsi untuk komputasi mean cluster
cpdef np.ndarray[np.float64_t, ndim=1] compute_cluster_mean(
    np.ndarray[np.float64_t, ndim=2] cluster_points
)

# Deklarasi fungsi untuk komputasi covariance cluster
cpdef np.ndarray[np.float64_t, ndim=2] compute_cluster_covariance(
    np.ndarray[np.float64_t, ndim=2] cluster_points,
    np.ndarray[np.float64_t, ndim=1] mean,
    double ridge_epsilon
)

# Deklarasi fungsi untuk komputasi squared Mahalanobis distance
cpdef double compute_mahalanobis_sq(
    np.ndarray[np.float64_t, ndim=1] diff,
    np.ndarray[np.float64_t, ndim=2] Sigma_inv
)

# Deklarasi fungsi untuk logistic mapping
cpdef double logistic_mapping(
    double cov_value,
    double cov_10,
    double cov_50,
    double cov_90,
    double q_min,
    double q_max,
    double alpha
)

# Deklarasi fungsi untuk bounded delta energy
cpdef double bounded_delta_energy_cython(
    double delta_T,
    double eps
)