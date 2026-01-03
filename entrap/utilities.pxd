# cython: language_level=3
# utilities.pxd - Header file for utilities module

import numpy as np
cimport numpy as np

# Validate metric parameter
cpdef object validate_metric(object metric)

# Estimate intrinsic dimension using Two-NN method
cpdef double estimate_intrinsic_dimension_twenn(
    np.ndarray X,
    bint X_is_dist = *
)

# Compute percentiles of coefficient of variation
cpdef tuple compute_cov_distribution(np.ndarray[np.float64_t, ndim=1] per_point_covs)

# Select landmarks using Latin Hypercube Sampling
cpdef tuple select_landmarks_lhs(
    np.ndarray points,
    int max_landmarks = *,
    int min_landmarks = *,
    double variance_threshold = *,
    str metric = *,
    dict metric_params = *
)