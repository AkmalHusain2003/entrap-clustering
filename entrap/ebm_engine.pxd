# cython: language_level=3
# ebm_engine.pxd - Header untuk EBM Reassignment Engine (Hybrid)

cimport numpy as np

# Helper function for sorting
cdef double _get_energy_value(tuple item)

cdef class EBM_Reassignment_Engine:
    cdef public double alpha
    cdef public double beta
    cdef public double eps0
    cdef public double lambda_T
    cdef public double lambda_G0
    cdef public double tau
    cdef public int max_landmarks
    cdef public int min_landmarks
    cdef public double landmark_variance_threshold
    cdef public double ridge_epsilon
    cdef public object metric
    cdef public dict metric_params
    cdef public bint use_memmap
    cdef public bint use_incremental_tda
    cdef public object energy_computer  # Pure Python object
    cdef public object incremental_tda  # Incremental TDA engine (optional)
    cdef public object empirical_noise_energy_
    cdef public dict noise_energy_details_
    
    cpdef double geometric_energy(
        self,
        np.ndarray x,
        np.ndarray mu,
        np.ndarray Sigma_inv,
        double log_det_Sigma
    )
    
    cpdef double compute_normalized_topological_energy(
        self,
        np.ndarray points,
        int cluster_size
    )
    
    cpdef double delta_topological_energy(
        self,
        np.ndarray x,
        np.ndarray cluster_points,
        double T_prev_norm,
        int cluster_size
    )
    
    cpdef double bounded_delta_energy(
        self,
        double delta_T,
        int cluster_size
    )
    
    cpdef double estimate_noise_energy(
        self,
        np.ndarray X,
        np.ndarray true_noise_indices
    )
    
    
    # Note: reassign is def (not cpdef) to allow list comprehensions and lambdas
    
    cpdef void cleanup(self)