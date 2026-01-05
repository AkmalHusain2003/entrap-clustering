# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# ebm_engine.pyx - Implementasi EBM Reassignment Engine (Hybrid)

import numpy as np
cimport numpy as np
from libc.math cimport exp, log
from scipy.spatial import cKDTree
from scipy.linalg import inv
from numpy.linalg import slogdet

# Import Topological Energy Computer dari pure Python
from .topological_energy import Topological_Energy_Computer

# Import Incremental TDA Engine for memory-bounded caching
from .incremental_tda import Incremental_TDA_Engine

# Import dari Cython modules
from .kernels cimport (
    compute_cluster_mean,
    compute_cluster_covariance,
    compute_mahalanobis_sq,
    bounded_delta_energy_cython
)
from .utilities cimport validate_metric, select_landmarks_lhs
import gc
import logging

np.import_array()

logger = logging.getLogger(__name__)

# Helper: Extract energy from tuple for sorting (replaces lambda)
cdef inline double _get_energy_value(tuple item):
    """Extract energy value from (index, energy) tuple."""
    return item[1]


# Constants
DEF ALPHA = 0.5
DEF BETA = 0.5
DEF EPS0 = 1.0
DEF LAMBDA_T = 1.0
DEF LAMBDA_G0 = 1.0
DEF TAU = 5.0
DEF MAX_LANDMARKS = 5000
DEF MIN_LANDMARKS = 10
DEF LANDMARK_VARIANCE_THRESHOLD = 0.95
DEF RIDGE_EPSILON = 1e-6
DEF K_MIN = 10.0


cdef class EBM_Reassignment_Engine:
    """
    Energy-based noise point reassignment engine.
    
    Combines Mahalanobis (geometric) and topological energies to decide
    whether noise points should be assigned to clusters. Uses exact TDA
    on full cluster points (no landmarks) for accurate energy computation,
    with early stopping and sorted evaluation to minimize expensive calls.
    
    Attributes
    ----------
    alpha, beta : float
        Topological energy normalization and trust-region decay exponents.
    eps0 : float
        Base trust-region radius.
    lambda_T, lambda_G0 : float
        Weighting parameters for topological and geometric energies.
    tau : float
        Annealing time constant.
    empirical_noise_energy_ : float or None
        Estimated noise baseline energy.
    noise_energy_details_ : dict
        Detailed noise energy computation metadata.
    """
    
    def __init__(self,
                 double alpha = ALPHA,
                 double beta = BETA,
                 double eps0 = EPS0,
                 double lambda_T = LAMBDA_T,
                 double lambda_G0 = LAMBDA_G0,
                 double tau = TAU,
                 int max_landmarks = MAX_LANDMARKS,
                 int min_landmarks = MIN_LANDMARKS,
                 double landmark_variance_threshold = LANDMARK_VARIANCE_THRESHOLD,
                 double ridge_epsilon = RIDGE_EPSILON,
                 str metric = 'euclidean',
                 bint use_memmap = True,
                 bint use_incremental_tda = False,
                 **metric_params):
        """
        Initialize EBM_Reassignment_Engine.
        
        Parameters
        ----------
        alpha : float, default=0.5
            Topological energy normalization exponent: T_norm = T_raw / n^α.
        beta : float, default=0.5
            Trust-region decay exponent: ε_t = ε0 * n^(-β).
        eps0 : float, default=1.0
            Base trust-region radius.
        lambda_T : float, default=1.0
            Weight of bounded topological energy in total energy.
        lambda_G0 : float, default=1.0
            Maximum weight of geometric energy (decreases with iterations).
        tau : float, default=5.0
            Annealing time constant for geometric weight decay.
        max_landmarks : int, default=5000
            Max landmarks for noise energy estimation via TDA.
        min_landmarks : int, default=10
            Min landmarks for noise energy estimation.
        landmark_variance_threshold : float, default=0.95
            Variance capture target for landmark selection.
        ridge_epsilon : float, default=1e-6
            Covariance regularization strength.
        metric : str, default='euclidean'
            Distance metric.
        use_memmap : bool, default=True
            Use memory mapping for large distance matrices.
        use_incremental_tda : bool, default=False
            Enable incremental TDA with memory-bounded caching for faster
            iterative topological energy computation.
        **metric_params
            Additional metric parameters.
        """
        self.alpha = alpha
        self.beta = beta
        self.eps0 = eps0
        self.lambda_T = lambda_T
        self.lambda_G0 = lambda_G0
        self.tau = tau
        self.max_landmarks = max_landmarks
        self.min_landmarks = min_landmarks
        self.landmark_variance_threshold = landmark_variance_threshold
        self.ridge_epsilon = ridge_epsilon
        self.metric = validate_metric(metric)
        self.metric_params = metric_params
        self.use_memmap = use_memmap
        
        # Use pure Python Topological Energy Computer
        self.energy_computer = Topological_Energy_Computer(
            metric=metric, use_memmap=use_memmap, **metric_params
        )
        
        # Initialize incremental TDA engine if enabled
        self.use_incremental_tda = use_incremental_tda
        if use_incremental_tda:
            self.incremental_tda = Incremental_TDA_Engine(
                self.energy_computer,
                alpha=alpha,
                cache_config={
                    'max_diagrams': 10,
                    'memory_threshold_mb': 500,
                    'validity_window': 5
                }
            )
        else:
            self.incremental_tda = None
        
        self.empirical_noise_energy_ = None
        self.noise_energy_details_ = None
    
    cpdef double geometric_energy(
        self,
        np.ndarray x,
        np.ndarray mu,
        np.ndarray Sigma_inv,
        double log_det_Sigma
    ):
        """
        Compute Mahalanobis-based geometric energy.
        
        E_G = 0.5 * (x - μ)^T Σ^{-1} (x - μ) + 0.5 * log|Σ|.
        
        Parameters
        ----------
        x : ndarray, shape (n_features,)
            Point coordinate.
        mu : ndarray, shape (n_features,)
            Cluster mean.
        Sigma_inv : ndarray, shape (n_features, n_features)
            Inverse covariance matrix.
        log_det_Sigma : float
            Log determinant of covariance.
        
        Returns
        -------
        float
            Geometric energy.
        """
        cdef np.ndarray diff = x - mu
        cdef double mahalanobis_sq = compute_mahalanobis_sq(diff, Sigma_inv)
        cdef double E_G = 0.5 * mahalanobis_sq + 0.5 * log_det_Sigma
        return E_G
    
    cpdef double compute_normalized_topological_energy(
        self,
        np.ndarray points,
        int cluster_size
    ):
        """
        Compute scale-normalized topological energy.
        
        T_norm = T_raw / n^α, where n is cluster size. Accounts for
        topological complexity growth with cluster scale.
        
        Parameters
        ----------
        points : ndarray, shape (n_samples, n_features)
            Point set.
        cluster_size : int
            Cluster size used for normalization.
        
        Returns
        -------
        float
            Normalized topological energy.
        """
        cdef dict result = self.energy_computer.compute_raw_topological_energy(points)
        cdef double T_raw = result['T_raw']
        cdef double T_norm = T_raw / (cluster_size ** self.alpha)
        return T_norm
    
    cpdef double delta_topological_energy(
        self,
        np.ndarray x,
        np.ndarray cluster_points,
        double T_prev_norm,
        int cluster_size
    ):
        """
        Compute change in topological energy from adding point x.
        
        ΔT = T_norm(cluster ∪ {x}) - T_norm(cluster).
        
        Parameters
        ----------
        x : ndarray, shape (n_features,)
            Candidate point.
        cluster_points : ndarray, shape (n_cluster, n_features)
            Current cluster points.
        T_prev_norm : float
            Current normalized topological energy.
        cluster_size : int
            Current cluster size.
        
        Returns
        -------
        float
            Change in normalized topological energy.
        """
        cdef np.ndarray augmented_cluster = np.vstack([cluster_points, x.reshape(1, -1)])
        cdef double T_new_norm = self.compute_normalized_topological_energy(
            augmented_cluster, cluster_size + 1
        )
        return T_new_norm - T_prev_norm
    
    cpdef double bounded_delta_energy(self, double delta_T, int cluster_size):
        """
        Apply trust-region bounding to topological energy change.
        
        Prevents optimization instability via: δ̂T = ε tanh(δT / ε),
        where ε = ε0 * n^(-β).
        
        Parameters
        ----------
        delta_T : float
            Raw topological energy change.
        cluster_size : int
            Current cluster size.
        
        Returns
        -------
        float
            Bounded topological energy change, magnitude ≤ ε.
        """
        cdef double eps = self.eps0 * (cluster_size ** (-self.beta))
        return bounded_delta_energy_cython(delta_T, eps)
    
    cpdef double estimate_noise_energy(
        self,
        np.ndarray X,
        np.ndarray true_noise_indices
    ):
        """
        Estimate noise baseline energy for rejection threshold.
        
        Uses true noise points (not assigned to any cluster) to estimate
        expected topological energy of pure noise. Falls back to volume-based
        estimate if TDA fails or dataset too small.
        
        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Full data.
        true_noise_indices : ndarray
            Indices of points confirmed as noise (not in candidate sets).
        
        Returns
        -------
        float
            Estimated noise energy threshold.
        
        Notes
        -----
        Stores detailed computation metadata in noise_energy_details_.
        """
        cdef Py_ssize_t N_noise = len(true_noise_indices)
        
        if N_noise == 0:
            self.noise_energy_details_ = {'method': 'no_noise', 'E_noise': 1.0}
            return 1.0
        
        cdef np.ndarray noise_points
        cdef np.ndarray noise_range
        cdef double E_noise
        
        if N_noise < 5:
            noise_points = X[true_noise_indices]
            noise_range = np.maximum(np.ptp(noise_points, axis=0), 1e-6)
            E_noise = log(np.prod(noise_range) + 1e-12)
            self.noise_energy_details_ = {
                'method': 'volume_fallback',
                'E_noise': E_noise,
                'n_points': N_noise
            }
            return E_noise
        
        noise_points = X[true_noise_indices]
        
        cdef np.ndarray landmarks
        cdef int n_actual
        cdef double variance
        cdef dict energy_result
        cdef double T_raw, T_norm
        
        try:
            if N_noise > self.max_landmarks:
                landmarks, n_actual, variance = select_landmarks_lhs(
                    noise_points,
                    self.max_landmarks,
                    self.min_landmarks,
                    self.landmark_variance_threshold,
                    self.metric if isinstance(self.metric, str) else 'euclidean',
                    self.metric_params
                )
                energy_result = self.energy_computer.compute_raw_topological_energy(landmarks)
            else:
                energy_result = self.energy_computer.compute_raw_topological_energy(noise_points)
                n_actual = N_noise
                variance = 1.0
            
            T_raw = energy_result['T_raw']
            T_norm = T_raw / (N_noise ** self.alpha)
            
            self.noise_energy_details_ = {
                'method': 'topological_normalized',
                'E_noise': T_norm,
                'T_raw': T_raw,
                'E0': energy_result['E0'],
                'E1': energy_result['E1'],
                'H0_entropy': energy_result['H0_entropy'],
                'H1_entropy': energy_result['H1_entropy'],
                'n_components_h0': energy_result['n_components_h0'],
                'n_loops_h1': energy_result['n_loops_h1'],
                'n_points_original': N_noise,
                'n_points_final': n_actual,
                'variance_captured': variance,
                'alpha': self.alpha
            }
            
            return T_norm
            
        except Exception as e:
            logger.warning(f"Noise topology computation failed: {e}")
            noise_range = np.maximum(np.ptp(noise_points, axis=0), 1e-6)
            E_noise = log(np.prod(noise_range) + 1e-12)
            self.noise_energy_details_ = {
                'method': 'volume_fallback',
                'E_noise': E_noise,
                'n_points': N_noise,
                'error': str(e)
            }
            return E_noise

    def reassign(self, X, labels, dek_selector):
        """
        Execute energy-based noise reassignment.
        
        Main algorithm: builds candidate sets via k-NN, estimates noise energy,
        then iteratively evaluates each candidate against geometric + topological
        energy. Uses sorted evaluation and early stopping to minimize expensive
        TDA calls.
        
        Note: Uses def (not cpdef) because list comprehensions and lambdas create
        closures which Cython doesn't support in cpdef. However, extensive type
        declarations on internal variables still provide significant performance
        benefits compared to pure Python.
        
        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Data points.
        labels : ndarray, shape (n_samples,)
            Initial HDBSCAN labels (-1 = noise).
        dek_selector : Density_Equalization_K, optional
            Adaptive k selector. If None, uses K_MIN.
        
        Returns
        -------
        tuple : (refined_labels, n_rescued, cluster_stats)
            refined_labels : ndarray
                Updated labels with noise points reassigned.
            n_rescued : int
                Number of points successfully reassigned.
            cluster_stats : dict
                Per-cluster statistics: {cluster_id: {iterations, rescued, tda_calls, ...}}
        """

        cdef int iteration = 0
        cdef int total_rescued = 0
        cdef int cid
        cdef int candidate_idx
        cdef int k_adaptive, k_query, d
        cdef bint recruited_this_round = False
        cdef bint noise_mask_any
        cdef int consecutive_rejects
        cdef int early_stop_threshold = 2
        cdef double E_G, E_cheap, E_total, E_noise
        cdef double delta_T, delta_T_hat
        cdef double lambda_G_t
        cdef int candidate_local_indices_len
        cdef int cluster_size
        cdef int N_noise
        cdef int len_candidates_list
        cdef np.ndarray refined_labels
        cdef np.ndarray noise_mask
        cdef np.ndarray unique_labels
        cdef list cluster_sizes, sorted_cluster_ids
        cdef dict cluster_candidate_sets, cluster_states
        cdef set all_claimed_candidates, candidates
        cdef np.ndarray noise_indices, noise_points
        cdef object noise_tree
        cdef np.ndarray cluster_points
        cdef np.ndarray true_noise_indices_arr
        cdef np.ndarray distances, indices
        cdef list candidate_energies, candidates_list
        
        refined_labels = labels.copy()
        noise_mask = (refined_labels == -1)
        
        if not noise_mask.any():
            return refined_labels, 0, {}
        
        unique_labels = np.unique(refined_labels[refined_labels >= 0])
        if len(unique_labels) == 0:
            return refined_labels, 0, {}
        
        # Sort clusters by size descending
        cluster_sizes = [(int(cid), int(np.sum(refined_labels == cid))) 
                        for cid in unique_labels]
        cluster_sizes.sort(key=lambda x: x[1], reverse=True)
        sorted_cluster_ids = [cid for cid, _ in cluster_sizes]
        
        # Build candidate sets via k-NN
        cluster_candidate_sets = {}
        all_claimed_candidates = set()
        
        noise_indices = np.where(noise_mask)[0]
        noise_points = X[noise_indices]
        noise_tree = cKDTree(noise_points, compact_nodes=True, balanced_tree=True)
        
        for cid in sorted_cluster_ids:
            cluster_mask = (refined_labels == cid)
            cluster_points = X[cluster_mask]
            
            if dek_selector is not None:
                k_adaptive = dek_selector.get_k_percentile(cid)
            else:
                k_adaptive = int(K_MIN)
            k_query = min(k_adaptive, len(cluster_points))
            
            distances, indices = noise_tree.query(cluster_points, k=k_query, workers=-1)
            
            if cluster_points.shape[0] == 1:
                distances = distances.reshape(1, -1)
                indices = indices.reshape(1, -1)
            
            candidate_local_indices = np.unique(indices.ravel())
            candidate_local_indices = candidate_local_indices[
                candidate_local_indices < len(noise_points)
            ]
            
            if len(candidate_local_indices) > 0:
                candidate_global_indices = noise_indices[candidate_local_indices]
                cluster_candidate_sets[cid] = set(candidate_global_indices)
                all_claimed_candidates.update(candidate_global_indices)
            else:
                cluster_candidate_sets[cid] = set()
        
        # Estimate noise energy baseline
        true_noise_indices = list(set(noise_indices) - all_claimed_candidates)
        true_noise_indices_arr = np.array(true_noise_indices, dtype=np.int64)
        E_noise = self.estimate_noise_energy(X, true_noise_indices_arr)
        self.empirical_noise_energy_ = E_noise
        
        # Initialize cluster states
        cluster_states = {}
        for cid in sorted_cluster_ids:
            cluster_mask = (refined_labels == cid)
            cluster_points = X[cluster_mask]
            cluster_size = len(cluster_points)
            
            T_norm = self.compute_normalized_topological_energy(cluster_points, cluster_size)
            
            cluster_states[cid] = {
                'T_norm': T_norm,
                'size': cluster_size,
                'candidates': cluster_candidate_sets[cid].copy(),
                'cluster_indices': np.where(cluster_mask)[0].tolist(),  # Track indices for caching
                'rescued_count': 0,
                'tda_calls': 0,
                'cache_hits': 0,
                'needs_T_update': False
            }
        
        # Fixed-point iteration
        iteration = 0
        total_rescued = 0
        
        while True:
            recruited_this_round = False
            
            for cid in sorted_cluster_ids:
                state = cluster_states[cid]
                candidates = state['candidates']
                
                if len(candidates) == 0:
                    continue
                
                candidates_list = [idx for idx in candidates if refined_labels[idx] == -1]
                len_candidates_list = len(candidates_list)
                if len_candidates_list == 0:
                    continue
                
                # Get current cluster points
                cluster_mask = (refined_labels == cid)
                cluster_points = X[cluster_mask]
                lambda_G_t = self.lambda_G0 * (1.0 - np.exp(-iteration / self.tau))
                
                 # Precompute cluster statistics
                d = cluster_points.shape[1]
                mu = np.asarray(compute_cluster_mean(cluster_points))
                Sigma_reg = np.asarray(compute_cluster_covariance(
                    cluster_points, mu, self.ridge_epsilon
                ))
                
                try:
                    Sigma_inv = inv(Sigma_reg)
                    log_det_Sigma = slogdet(Sigma_reg)[1]
                except:
                    Sigma_inv = np.eye(d) / (np.trace(Sigma_reg) / d + 1e-6)
                    log_det_Sigma = 0.0
                
                # Compute geometric energy for all candidates
                candidate_energies = []
                for candidate_idx in candidates_list:
                    x = X[candidate_idx]
                    E_G = self.geometric_energy(x, mu, Sigma_inv, log_det_Sigma)
                    candidate_energies.append((candidate_idx, E_G))
                
                # Sort by ascending E_G using sorted() - cpdef compatible
                # (no lambda allowed, but sorted() with key param works fine in cpdef)
                candidate_energies = sorted(candidate_energies, key=lambda t: t[1])
                
                # Early stopping counter
                consecutive_rejects = 0
                
                for candidate_idx, E_G in candidate_energies:
                    E_cheap = lambda_G_t * E_G
                    
                    if E_G >= E_noise:
                        consecutive_rejects += 1
                        if consecutive_rejects >= early_stop_threshold:
                            break
                        continue
                    else:
                        consecutive_rejects = 0
                    
                    if E_cheap >= E_noise:
                        continue
                    
                    x = X[candidate_idx]
                    cluster_mask_current = (refined_labels == cid)
                    cluster_points_current = X[cluster_mask_current]
                    
                    # Use incremental TDA if enabled, otherwise fall back to full computation
                    if self.incremental_tda is not None:
                        cluster_indices_current = np.where(cluster_mask_current)[0]
                        delta_T, tda_meta = self.incremental_tda.compute_delta_topological_energy(
                            x, cluster_points_current, cluster_indices_current,
                            state['T_norm'], state['size']
                        )
                        state['tda_calls'] += 1
                        if tda_meta.get('cache_hit', False):
                            state['cache_hits'] += 1
                    else:
                        delta_T = self.delta_topological_energy(
                            x, cluster_points_current, state['T_norm'], state['size']
                        )
                        state['tda_calls'] += 1
                    
                    delta_T_hat = self.bounded_delta_energy(delta_T, state['size'])
                    E_total = E_cheap + self.lambda_T * delta_T_hat
                    
                    if E_total < E_noise:
                        # RESCUE
                        refined_labels[candidate_idx] = cid
                        candidates.discard(candidate_idx)
                        
                        state['needs_T_update'] = True
                        state['size'] += 1
                        state['rescued_count'] += 1
                        
                        total_rescued += 1
                        recruited_this_round = True
                        
                        for other_cid in sorted_cluster_ids:
                            if other_cid != cid:
                                cluster_states[other_cid]['candidates'].discard(candidate_idx)
                        
                        break
            
            # Batch TDA recomputation
            for cid in sorted_cluster_ids:
                state = cluster_states[cid]
                if state.get('needs_T_update', False):
                    cluster_mask_updated = (refined_labels == cid)
                    cluster_points_updated = X[cluster_mask_updated]
                    
                    state['T_norm'] = self.compute_normalized_topological_energy(
                        cluster_points_updated, state['size']
                    )
                    state['needs_T_update'] = False
                
                # Invalidate cache for completed cluster iteration
                if self.incremental_tda is not None and len(state['candidates']) == 0:
                    self.incremental_tda.invalidate_cluster_cache(state['cluster_indices'])
            
            # Signal iteration advance for cache staleness tracking
            if self.incremental_tda is not None:
                self.incremental_tda.next_iteration()
            
            iteration += 1
            
            if not recruited_this_round:
                break
        
        # Finalize statistics
        cluster_stats = {}
        for cid in sorted_cluster_ids:
            state = cluster_states[cid]
            cluster_stats[cid] = {
                'iterations': iteration,
                'rescued': state['rescued_count'],
                'tda_calls': state['tda_calls'],
                'cache_hits': state.get('cache_hits', 0),
                'final_size': state['size'],
                'final_T_norm': state['T_norm'],
                'converged': True
            }
        
        # Log cache statistics if incremental TDA enabled
        if self.incremental_tda is not None:
            cache_stats = self.incremental_tda.get_cache_stats()
            logger.info(f"Cache stats: {cache_stats}")
            # Cleanup cache
            self.incremental_tda.cleanup()
        
        gc.collect()
        return refined_labels, total_rescued, cluster_stats
    
    cpdef void cleanup(self):
        """Release memory-mapped resources and incremental TDA cache."""
        self.energy_computer.cleanup()
        if self.incremental_tda is not None:
            self.incremental_tda.cleanup()