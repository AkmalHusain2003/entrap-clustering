"""
dek_selector.py - Pure Python implementation of Density Equalization K selector
"""

import numpy as np
import gc
from scipy.spatial import cKDTree

from .utilities import (
    estimate_intrinsic_dimension_twenn,
    compute_cov_distribution
)
from .kernels import compute_cov_from_rows, logistic_mapping

# Constants
DEK_Q_MIN = 0.1
DEK_Q_MAX = 0.9
K_MIN = 10.0
K_MAX = 50.0
M_MIN = 2.0
K_PERCENTILE = 50.0


class Density_Equalization_K:
    """
    Adaptive k-NN parameter selection via density equalization.
    
    Fits per-point and per-cluster adaptive k values based on local density
    heterogeneity (coefficient of variation of neighbor distances).
    Denser regions use larger k; sparse regions use smaller k.
    
    Attributes
    ----------
    q_min, q_max : float
        Quantile range for neighbor radius selection.
    k_min, k_max : float
        Bounds on k-NN parameter.
    m_min : float
        Minimum neighborhood size estimate.
    alpha : float
        Sigmoid steepness in CoV → quantile mapping.
    cluster_k_values_ : dict
        Per-point k values: {cluster_id: ndarray of k values}.
    cluster_intrinsic_dims_ : dict
        Intrinsic dimension per cluster.
    fitted_ : bool
        Whether fit() has been called.
    """
    
    def __init__(self, alpha=10.0):
        """
        Initialize Density_Equalization_K.
        
        Parameters
        ----------
        alpha : float, default=10.0
            Sigmoid steepness in logistic CoV mapping.
        """
        self.q_min = DEK_Q_MIN
        self.q_max = DEK_Q_MAX
        self.k_min = K_MIN
        self.k_max = K_MAX
        self.m_min = M_MIN
        self.alpha = float(alpha)
        self.cluster_k_values_ = {}
        self.cluster_intrinsic_dims_ = {}
        self.fitted_ = False
        self.last_estimated_dim_ = 1.0
    
    def _compute_adaptive_m(self, X, n):
        """
        Compute adaptive neighborhood size via Two-NN intrinsic dimension.
        
        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Cluster points.
        n : int
            Number of samples.
        
        Returns
        -------
        int
            Adaptive m in [m_min, n-1].
        """
        
        if n <= 2:
            return max(int(self.m_min), n - 1)
        
        d_hat = estimate_intrinsic_dimension_twenn(X, False)
        self.last_estimated_dim_ = d_hat
        m_adaptive = int(np.floor(n ** (1.0 / (d_hat + 1.0))))
        m_adaptive = max(int(self.m_min), m_adaptive)
        m_adaptive = min(m_adaptive, n - 1)
        return m_adaptive
    
    def fit(self, X, labels):
        """
        Fit adaptive k values for each cluster.
        
        Computes per-point k-NN parameters using local density heterogeneity
        (coefficient of variation). Uses logistic mapping of CoV → quantile
        to select adaptive neighborhood radius per point.
        
        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Data points.
        labels : ndarray, shape (n_samples,)
            Cluster labels (-1 = noise).
        
        Returns
        -------
        self
            Fitted estimator.
        """
        
        X = np.asarray(X, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.int64)
        
        unique_clusters = np.unique(labels[labels >= 0])
        self.cluster_k_values_.clear()
        self.cluster_intrinsic_dims_.clear()
        
        for cid in unique_clusters:
            mask = (labels == cid)
            cluster_points = X[mask]
            n_points = cluster_points.shape[0]
            
            if n_points <= 1:
                self.cluster_k_values_[cid] = np.full(n_points, int(self.k_min), dtype=np.int64)
                self.cluster_intrinsic_dims_[cid] = 1.0
                continue
            
            m_adaptive = self._compute_adaptive_m(cluster_points, n_points)
            d_hat = self.last_estimated_dim_
            self.cluster_intrinsic_dims_[cid] = d_hat
            
            k_query = min(m_adaptive + 1, n_points)
            tree = cKDTree(cluster_points, leafsize=40, compact_nodes=True, balanced_tree=True)
            distances, indices = tree.query(cluster_points, k=k_query, p=2, workers=-1)
            
            if k_query > 1:
                neighbor_dist = distances[:, 1:]
            else:
                neighbor_dist = np.zeros((n_points, 0), dtype=np.float64)
            
            if neighbor_dist.shape[1] == 0:
                self.cluster_k_values_[cid] = np.full(n_points, int(self.k_min), dtype=np.int64)
                continue
            
            per_point_covs = compute_cov_from_rows(neighbor_dist)
            cov_10, cov_50, cov_90 = compute_cov_distribution(per_point_covs)
            
            q_adaptive = np.zeros(n_points, dtype=np.float64)
            for i in range(n_points):
                q_adaptive[i] = logistic_mapping(
                    per_point_covs[i], cov_10, cov_50, cov_90,
                    self.q_min, self.q_max, self.alpha
                )
            
            r_adaptive = np.zeros(n_points, dtype=np.float64)
            for i in range(n_points):
                row_distances = neighbor_dist[i, :]
                if len(row_distances) > 0:
                    r_adaptive[i] = np.quantile(row_distances, q_adaptive[i])
                else:
                    r_adaptive[i] = 0.0
            
            r_adaptive = np.maximum(r_adaptive, 1e-12)
            neighbor_lists = tree.query_ball_tree(tree, r=r_adaptive.max(), p=2)
            
            k_values = np.zeros(n_points, dtype=np.int64)
            for i in range(n_points):
                count = 0
                for j in neighbor_lists[i]:
                    if i != j:
                        dist_ij = np.linalg.norm(cluster_points[i] - cluster_points[j])
                        if dist_ij <= r_adaptive[i]:
                            count += 1
                k_values[i] = count
            
            k_values = np.clip(k_values, int(self.k_min), int(self.k_max)).astype(np.int64)
            if np.all(k_values == 0):
                k_values[:] = int(self.k_min)
            
            self.cluster_k_values_[cid] = k_values
        
        self.fitted_ = True
        gc.collect()
        return self
    
    def get_k_percentile(self, cluster_id, percentile=K_PERCENTILE):
        """
        Get k value at specified percentile for a cluster.
        
        Parameters
        ----------
        cluster_id : int
            Cluster ID.
        percentile : float, default=50.0
            Percentile of k values to return.
        
        Returns
        -------
        int
            k value at specified percentile, or k_min if not fitted.
        """
        if (not self.fitted_) or (cluster_id not in self.cluster_k_values_):
            return int(self.k_min)
        
        vals = self.cluster_k_values_[cluster_id]
        if vals.size > 0:
            return int(np.round(np.percentile(vals, percentile)))
        else:
            return int(self.k_min)
    
    def get_intrinsic_dimension(self, cluster_id):
        """
        Get estimated intrinsic dimension for cluster.
        
        Parameters
        ----------
        cluster_id : int
            Cluster ID.
        
        Returns
        -------
        float
            Estimated intrinsic dimension, or 1.0 if not fitted.
        """
        if (not self.fitted_) or (cluster_id not in self.cluster_intrinsic_dims_):
            return 1.0
        return self.cluster_intrinsic_dims_[cluster_id]