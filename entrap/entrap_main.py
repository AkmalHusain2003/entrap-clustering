"""
entrap_main.py - Main ENTRAP class and results container
"""

import numpy as np
import warnings
import time
import logging
from typing import Dict, Optional
from dataclasses import dataclass

from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_array

try:
    from hdbscan import HDBSCAN
except ImportError as e:
    raise ImportError(f"Missing dependency: {e}")

# Import dari modul dengan absolute path
from .ebm_engine import EBM_Reassignment_Engine
from .dek_selector import Density_Equalization_K
from .utilities import validate_metric

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


# ============================================================================
# HYPERPARAMETERS
# ============================================================================

ALPHA = 0.5
BETA = 0.5
EPS0 = 1.0
LAMBDA_T = 1.0
LAMBDA_G0 = 1.0
TAU = 5.0

MAX_LANDMARKS = 5000
MIN_LANDMARKS = 10
LANDMARK_VARIANCE_THRESHOLD = 0.95
RIDGE_EPSILON = 1e-6


# ============================================================================
# RESULTS CONTAINER
# ============================================================================

@dataclass
class ENTRAP_Results:
    """
    Container for ENTRAP clustering results.
    
    Attributes
    ----------
    labels : ndarray, shape (n_samples,)
        Final cluster labels (-1 = noise).
    probabilities : ndarray, shape (n_samples,)
        HDBSCAN membership probabilities.
    noise_rescued : int
        Number of points reassigned from noise.
    execution_time : float
        Total runtime in seconds.
    n_clusters : int
        Number of non-noise clusters.
    cluster_stats : dict
        Per-cluster statistics from reassignment.
    empirical_noise_energy : float
        Estimated noise baseline energy.
    noise_energy_details : dict, optional
        Detailed noise energy computation metadata.
    hyperparameters : dict, optional
        All hyperparameters used.
    """
    labels: np.ndarray
    probabilities: np.ndarray
    noise_rescued: int
    execution_time: float
    n_clusters: int
    cluster_stats: Dict[int, Dict]
    empirical_noise_energy: float
    noise_energy_details: Optional[Dict] = None
    hyperparameters: Optional[Dict] = None


# ============================================================================
# MAIN CLASSIFIER
# ============================================================================

class ENTRAP(BaseEstimator, ClusterMixin):
    """
    ENergy-based Topological Rescue of Ambiguous Points (ENTRAP).
    
    Two-stage clustering method: (1) initial density-based clustering via HDBSCAN,
    (2) energy-based refinement to rescue noise points that belong to clusters.
    Uses geometric (Mahalanobis) + topological (persistent homology) energies
    for principled point reassignment.
    
    The method minimizes total energy E = λ_G * E_G + λ_T * ΔT, where:
    - E_G: Mahalanobis distance scaled by cluster covariance
    - ΔT: Change in normalized topological energy (H0 + H1 lifetimes + entropies)
    - Comparison to empirical noise energy establishes acceptance threshold
    
    Parameters
    ----------
    min_cluster_size : int, default=30
        Minimum cluster size for HDBSCAN initial clustering.
    min_samples : int, optional
        Minimum samples for HDBSCAN reachability calculation.
        Defaults to min_cluster_size if not specified.
    alpha : float, default=0.5
        Topological energy scale normalization exponent.
        T_norm = T_raw / n^α. Higher α suppresses noise in large clusters.
    beta : float, default=0.5
        Trust-region decay exponent.
        ε_t = ε0 * n^(-β). Controls convergence rate.
    eps0 : float, default=1.0
        Base trust-region radius.
    lambda_T : float, default=1.0
        Weight of bounded topological energy change in total energy.
    lambda_G0 : float, default=1.0
        Initial weight of geometric energy (anneals with iterations).
    tau : float, default=5.0
        Annealing time constant for geometric weight decay.
        λ_G(t) = λ_G0 * (1 - exp(-t / τ)).
    max_landmarks : int, default=5000
        Maximum number of landmarks for noise energy estimation.
    min_landmarks : int, default=10
        Minimum number of landmarks.
    landmark_variance_threshold : float, default=0.95
        Target variance capture fraction for landmark selection.
    ridge_epsilon : float, default=1e-6
        Tikhonov regularization strength for covariance matrix inversion.
    metric : str or callable, default='euclidean'
        Distance metric. Supports scikit-learn metrics or custom callable.
    metric_params : dict, optional
        Additional parameters for distance metric.
    use_memmap : bool, default=True
        Use disk-backed memory mapping for distance matrices > 50k samples.
        Reduces peak RAM at cost of I/O overhead.
    
    Attributes
    ----------
    labels_ : ndarray, shape (n_samples,)
        Final cluster labels after reassignment.
    probabilities_ : ndarray, shape (n_samples,)
        HDBSCAN membership probabilities.
    result_ : ENTRAP_Results
        Complete results object with timing and statistics.
    
    Notes
    -----
    Uses ripser for persistent homology (requires ripser-py).
    Uses HDBSCAN for initial clustering (requires hdbscan).
    Critical numerical loops are Cython-compiled for speed.
    
    Example
    -------
    >>> from entrap import ENTRAP
    >>> import numpy as np
    >>> X = np.random.randn(300, 10)
    >>> clf = ENTRAP(min_cluster_size=20, lambda_T=1.5)
    >>> labels = clf.fit_predict(X)
    >>> stats = clf.get_summary()
    """

    def __init__(self,
                 min_cluster_size: int = 30,
                 min_samples: Optional[int] = None,
                 alpha: float = ALPHA,
                 beta: float = BETA,
                 eps0: float = EPS0,
                 lambda_T: float = LAMBDA_T,
                 lambda_G0: float = LAMBDA_G0,
                 tau: float = TAU,
                 max_landmarks: int = MAX_LANDMARKS,
                 min_landmarks: int = MIN_LANDMARKS,
                 landmark_variance_threshold: float = LANDMARK_VARIANCE_THRESHOLD,
                 ridge_epsilon: float = RIDGE_EPSILON,
                 metric: str = 'euclidean',
                 metric_params: Optional[dict] = None,
                 use_memmap: bool = True):
        """
        Initialize ENTRAP clusterer.
        
        Parameters
        ----------
        See class docstring for full parameter descriptions.
        """
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
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
        self.metric_params = metric_params or {}
        self.use_memmap = use_memmap

        self.ebm_engine = EBM_Reassignment_Engine(
            alpha=alpha,
            beta=beta,
            eps0=eps0,
            lambda_T=lambda_T,
            lambda_G0=lambda_G0,
            tau=tau,
            max_landmarks=max_landmarks,
            min_landmarks=min_landmarks,
            landmark_variance_threshold=landmark_variance_threshold,
            ridge_epsilon=ridge_epsilon,
            metric=metric,
            use_memmap=use_memmap,
            **self.metric_params
        )

        self.labels_ = None
        self.probabilities_ = None
        self.hdbscan_clusterer_ = None
        self.result_ = None

    def fit(self, X, y=None):
        """
        Fit ENTRAP clustering model.
        
        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Training data.
        y : ignored
            Present for API consistency.
        
        Returns
        -------
        self
            Fitted estimator.
        
        Raises
        ------
        ImportError
            If required dependencies (hdbscan, ripser) are unavailable.
        """
        start_time = time.time()
        X = check_array(X, accept_sparse=False, ensure_2d=True)

        # Initial clustering with HDBSCAN
        hdbscan = HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples or self.min_cluster_size,
            metric=self.metric,
            algorithm='best',
            core_dist_n_jobs=-1,
            **self.metric_params
        )
        hdbscan.fit(X)
        self.hdbscan_clusterer_ = hdbscan
        labels = hdbscan.labels_.copy()
        probabilities = hdbscan.probabilities_.copy()

        # Adaptive k-NN selection
        dek_selector = Density_Equalization_K()
        try:
            dek_selector.fit(X, labels)
        except:
            dek_selector = None

        # Energy-based reassignment
        try:
            labels_refined, n_rescued, cluster_stats = self.ebm_engine.reassign(
                X, labels, dek_selector
            )
        except Exception as e:
            logger.error(f"EBM reassignment failed: {e}")
            raise

        self.labels_ = labels_refined
        self.probabilities_ = probabilities
        
        elapsed = time.time() - start_time
        n_clusters_final = len(np.unique(labels_refined[labels_refined >= 0]))

        hyperparameters = {
            'alpha': self.alpha,
            'beta': self.beta,
            'eps0': self.eps0,
            'lambda_T': self.lambda_T,
            'lambda_G0': self.lambda_G0,
            'tau': self.tau,
            'max_landmarks': self.max_landmarks,
            'min_landmarks': self.min_landmarks,
            'landmark_variance_threshold': self.landmark_variance_threshold,
            'ridge_epsilon': self.ridge_epsilon
        }

        self.result_ = ENTRAP_Results(
            labels=labels_refined,
            probabilities=probabilities,
            noise_rescued=n_rescued,
            execution_time=elapsed,
            n_clusters=n_clusters_final,
            cluster_stats=cluster_stats,
            empirical_noise_energy=self.ebm_engine.empirical_noise_energy_,
            noise_energy_details=self.ebm_engine.noise_energy_details_,
            hyperparameters=hyperparameters
        )

        self.ebm_engine.cleanup()
        return self

    def fit_predict(self, X, y=None):
        """
        Fit clustering model and return labels.
        
        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Training data.
        y : ignored
            Present for API consistency.
        
        Returns
        -------
        ndarray, shape (n_samples,)
            Cluster labels (-1 = noise).
        """
        return self.fit(X, y).labels_

    def get_summary(self) -> dict:
        """
        Get high-level summary of clustering results.
        
        Returns
        -------
        dict
            Summary dictionary with keys:
            - method : str - 'ENTRAP'
            - version : str
            - n_clusters : int
            - n_noise : int - Final noise count
            - cluster_sizes : list[int]
            - noise_rescued : int
            - execution_time : float
            - metric : str
            - empirical_noise_energy : float
            - hyperparameters : dict
            - noise_energy_method : str
            - noise_topology : dict (optional)
            - tda_efficiency : dict
        
        Raises
        ------
        ValueError
            If fit() has not been called.
        """
        if self.labels_ is None:
            raise ValueError("Must call fit() first")

        unique_labels = np.unique(self.labels_[self.labels_ >= 0])
        cluster_sizes = [int(np.sum(self.labels_ == l)) for l in unique_labels]

        summary = {
            'method': 'ENTRAP',
            'version': '1.0',
            'n_clusters': self.result_.n_clusters,
            'n_noise': int(np.sum(self.labels_ == -1)),
            'cluster_sizes': cluster_sizes,
            'noise_rescued': self.result_.noise_rescued,
            'execution_time': self.result_.execution_time,
            'metric': self.metric if isinstance(self.metric, str) else 'custom',
            'empirical_noise_energy': self.result_.empirical_noise_energy,
            'hyperparameters': self.result_.hyperparameters
        }
        
        if self.result_.noise_energy_details:
            summary['noise_energy_method'] = self.result_.noise_energy_details['method']
            if self.result_.noise_energy_details['method'] == 'topological_normalized':
                ned = self.result_.noise_energy_details
                summary['noise_topology'] = {
                    'E_noise': ned['E_noise'],
                    'T_raw': ned['T_raw'],
                    'H0_components': ned['n_components_h0'],
                    'H1_loops': ned['n_loops_h1'],
                    'n_points_original': ned['n_points_original'],
                    'n_points_final': ned['n_points_final'],
                    'alpha': ned['alpha']
                }
        
        total_tda_calls = sum(stats['tda_calls'] for stats in self.result_.cluster_stats.values())
        total_rescued = sum(stats['rescued'] for stats in self.result_.cluster_stats.values())
        summary['tda_efficiency'] = {
            'total_tda_calls': total_tda_calls,
            'total_rescued': total_rescued,
            'efficiency_ratio': total_tda_calls / max(total_rescued, 1)
        }
        
        return summary

    def get_cluster_stats(self, cluster_id: int) -> Optional[Dict]:
        """
        Get per-cluster reassignment statistics.
        
        Parameters
        ----------
        cluster_id : int
            Cluster ID (0-indexed).
        
        Returns
        -------
        dict or None
            Cluster statistics if available.
        
        Raises
        ------
        ValueError
            If fit() has not been called.
        """
        if self.result_ is None:
            raise ValueError("Must call fit() first")
        return self.result_.cluster_stats.get(cluster_id, None)

    def __del__(self):
        """Clean up resources on object deletion."""
        try:
            if hasattr(self, 'ebm_engine'):
                self.ebm_engine.cleanup()
        except:
            pass