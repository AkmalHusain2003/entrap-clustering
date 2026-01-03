"""
topological_energy.py - Pure Python implementation of Topological Energy Computer
"""

import numpy as np
import gc
import logging
from scipy.spatial.distance import cdist
from ripser import ripser

from .utilities import validate_metric

logger = logging.getLogger(__name__)

# Constants
W0 = 1.0
W1 = 1.0
U0 = 0.5
U1 = 0.5


class Topological_Energy_Computer:
    """
    Compute topological energy via persistent homology.
    
    Uses Rips complex and persistent homology (H0, H1) to quantify
    cluster topology. Combines persistence lifetimes and entropy
    into a single topological energy score.
    
    Attributes
    ----------
    metric : str or callable
        Distance metric.
    use_memmap : bool
        Whether to use memory-mapped arrays for large distance matrices.
    w0, w1 : float
        Weights for H0 and H1 persistence energies.
    u0, u1 : float
        Weights for H0 and H1 entropy terms.
    """
    
    def __init__(self, metric='euclidean', use_memmap=True, **metric_params):
        """
        Initialize Topological_Energy_Computer.
        
        Parameters
        ----------
        metric : str, default='euclidean'
            Distance metric.
        use_memmap : bool, default=True
            Use memory-mapped arrays for large matrices.
        **metric_params
            Additional metric parameters.
        """
        
        self.metric = validate_metric(metric)
        self.metric_params = metric_params
        self.use_memmap = use_memmap
        
        # Import Memory_Manager only if needed
        if use_memmap:
            from .memory_manager import Memory_Manager
            self.memory_manager = Memory_Manager()
        else:
            self.memory_manager = None
            
        self.w0 = W0
        self.w1 = W1
        self.u0 = U0
        self.u1 = U1
    
    def compute_persistence(self, X, maxdim=1):
        """
        Compute persistence diagrams for H0 and H1 homology.
        
        Uses Rips complex and persistent homology to extract topological
        features. Filters out infinite deaths (H0) and infinite loops (H1).
        
        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Data points or distance matrix (if metric is callable).
        maxdim : int, default=1
            Maximum homology dimension to compute (0=connected components,
            1=loops/cycles).
        
        Returns
        -------
        dgm_h0 : ndarray, shape (n_components, 2)
            H0 persistence diagram (birth, death) for connected components.
        dgm_h1 : ndarray, shape (n_loops, 2)
            H1 persistence diagram for 1-dimensional holes.
        
        Notes
        -----
        Automatically uses memory mapping for n > 50000 if use_memmap=True.
        Filters out infinite death times before returning.
        """
        n = len(X)
        if n < 3:
            return np.array([[0.0, 0.0]]), np.array([])
        
        try:
            if callable(self.metric):
                dist = cdist(X, X, metric=self.metric, **self.metric_params)
                
                if self.use_memmap and n > 50000:
                    dist_memmap = self.memory_manager.create(dist.shape, dtype=np.float64, name='dist')
                    dist_memmap[:] = dist
                    dist_memmap.flush()
                    memmap_path = str(self.memory_manager._files[-1])
                    del dist_memmap
                    gc.collect()
                    dist_memmap = np.memmap(memmap_path, dtype=np.float64, mode='r', shape=dist.shape)
                    dgm = ripser(dist_memmap, distance_matrix=True, maxdim=maxdim)['dgms']
                    del dist_memmap
                    gc.collect()
                else:
                    dgm = ripser(dist, distance_matrix=True, maxdim=maxdim)['dgms']
            else:
                dgm = ripser(X, distance_matrix=False, maxdim=maxdim)['dgms']
            
            dgm_h0 = dgm[0]
            finite_mask = ~np.isinf(dgm_h0[:, 1])
            dgm_h0_finite = dgm_h0[finite_mask]
            if len(dgm_h0_finite) == 0:
                dgm_h0_finite = np.array([[0.0, 0.0]])
            
            dgm_h1 = np.array([])
            if maxdim >= 1 and len(dgm) > 1:
                dgm_h1 = dgm[1]
                finite_mask_h1 = ~np.isinf(dgm_h1[:, 1])
                dgm_h1 = dgm_h1[finite_mask_h1]
            
            return dgm_h0_finite, dgm_h1
        
        except Exception as e:
            logger.error(f"Persistence computation failed: {e}")
            raise
    
    def compute_entropy(self, lifetimes):
        """
        Compute Shannon entropy of persistence lifetimes.
        
        Treats lifetimes as a probability distribution and computes
        entropy: H = -Σ p_i log(p_i). High entropy indicates uniform
        persistence across features; low entropy indicates localized
        significant features.
        
        Parameters
        ----------
        lifetimes : ndarray, shape (n_features,)
            Persistence lifetimes (birth-death differences).
        
        Returns
        -------
        float
            Shannon entropy in nats. Returns 0 if no lifetimes.
        """
        lifetimes = lifetimes[lifetimes > 1e-12]
        if len(lifetimes) == 0:
            return 0.0
        
        total = np.sum(lifetimes)
        if total <= 1e-12:
            return 0.0
        
        p = lifetimes / total
        entropy = -np.sum(p * np.log(p + 1e-12))
        return float(entropy)
    
    def compute_raw_topological_energy(self, X):
        """
        Compute raw topological energy from persistence diagrams.
        
        Combines H0 lifetime sum, H1 lifetime sum, and entropies
        of both into a single energy score:
        T_raw = w0*E0 + w1*E1 + u0*H(H0) + u1*H(H1).
        
        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Data points.
        
        Returns
        -------
        dict
            Dictionary containing:
            - T_raw : float - Total topological energy
            - E0 : float - H0 persistence energy
            - E1 : float - H1 persistence energy
            - H0_entropy : float - Entropy of H0 lifetimes
            - H1_entropy : float - Entropy of H1 lifetimes
            - n_components_h0 : int - Number of connected components
            - n_loops_h1 : int - Number of 1-dimensional holes
        """
        dgm_h0, dgm_h1 = self.compute_persistence(X, maxdim=1)
        
        lifetimes_h0 = dgm_h0[:, 1] - dgm_h0[:, 0]
        lifetimes_h0 = lifetimes_h0[lifetimes_h0 > 1e-12]
        E0 = float(np.sum(lifetimes_h0)) if len(lifetimes_h0) > 0 else 0.0
        
        if len(dgm_h1) > 0:
            lifetimes_h1 = dgm_h1[:, 1] - dgm_h1[:, 0]
        else:
            lifetimes_h1 = np.array([])
        lifetimes_h1 = lifetimes_h1[lifetimes_h1 > 1e-12]
        E1 = float(np.sum(lifetimes_h1)) if len(lifetimes_h1) > 0 else 0.0
        
        H0_entropy = self.compute_entropy(lifetimes_h0)
        H1_entropy = self.compute_entropy(lifetimes_h1) if len(lifetimes_h1) > 0 else 0.0
        
        T_raw = (self.w0 * E0 + self.w1 * E1 +
                self.u0 * H0_entropy + self.u1 * H1_entropy)
        
        return {
            'T_raw': T_raw,
            'E0': E0,
            'E1': E1,
            'H0_entropy': H0_entropy,
            'H1_entropy': H1_entropy,
            'n_components_h0': len(lifetimes_h0),
            'n_loops_h1': len(lifetimes_h1)
        }
    
    def cleanup(self):
        """Release memory-mapped resources."""
        if self.memory_manager:
            self.memory_manager.cleanup()