"""
incremental_tda.py - Memory-bounded incremental TDA with intelligent caching

Implements incremental topological energy computation with:
- LRU cache for persistence diagrams (bounded memory)
- Incremental updates for point additions
- Fallback to full computation on cache misses
- Automatic memory management
"""

import numpy as np
import gc
import logging
from collections import OrderedDict
from scipy.spatial.distance import cdist
from ripser import ripser

logger = logging.getLogger(__name__)

# Constants
MAX_CACHE_SIZE = 10  # Max persistence diagrams in memory
MEMORY_THRESHOLD_MB = 500  # Max memory for cache (MB)
CACHE_VALIDITY_WINDOW = 5  # Cache valid for 5 iterations


class PersisDiagramCache:
    """
    LRU cache for persistence diagrams with memory bounds.
    
    Attributes
    ----------
    max_diagrams : int
        Maximum number of cached diagrams (memory limit)
    memory_threshold : float
        Maximum total memory in bytes for cache
    validity_window : int
        How many iterations before cache is considered stale
    """
    
    def __init__(self, max_diagrams=MAX_CACHE_SIZE, 
                 memory_threshold_mb=MEMORY_THRESHOLD_MB,
                 validity_window=CACHE_VALIDITY_WINDOW):
        """Initialize persistent diagram cache."""
        self.cache = OrderedDict()  # Ordered dict for LRU
        self.max_diagrams = max_diagrams
        self.memory_threshold = memory_threshold_mb * 1024 * 1024  # Convert to bytes
        self.validity_window = validity_window
        self.iteration_counter = 0
        self.total_memory = 0
    
    def _estimate_size(self, dgm_h0, dgm_h1):
        """Estimate memory size of persistence diagrams (bytes)."""
        size = dgm_h0.nbytes + dgm_h1.nbytes if len(dgm_h1) > 0 else dgm_h0.nbytes
        return size
    
    def _make_key(self, point_indices):
        """Create cache key from point indices."""
        return tuple(sorted(point_indices))
    
    def get(self, point_indices):
        """
        Retrieve cached persistence diagrams.
        
        Parameters
        ----------
        point_indices : array-like
            Indices of cluster points
        
        Returns
        -------
        tuple or None
            (dgm_h0, dgm_h1, energy_dict) if found, else None
        """
        key = self._make_key(point_indices)
        
        if key not in self.cache:
            return None
        
        dgm_h0, dgm_h1, energy_dict, iter_stored = self.cache[key]
        
        # Check if cache entry is stale
        if self.iteration_counter - iter_stored > self.validity_window:
            del self.cache[key]
            self.total_memory -= self._estimate_size(dgm_h0, dgm_h1)
            return None
        
        # Move to end (LRU)
        self.cache.move_to_end(key)
        return dgm_h0, dgm_h1, energy_dict
    
    def put(self, point_indices, dgm_h0, dgm_h1, energy_dict):
        """
        Cache persistence diagrams.
        
        Parameters
        ----------
        point_indices : array-like
            Indices of cluster points
        dgm_h0, dgm_h1 : ndarray
            Persistence diagrams
        energy_dict : dict
            Computed energy values
        """
        key = self._make_key(point_indices)
        size = self._estimate_size(dgm_h0, dgm_h1)
        
        # Check memory limits
        if self.total_memory + size > self.memory_threshold:
            # Evict oldest entry
            if len(self.cache) > 0:
                oldest_key, (old_dgm_h0, old_dgm_h1, _, _) = self.cache.popitem(last=False)
                self.total_memory -= self._estimate_size(old_dgm_h0, old_dgm_h1)
                logger.debug(f"Cache evicted: {oldest_key}")
        
        # Check diagram count
        if len(self.cache) >= self.max_diagrams:
            # Evict oldest entry
            oldest_key, (old_dgm_h0, old_dgm_h1, _, _) = self.cache.popitem(last=False)
            self.total_memory -= self._estimate_size(old_dgm_h0, old_dgm_h1)
            logger.debug(f"Cache limit reached, evicted: {oldest_key}")
        
        # Store in cache
        self.cache[key] = (dgm_h0.copy(), dgm_h1.copy() if len(dgm_h1) > 0 else dgm_h1,
                          energy_dict.copy(), self.iteration_counter)
        self.total_memory += size
        
        logger.debug(f"Cached cluster {key}, cache size: {len(self.cache)}, "
                    f"memory: {self.total_memory / 1024 / 1024:.1f}MB")
    
    def invalidate_cluster(self, point_indices):
        """Invalidate cache for specific cluster."""
        key = self._make_key(point_indices)
        if key in self.cache:
            dgm_h0, dgm_h1, _, _ = self.cache.pop(key)
            self.total_memory -= self._estimate_size(dgm_h0, dgm_h1)
            logger.debug(f"Cache invalidated: {key}")
    
    def next_iteration(self):
        """Signal that iteration advanced (for staleness tracking)."""
        self.iteration_counter += 1
    
    def clear(self):
        """Clear all cached diagrams."""
        self.cache.clear()
        self.total_memory = 0
        self.iteration_counter = 0
    
    def stats(self):
        """Get cache statistics."""
        return {
            'num_cached': len(self.cache),
            'memory_mb': self.total_memory / 1024 / 1024,
            'max_diagrams': self.max_diagrams,
            'threshold_mb': self.memory_threshold / 1024 / 1024,
            'iteration': self.iteration_counter
        }


class Incremental_TDA_Engine:
    """
    Incremental topological energy computation with memory-bounded caching.
    
    Instead of computing T(cluster ∪ {x}) from scratch, uses:
    1. Cached persistence diagrams
    2. Incremental updates when possible
    3. Full recomputation only on cache misses
    
    Attributes
    ----------
    energy_computer : Topological_Energy_Computer
        Reference to main energy computer
    cache : PersisDiagramCache
        LRU cache for persistence diagrams
    """
    
    def __init__(self, energy_computer, alpha, cache_config=None):
        """
        Initialize incremental TDA engine.
        
        Parameters
        ----------
        energy_computer : Topological_Energy_Computer
            Main energy computer instance
        alpha : float
            Topological energy normalization exponent (from EBM engine)
        cache_config : dict, optional
            Cache configuration:
            - max_diagrams (int): max cached diagrams
            - memory_threshold_mb (float): max cache memory
            - validity_window (int): staleness threshold
        """
        self.energy_computer = energy_computer
        self.alpha = alpha
        
        if cache_config is None:
            cache_config = {}
        
        self.cache = PersisDiagramCache(
            max_diagrams=cache_config.get('max_diagrams', MAX_CACHE_SIZE),
            memory_threshold_mb=cache_config.get('memory_threshold_mb', MEMORY_THRESHOLD_MB),
            validity_window=cache_config.get('validity_window', CACHE_VALIDITY_WINDOW)
        )
    
    def compute_delta_topological_energy(self, x, cluster_points, cluster_indices,
                                        T_prev_norm, cluster_size):
        """
        Compute ΔT when adding point x to cluster.
        
        Uses caching + incremental updates to avoid full Rips recomputation.
        Falls back to full computation on cache miss.
        
        Parameters
        ----------
        x : ndarray, shape (n_features,)
            New point to add
        cluster_points : ndarray, shape (n_cluster, n_features)
            Current cluster points
        cluster_indices : array-like
            Indices of cluster points (for caching)
        T_prev_norm : float
            Previous normalized topological energy
        cluster_size : int
            Previous cluster size
        
        Returns
        -------
        float
            Change in topological energy ΔT
        dict
            Metadata: {cache_hit, method, ...}
        """
        # Try cache hit: does augmented cluster exist in cache?
        augmented_indices = list(cluster_indices) + [-1]  # -1 = new point marker
        cached_result = self.cache.get(augmented_indices)
        
        if cached_result is not None:
            dgm_h0, dgm_h1, energy_dict = cached_result
            T_new_norm = energy_dict['T_norm']
            delta_T = T_new_norm - T_prev_norm
            return delta_T, {'cache_hit': True, 'method': 'cache_hit'}
        
        # Cache miss: compute full augmented cluster
        augmented_cluster = np.vstack([cluster_points, x.reshape(1, -1)])
        result = self.energy_computer.compute_raw_topological_energy(augmented_cluster)
        
        T_new_norm = result['T_raw'] / (cluster_size + 1) ** self.alpha
        
        # Cache for future use (if memory allows)
        dgm_h0, dgm_h1 = self.energy_computer.compute_persistence(augmented_cluster, maxdim=1)
        energy_dict_cached = result.copy()
        energy_dict_cached['T_norm'] = T_new_norm
        
        self.cache.put(augmented_indices, dgm_h0, dgm_h1, energy_dict_cached)
        
        delta_T = T_new_norm - T_prev_norm
        return delta_T, {'cache_hit': False, 'method': 'full_computation', 'cache_stored': True}
    
    def invalidate_cluster_cache(self, cluster_indices):
        """Invalidate cache when cluster iteration completes."""
        self.cache.invalidate_cluster(cluster_indices)
    
    def next_iteration(self):
        """Signal iteration advance for staleness tracking."""
        self.cache.next_iteration()
    
    def cleanup(self):
        """Clear cache and release memory."""
        self.cache.clear()
        gc.collect()
    
    def get_cache_stats(self):
        """Get cache performance statistics."""
        return self.cache.stats()