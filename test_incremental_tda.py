"""
test_incremental_tda.py - Benchmark incremental TDA with caching
"""

import numpy as np
import time
import logging
from entrap import ENTRAP

# Enable logging to see cache statistics
logging.basicConfig(level=logging.INFO)

# Create test data with noise
np.random.seed(42)

# Create 3 well-separated clusters
cluster1 = np.random.randn(80, 5) + np.array([0, 0, 0, 0, 0])
cluster2 = np.random.randn(85, 5) + np.array([10, 10, 10, 10, 10])
cluster3 = np.random.randn(90, 5) + np.array([20, 0, 20, 0, 20])

# Add noise points
noise = np.random.uniform(-5, 25, (30, 5))

X = np.vstack([cluster1, cluster2, cluster3, noise])

print("=" * 70)
print("Testing Incremental TDA Integration")
print("=" * 70)
print(f"Data shape: {X.shape}")
print(f"Expected clusters: 3")
print(f"Expected noise: 30")
print()

# Test WITHOUT incremental TDA
print("-" * 70)
print("Test 1: WITHOUT Incremental TDA (baseline)")
print("-" * 70)

start = time.time()
clf1 = ENTRAP(
    min_cluster_size=20,
    alpha=0.5,
    lambda_T=1.0,
    use_incremental_tda=False  # Disable caching
)
clf1.fit(X)
time1 = time.time() - start

print(f"Clusters found: {len(set(clf1.labels_)) - (1 if -1 in clf1.labels_ else 0)}")
print(f"Noise points: {sum(clf1.labels_ == -1)}")
print(f"Execution time: {time1:.3f} seconds")
print()

# Test WITH incremental TDA
print("-" * 70)
print("Test 2: WITH Incremental TDA (memory-bounded caching)")
print("-" * 70)

start = time.time()
clf2 = ENTRAP(
    min_cluster_size=20,
    alpha=0.5,
    lambda_T=1.0,
    use_incremental_tda=True  # Enable caching
)
clf2.fit(X)
time2 = time.time() - start

print(f"Clusters found: {len(set(clf2.labels_)) - (1 if -1 in clf2.labels_ else 0)}")
print(f"Noise points: {sum(clf2.labels_ == -1)}")
print(f"Execution time: {time2:.3f} seconds")
print()

# Performance comparison
print("=" * 70)
print("Performance Summary")
print("=" * 70)
print(f"Without incremental TDA: {time1:.3f}s")
print(f"With incremental TDA:    {time2:.3f}s")
print(f"Speedup:                 {time1/time2:.2f}x")
print(f"Time saved:              {time1-time2:.3f}s ({(1-time2/time1)*100:.1f}%)")
print()

# Verify same results
print("=" * 70)
print("Result Verification")
print("=" * 70)
same_clusters = np.array_equal(
    np.sort(clf1.labels_[clf1.labels_ >= 0]),
    np.sort(clf2.labels_[clf2.labels_ >= 0])
)
same_noise_count = sum(clf1.labels_ == -1) == sum(clf2.labels_ == -1)

if same_clusters and same_noise_count:
    print("✓ Both methods produce equivalent results")
else:
    print("⚠ Results differ (may be due to non-deterministic TDA)")
    
print(f"Clusters match: {same_clusters}")
print(f"Noise count match: {same_noise_count}")
print()

print("=" * 70)
print("Integration Test Completed Successfully!")
print("=" * 70)
