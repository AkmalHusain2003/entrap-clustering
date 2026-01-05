import numpy as np
from entrap import ENTRAP

# Create synthetic data with clusters
np.random.seed(42)
n_samples = 200
n_features = 5

# Create 3 clusters
cluster1 = np.random.randn(60, n_features) + np.array([0, 0, 0, 0, 0])
cluster2 = np.random.randn(70, n_features) + np.array([5, 5, 5, 5, 5])
cluster3 = np.random.randn(50, n_features) + np.array([-5, -5, -5, -5, -5])

# Add some noise
noise = np.random.uniform(-10, 10, (20, n_features))

X = np.vstack([cluster1, cluster2, cluster3, noise])

print("=" * 60)
print("ENTRAP Clustering Test")
print("=" * 60)
print(f"Data shape: {X.shape}")
print(f"Expected clusters: 3")
print(f"Expected noise points: 20")
print()

# Run ENTRAP
print("Fitting ENTRAP...")
clf = ENTRAP(min_cluster_size=20, alpha=0.5, lambda_T=1.0, use_incremental_tda=True)
labels = clf.fit_predict(X)

print("✓ Fitting complete!")
print()

# Get summary
summary = clf.get_summary()

print("Results:")
print("-" * 60)
print(f"Number of clusters found: {summary['n_clusters']}")
print(f"Number of noise points: {summary['n_noise']}")
print(f"Points rescued: {summary['noise_rescued']}")
print(f"Cluster sizes: {summary['cluster_sizes']}")
print(f"Execution time: {summary['execution_time']:.3f} seconds")
print(f"Empirical noise energy: {summary['empirical_noise_energy']:.6f}")
print(f"TDA efficiency ratio: {summary['tda_efficiency']['efficiency_ratio']:.2f}")
print()

print("=" * 60)
print("Test completed successfully!")
print("=" * 60)
