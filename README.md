# ENTRAP: ENergy-based Topological Rescue of Ambiguous Points
    
Two-stage clustering method: 
1. Initial density-based clustering via HDBSCAN.
2. Energy-based refinement to rescue noise points that belong to clusters.
Uses geometric (Mahalanobis) + topological (persistent homology) energies
for principled point reassignment.
