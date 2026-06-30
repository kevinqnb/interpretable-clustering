"""
Simple test/demonstration of the DensityFilterTree class.
This shows how to use the new DensityFilterTree for training on density-filtered data.
"""

import numpy as np
from sklearn.datasets import make_blobs
from intercluster.decision_trees import DensityFilterTree

# Generate synthetic data with some outliers
np.random.seed(42)
X, y_true = make_blobs(n_samples=300, centers=3, n_features=2, random_state=42, cluster_std=0.6)

# Add some outliers
n_outliers = 20
outlier_indices = np.random.choice(len(X), n_outliers, replace=False)
X[outlier_indices] += np.random.normal(0, 3, (n_outliers, 2))

# Convert labels to the expected format (List[Set[int]])
y = [set([int(label)]) for label in y_true]

# Create and fit a DensityFilterTree
tree = DensityFilterTree(
    eps=1.5,
    min_samples=5,
    criterion='entropy',
    max_depth=5,
    random_state=42
)

print("Fitting DensityFilterTree with DBSCAN pre-processing...")
print(f"Original dataset size: {len(X)}")

tree.fit(X, y)

print(f"Dataset size after DBSCAN filtering: {len(tree.X)}")
print(f"Number of inliers: {len(tree.X)}")
print(f"Number of outliers removed: {len(X) - len(tree.X)}")
print(f"Tree depth: {tree.depth}")
print(f"Number of leaves: {tree.leaf_count}")

# Test prediction on the full dataset (including the filtered outliers)
predictions = tree.predict(X)
print(f"\nPredictions made for full dataset (shape: {len(predictions)})")
print(f"First 5 predictions (as leaf labels): {predictions[:5]}")

print("\nDensityFilterTree successfully created and tested!")
