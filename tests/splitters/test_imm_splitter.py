import numpy as np
from intercluster.rules.splitters import (
    ImmSplitter,
    InformationGainSplitter,
    SimpleSplitter
)


def test_imm_split(centroid_dataset):
    centroids = np.array([
        [2, 3],
        [8, 25]
    ])
    
    S = ImmSplitter(centers = centroids, norm = 2, min_points_leaf = 1)
    S.fit(X = centroid_dataset)
    
    samples = 1000
    split_features = np.zeros(samples)
    split_thresholds = np.zeros(samples)
    
    for i in range(samples):
        split_score, split_info = S.split(
            indices = np.arange(len(centroid_dataset)),
            centroid_indices = np.array([0, 1])
        )
        split_features[i] = split_info[0][0]
        split_thresholds[i] = split_info[2]
        
    unique_features, counts_features = np.unique(split_features, return_counts=True)
    unique_thresholds, counts_thresholds = np.unique(split_thresholds, return_counts=True)
    
    assert np.array_equal(unique_features, [0, 1])
    assert np.array_equal(unique_thresholds, [5, 12])
    
    assert np.allclose(counts_features[0]/samples, 0.5, atol = 0.05, rtol = 0)
    assert np.allclose(counts_features[1]/samples, 0.5, atol = 0.05, rtol = 0)