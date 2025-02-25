import numpy as np
from intercluster.rules.splitters import (
    CentroidSplitter,
    ImmSplitter,
    InformationGainSplitter,
    SimpleSplitter,
    UnsupervisedSplitter,
)


def test_centroid_score(centroid_dataset):
    centroids = np.array([
        [0, 0],
        [5, 5]
    ])
    
    S1 = CentroidSplitter(centers = centroids, norm = 1, min_points_leaf = 1)
    S2 = CentroidSplitter(centers = centroids, norm = 2, min_points_leaf = 1)
    S1.fit(X = centroid_dataset)
    S2.fit(X = centroid_dataset)
    
    true_center_dists1 = np.array([
        [7, 3],
        [17, 7],
        [23, 13],
        [31, 21]
    ])
    
    true_center_dists2 = np.array([
        [5**2, 5],
        [13**2, 7**2],
        [17**2, 3**2 + 10**2],
        [25**2, 2**2 + 19**2]
    ])
    
    assert np.array_equal(S1.center_dists, true_center_dists1)
    assert np.array_equal(S2.center_dists, true_center_dists2)
    
    
def test_centroid_split(centroid_dataset):
    centroids = np.array([
        [2, 3],
        [8, 25]
    ])
    
    S = CentroidSplitter(centers = centroids, norm = 2, min_points_leaf = 1)
    S.fit(X = centroid_dataset)
    
    samples = 1000
    split_features = np.zeros(samples)
    split_thresholds = np.zeros(samples)
    
    for i in range(samples):
        split_score, split_info = S.split(np.arange(len(centroid_dataset)))
        split_features[i] = split_info[0][0]
        split_thresholds[i] = split_info[2]
        
    unique_features, counts_features = np.unique(split_features, return_counts=True)
    unique_thresholds, counts_thresholds = np.unique(split_thresholds, return_counts=True)
    
    assert np.array_equal(unique_features, [0, 1])
    assert np.array_equal(unique_thresholds, [5, 12])
    
    assert np.allclose(counts_features[0]/samples, 0.5, atol = 0.05, rtol = 0)
    assert np.allclose(counts_features[1]/samples, 0.5, atol = 0.05, rtol = 0)