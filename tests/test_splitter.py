import numpy as np
from intercluster.rules import SimpleSplitter, UnsupervisedSplitter, CentroidSplitter


def test_with_simple_dataset(simple_dataset):
    splitter = SimpleSplitter()
    splitter.fit(X = simple_dataset, y = np.zeros(4))
    split_score, split_info = splitter.split(np.arange(len(simple_dataset)))
    split_features = split_info[0]
    split_weights = split_info[1]
    split_threshold = split_info[2]
    assert split_score == 9
    assert split_features == [0]
    assert split_weights == [1]
    assert split_threshold == 1
    
    
def test_with_simple_tiebreak(simple_tiebreak_dataset):
    splitter = SimpleSplitter()
    splitter.fit(X = simple_tiebreak_dataset, y = np.zeros(4, dtype = np.int32))
    
    samples = 1000
    split_features = np.zeros(samples)
    split_thresholds = np.zeros(samples)
    for i in range(samples):
        split_score, split_info = splitter.split(np.arange(len(simple_tiebreak_dataset)))
        assert split_score == 9
        split_features[i] = split_info[0][0]
        split_thresholds[i] = split_info[2]
        
    unique_features, counts_features = np.unique(split_features, return_counts=True)
    unique_thresholds, counts_thresholds = np.unique(split_thresholds, return_counts=True)
    
    assert np.array_equal(unique_features, [0, 1])
    assert np.array_equal(unique_thresholds, [1, 2])
    assert np.allclose(counts_features[0]/samples, 0.5, atol = 0.05, rtol = 0)
    assert np.allclose(counts_features[1]/samples, 0.5, atol = 0.05, rtol = 0)
    
    
def test_unsupervised_score(unsupervised_dataset):
    S1 = UnsupervisedSplitter(norm = 1, min_points_leaf = 1)
    S2 = UnsupervisedSplitter(norm = 2, min_points_leaf = 1)
    S1.fit(X = unsupervised_dataset)
    S2.fit(X = unsupervised_dataset)
    
    assert np.isclose(S1.score(np.arange(len(unsupervised_dataset))), 400, atol = 1e-5)
    assert np.isclose(S2.score(np.arange(len(unsupervised_dataset))), 800, atol = 1e-5)
    
    
def test_unsupervised_split(unsupervised_dataset):
    S = UnsupervisedSplitter(norm = 2, min_points_leaf = 1)
    S.fit(X = unsupervised_dataset)
    
    samples = 1000
    split_features = np.zeros(samples)
    split_thresholds = np.zeros(samples)
    
    for i in range(samples):
        split_score, split_info = S.split(np.arange(len(unsupervised_dataset)))
        split_features[i] = split_info[0][0]
        split_thresholds[i] = split_info[2]
        
    unique_features, counts_features = np.unique(split_features, return_counts=True)
    unique_thresholds, counts_thresholds = np.unique(split_thresholds, return_counts=True)
    
    assert np.array_equal(unique_features, [0, 1])
    assert np.array_equal(unique_thresholds, [0])
    
    assert np.allclose(counts_features[0]/samples, 0.5, atol = 0.05, rtol = 0)
    assert np.allclose(counts_features[1]/samples, 0.5, atol = 0.05, rtol = 0)
    
    
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