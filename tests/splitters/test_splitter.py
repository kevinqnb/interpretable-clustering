import numpy as np
from intercluster.splitters import (
    CentroidSplitter,
    ImmSplitter,
    InformationGainSplitter,
    SimpleSplitter,
    UnsupervisedSplitter,
)


def test_with_simple_dataset(simple_dataset):
    splitter = SimpleSplitter()
    splitter.fit(X = simple_dataset, y = np.zeros(4))
    gain, split_info = splitter.split(np.arange(len(simple_dataset)))
    split_features = split_info[0]
    split_weights = split_info[1]
    split_threshold = split_info[2]
    assert gain == 7 - 9
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
        gain, split_info = splitter.split(np.arange(len(simple_tiebreak_dataset)))
        assert gain == 7 - 9
        split_features[i] = split_info[0][0]
        split_thresholds[i] = split_info[2]
        
    unique_features, counts_features = np.unique(split_features, return_counts=True)
    unique_thresholds, counts_thresholds = np.unique(split_thresholds, return_counts=True)
    
    assert np.array_equal(unique_features, [0, 1])
    assert np.array_equal(unique_thresholds, [1, 2])
    assert np.allclose(counts_features[0]/samples, 0.5, atol = 0.05, rtol = 0)
    assert np.allclose(counts_features[1]/samples, 0.5, atol = 0.05, rtol = 0)