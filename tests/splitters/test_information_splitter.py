import numpy as np
from intercluster.decision_trees.splitters import (
    ImmSplitter,
    InformationGainSplitter,
    SimpleSplitter,
)
from intercluster.measurement_utils import entropy
from intercluster.utils import labels_format


def test_information_split(simple_dataset):
    S = InformationGainSplitter()
    y = np.array([0, 0, 0, 1])
    S.fit(X = simple_dataset, y = labels_format(y))

    samples = 1000
    split_features = np.zeros(samples)
    split_thresholds = np.zeros(samples)
    for i in range(samples):
        gain, condition = S.split(np.arange(len(simple_dataset)))
        assert gain == entropy(y)
        split_features[i] = condition.features[0]
        split_thresholds[i] = condition.threshold
           
    unique_features, counts_features = np.unique(split_features, return_counts=True)
    unique_thresholds, counts_thresholds = np.unique(split_thresholds, return_counts=True)
    
    assert np.array_equal(unique_features, [0, 1])
    assert np.isclose(counts_features[0]/samples, 0.5, atol = 0.05, rtol = 0)
    assert np.isclose(counts_features[1]/samples, 0.5, atol = 0.05, rtol = 0)
    
    assert np.array_equal(unique_thresholds, [3, 5])
    assert np.isclose(counts_thresholds[0]/samples, 0.5, atol = 0.05, rtol = 0)
    assert np.isclose(counts_thresholds[1]/samples, 0.5, atol = 0.05, rtol = 0)
    
    
    
    
    
    