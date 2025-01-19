import numpy as np
from intercluster.utils import *

def test_tiebreak():
    # Test proxy tiebreak
    scores = np.array([1, 1, 1, 1, 1])
    proxy = np.array([1, 2, 3, 4, 5])
    assert np.array_equal(tiebreak(scores, proxy), [0, 1, 2, 3, 4])
    
    # Test random tiebreak
    scores = np.array([1, 1, 1, 2, 2])
    proxy = np.array([1, 1, 2, 3, 4])
    samples = 1000
    lowest = np.zeros(samples)
    lowest_with_proxy = np.zeros(samples)
    for i in range(samples):
        lowest[i] = tiebreak(scores)[0]
        lowest_with_proxy[i] = tiebreak(scores, proxy)[0]
        
    _, counts = np.unique(lowest, return_counts=True)
    _, counts_with_proxy = np.unique(lowest_with_proxy, return_counts=True)
    
    assert len(counts) == 3
    assert np.allclose(counts[0]/samples, 1/3, atol = 0.05, rtol = 0)
    assert np.allclose(counts[1]/samples, 1/3, atol = 0.05, rtol = 0)
    assert np.allclose(counts[2]/samples, 1/3, atol = 0.05, rtol = 0)
    
    assert len(counts_with_proxy) == 2
    assert np.allclose(counts_with_proxy[0]/samples, 1/2, atol = 0.05, rtol = 0)
    assert np.allclose(counts_with_proxy[1]/samples, 1/2, atol = 0.05, rtol = 0)