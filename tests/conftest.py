import numpy as np
import pytest
    
@pytest.fixture
def simple_dataset():
    return np.array([
        [1, 2],
        [3, 1],
        [5, 3],
        [7, 4]
    ], dtype = np.float64)
    
@pytest.fixture
def simple_tiebreak_dataset():
    return np.array([
        [1, 2],
        [3, 3],
        [5, 3],
        [7, 4]
    ], dtype = np.float64)
    
@pytest.fixture
def unsupervised_dataset():
    X = np.zeros((100, 2))
    X[:50, 0] = 4
    X[50:, 1] = 4
    return X


@pytest.fixture
def centroid_dataset():
    return np.array([
        [3, 4],
        [5, 12],
        [8, 15],
        [7, 24]
    ])