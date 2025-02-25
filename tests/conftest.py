import numpy as np
from sklearn import cluster, datasets, preprocessing
import pytest
from intercluster.utils import labels_format
    
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
    
    
@pytest.fixture
def example_dataset():
    # simple dataset with 3 blob clusters
    n = 2000
    k = 3
    random_state = 170
    data, labels = datasets.make_blobs(
        n_samples=n, cluster_std=[1.0, 2.5, 0.5], random_state=random_state
    )
    data = preprocessing.MinMaxScaler().fit_transform(data)
    kmeans = cluster.KMeans(n_clusters=k, random_state=random_state).fit(data)
    
    return data, labels_format(kmeans.labels_)


@pytest.fixture
def satisfies_dataset():
    return np.array([
        [0, 1],
        [0, 1],
        [1, 2],
        [1, 2],
        [1, 3],
        [1, 3],
        [2, 1],
        [2, 1]
    ])