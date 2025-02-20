import numpy as np
from intercluster.experiments import DistanceRatio


def test_distance_ratio():
    dratio = DistanceRatio()
    X = np.zeros((4, 2))
    centers = np.array([
        [3, 4],
        [6, 8],
        [5, 12],
        [9, 12]
    ])
    center_dist_arr = np.array([
        [5, 5, 5, 5],
        [10, 10, 10, 10],
        [13, 13, 13, 13],
        [15, 15, 15, 15]
    ]).T

    # First test some edge cases:
    # 1) all assigned to a single cluster
    assignment = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
    ])

    assert np.isnan(dratio(X, assignment, centers))

    # 2) all assigned to a multiple clusters
    assignment = np.array([
        [1, 0, 0, 1],
        [0, 1, 0, 1],
        [1, 0, 0, 1],
        [0, 0, 1, 1],
    ])

    assert dratio(X, assignment, centers) == 1


    # 3) all uncovered
    assignment = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ])
    assert dratio(X, assignment, centers) == 1


    # Now testing to make sure values themselves make sense.
    X = np.zeros((4, 2))
    X[0,:] = [10, 12]
    X[1,:] = [11, 12]
    centers = np.array([
        [3, 4],
        [6, 8],
        [5, 12],
        [9, 12]
    ])

    assignment = np.array([
        [1, 0, 0, 1],
        [0, 1, 0, 1],
        [1, 0, 0, 0],
        [0, 0, 1, 0],
    ])
    all_points_mean = np.mean([5, 6/2, 10/5, 10/5])
    overlap_points_mean = np.mean([5, 6/2])
    assert dratio(X, assignment, centers) == (all_points_mean/overlap_points_mean)


    # Test for a point which lies directly upon a cluster. 
    X = np.zeros((4, 2))
    X[0,:] = [9, 12]
    X[1,:] = [11, 12]
    centers = np.array([
        [3, 4],
        [6, 8],
        [5, 12],
        [9, 12]
    ])

    assignment = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 1],
        [1, 0, 0, 0],
        [0, 0, 1, 0],
    ])
    all_points_mean = np.mean([np.inf, 6/2, 10/5, 10/5])
    overlap_points_mean = np.mean([6/2])
    assert dratio(X, assignment, centers) == (all_points_mean/overlap_points_mean)