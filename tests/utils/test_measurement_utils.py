import numpy as np
import pytest
from sklearn.metrics.pairwise import pairwise_distances
from intercluster.measurement_utils import (
    mode,
    entropy,
    coverage,
    overlap,
    center_dists,
    kmeans_cost,
    distance_ratio_score,
    silhouette_score,
    _point_silhouette_score,
    mistakes,
    clustering_distance,
    rule_pairwise_difference,
)


####################################################################################################


def test_mode():
    arr = np.array([1,2,2,3,3,4,4,4,5,7])
    assert mode(arr) == 4

    # Tie-break case: two values (2 and 3) each appear equally often as the mode.
    samples = 1000
    arr = np.array([1,2,2,3,3,4,4,4,5,3])
    sampled_modes = np.zeros(samples)
    for i in range(samples):
        sampled_modes[i] = mode(arr)

    _,counts = np.unique(sampled_modes, return_counts = True)
    assert np.isclose(counts[0]/samples, 0.5, atol = 0.05, rtol = 0)
    assert np.isclose(counts[1]/samples, 0.5, atol = 0.05, rtol = 0)


####################################################################################################


def test_entropy():
    assert np.isclose(entropy(np.array([1, 1, 1, 1, 1, 1])), 0, atol = 1e-5)
    assert np.isclose(entropy(np.array([1, 0, 1, 0, 1, 0])), 1, atol = 1e-5)

    uneven = np.array([2,2,2,3,3,3,2,2,2,2])
    assert np.isclose(entropy(uneven), 0.8816, atol = 0.05, rtol = 0)

    multiclass = np.array([1,1,2,2,3,3,4,4,5,5])
    assert np.isclose(entropy(multiclass),  -np.log2(1/5), atol = 0.05, rtol = 0)


####################################################################################################


def test_coverage():
    assignment1 = np.ones((4,4))
    assignment2 = np.zeros((4,4))
    assignment3 = np.array([
        [1,0,1,0],
        [0,1,0,0],
        [1,1,1,1],
        [0,0,0,0]
    ])

    assert coverage(assignment1) == 1
    assert coverage(assignment2) == 0
    assert coverage(assignment3) == 3/4

    # percentage=False returns a raw covered-point count instead of a fraction.
    assert coverage(assignment3, percentage = False) == 3

    # Weighted coverage: only covered points contribute their weight.
    weights = np.array([2., 1., 1., 5.])
    assert coverage(assignment3, weights = weights, percentage = False) == 2 + 1 + 1

    with pytest.raises(ValueError):
        coverage(assignment3, weights = np.array([1., 2.]))


####################################################################################################


def test_overlap():
    assignment1 = np.ones((4,4))
    assignment2 = np.zeros((4,4))
    assignment3 = np.array([
        [1,0,1,0],
        [0,1,0,0],
        [1,1,1,1],
        [0,0,0,0]
    ])

    assert overlap(assignment1) == 4
    assert np.isnan(overlap(assignment2))
    assert overlap(assignment3) == 7/3


####################################################################################################


def test_center_dists():
    X = np.zeros((4, 2))
    centers = np.array([
        [3, 4],
        [6, 8],
        [5, 12],
        [9, 12]
    ])

    center_dist_arr = center_dists(X, centers, norm = 2, square = False)
    assert center_dist_arr.shape == (4,4)

    test_arr = np.array([
        [5, 5, 5, 5],
        [10, 10, 10, 10],
        [13, 13, 13, 13],
        [15, 15, 15, 15]
    ]).T
    assert np.all(center_dist_arr == test_arr)

    center_dist_arr_square = center_dists(X, centers, norm = 2, square = True)
    assert center_dist_arr_square.shape == (4,4)
    assert np.all(center_dist_arr_square == test_arr ** 2)

    test_arr2 = np.array([
        [7, 7, 7, 7],
        [14, 14, 14, 14],
        [17, 17, 17, 17],
        [21, 21, 21, 21]
    ]).T
    center_dist_arr = center_dists(X, centers, norm = 1, square = False)
    assert center_dist_arr.shape == (4,4)
    assert np.all(center_dist_arr == test_arr2)

    X = np.array([
        [1],
        [2],
        [3]
    ])
    centers = np.array([
        [0],
        [0],
        [0]
    ])
    center_dist_arr_square = center_dists(X, centers, norm = 1, square = True)
    assert center_dist_arr_square.shape == (3,3)
    test_arr = np.array([
        [1,1,1],
        [2,2,2],
        [3,3,3]
    ])
    assert np.all(center_dist_arr_square == test_arr)

    with pytest.raises(ValueError):
        center_dists(X, centers, norm = 3)


####################################################################################################


def test_kmeans_cost():
    n = 4
    k = 4
    X = np.zeros((4, 2))
    centers = np.array([
        [3, 4],
        [6, 8],
        [5, 12],
        [9, 12]
    ])

    test_arr = np.array([
        [5, 5, 5, 5],
        [10, 10, 10, 10],
        [13, 13, 13, 13],
        [15, 15, 15, 15]
    ]).T**2

    assignment1 = np.ones((n,k))
    cost1 = kmeans_cost(X, centers, assignment1, average = False, normalize = False, norm = 2)
    assert cost1 == np.sum(test_arr)

    cost2 = kmeans_cost(X, centers, assignment1, average = False, normalize = True, norm = 2)
    assert cost2 == (np.sum(test_arr)/n)

    cost3 = kmeans_cost(X, centers, assignment1, average = True, normalize = True, norm = 2)
    assert cost3 == (np.sum(test_arr)/(n*k))

    assignment2 = np.array([
        [1,0,1,0],
        [0,1,0,0],
        [1,1,1,1],
        [0,0,0,0]
    ])

    cost4 = kmeans_cost(X, centers, assignment2, average = False, normalize = False, norm = 2)
    assert cost4 == np.sum(test_arr * assignment2)

    cost5 = kmeans_cost(X, centers, assignment2, average = False, normalize = True, norm = 2)
    assert cost5 == (np.sum(test_arr * assignment2) / 3)

    cost6 = kmeans_cost(X, centers, assignment2, average = True, normalize = True, norm = 2)
    point_costs = np.sum(test_arr * assignment2, axis = 1)
    assert cost6 == (np.sum(point_costs / np.array([2,1,4,1])) / 3)

    assignment3 = np.zeros((4,4))
    cost7 = kmeans_cost(X, centers, assignment3, average = False, normalize = False, norm = 2)
    assert cost7 == 0

    cost8 = kmeans_cost(X, centers, assignment3, average = True, normalize = True, norm = 2)
    assert cost8 == np.inf

    with pytest.raises(ValueError):
        kmeans_cost(X, centers, np.ones((n + 1, k)))


####################################################################################################


def test_distance_ratio_score():
    # All four points sit at the origin; distances to the 4 centers are 5, 10, 13, 15
    # (same setup as test_center_dists), so closest=5, second-closest=10 for every point.
    X = np.zeros((4, 2))
    centers = np.array([
        [3, 4],
        [6, 8],
        [5, 12],
        [9, 12]
    ])
    expected = np.exp(-5 * (5/10))
    assert np.allclose(distance_ratio_score(X, centers), np.full(4, expected), atol = 1e-8)

    # A point that sits exactly on its closest center gets ratio 0 -> score exp(0) = 1
    # (its "confidently assigned" case); a point far from everything but still closer
    # to one center than the rest gets an intermediate score.
    X2 = np.zeros((2, 2))
    X2[0,:] = [9, 12]    # exactly on centers[3]
    X2[1,:] = [20, 12]
    scores = distance_ratio_score(X2, centers)
    assert np.isclose(scores[0], 1.0, atol = 1e-8)
    assert np.isclose(scores[1], np.exp(-5 * (11/14.56022)), atol = 1e-4)

    # Custom sigma scales the exponential decay.
    scores_sigma1 = distance_ratio_score(X, centers, sigma = 1)
    assert np.allclose(scores_sigma1, np.full(4, np.exp(-1 * 0.5)), atol = 1e-8)


####################################################################################################


def test_point_silhouette_score():
    # Two well-separated pairs of identical points: {0,1} and {2,3}.
    X = np.array([
        [1,1,0,0],
        [1,1,0,0],
        [0,0,1,1],
        [0,0,1,1]
    ], dtype = float)
    distances = pairwise_distances(X)

    assignment1 = np.array([
        [1,0],
        [1,0],
        [0,1],
        [0,1]
    ])
    # Each point's own cluster is at distance 0 from its twin and far from the other pair,
    # so every point is perfectly well-clustered.
    assert _point_silhouette_score(distances, assignment1, 0, 0) == 1
    assert _point_silhouette_score(distances, assignment1, 1, 0) == 1
    assert _point_silhouette_score(distances, assignment1, 2, 1) == 1
    assert _point_silhouette_score(distances, assignment1, 3, 1) == 1

    with pytest.raises(AssertionError):
        _point_silhouette_score(distances, assignment1, 0, 1)
    with pytest.raises(AssertionError):
        _point_silhouette_score(distances, assignment1, 2, 0)

    # Point 0 is now alone in cluster 0 (a singleton, intra defined as 0) while its twin,
    # point 1, was moved into cluster 1 -- 0's own "cluster" trivially beats any nonzero
    # inter-cluster distance, so it still reads as well-separated (score 1). Point 1, now
    # grouped with its true opposites {2,3}, is equidistant from those (2) and from its
    # true twin point 0 that got left in cluster 0 alone (0) -- but 0 is itself a distance
    # of 0 away, so intra (avg to 2,3) > inter (distance to lone point 0) -> badly placed (-1).
    assignment2 = np.array([
        [1,0],
        [0,1],
        [0,1],
        [0,1]
    ])
    assert _point_silhouette_score(distances, assignment2, 0, 0) == 1
    assert _point_silhouette_score(distances, assignment2, 1, 1) == -1


def test_silhouette_score():
    X = np.array([
        [1,1,0,0],
        [1,1,0,0],
        [0,0,1,1],
        [0,0,1,1]
    ], dtype = float)
    distances = pairwise_distances(X)

    assignment = np.array([
        [1,0],
        [1,0],
        [0,1],
        [0,1]
    ])
    # Perfectly separated pairs -> every point's silhouette is 1.
    assert np.isclose(silhouette_score(distances, assignment), 1.0)

    with pytest.raises(ValueError):
        silhouette_score(np.zeros((3,4)), assignment)

    with pytest.raises(ValueError):
        silhouette_score(distances, np.ones((3,2)))

    # Fewer than 2 non-empty clusters -> undefined, warns and returns nan.
    single_cluster_assignment = np.array([[1],[1],[1],[1]])
    with pytest.warns(UserWarning):
        assert np.isnan(silhouette_score(distances, single_cluster_assignment))


####################################################################################################


def test_mistakes():
    # 4 points, 2 clusters; ground truth: points 0,1 -> cluster 0, points 2,3 -> cluster 1.
    ground_truth_assignment = np.array([
        [1,0],
        [1,0],
        [0,1],
        [0,1]
    ])
    # 2 rules: rule 0 covers points 0,1,2 (assigned to cluster 0 -> point 2 is a mistake);
    # rule 1 covers point 3 only (assigned to cluster 1 -> no mistakes).
    data_to_rule_assignment = np.array([
        [1,0],
        [1,0],
        [1,0],
        [0,1]
    ])
    rule_to_cluster_assignment = np.array([
        [1,0],
        [0,1]
    ])
    assert mistakes(ground_truth_assignment, data_to_rule_assignment, rule_to_cluster_assignment) == 1

    # A rule not assigned to any cluster contributes no mistakes, regardless of coverage.
    rule_to_cluster_assignment_unassigned = np.array([
        [0,0],
        [0,1]
    ])
    assert mistakes(
        ground_truth_assignment, data_to_rule_assignment, rule_to_cluster_assignment_unassigned
    ) == 0

    with pytest.raises(AssertionError):
        mistakes(
            ground_truth_assignment,
            data_to_rule_assignment,
            np.array([[1,1],[0,1]]),  # a rule assigned to two clusters
        )


####################################################################################################


def test_clustering_distance():
    labels1 = [{0}, {0}, {1}, {1}]
    labels2 = [{0}, {1}, {1}, {1}]

    # Pairs (0,1): same in labels1, different in labels2 -> differs.
    # Pairs (0,2),(0,3): different in both -> agree.
    # Pairs (1,2),(1,3): different in labels1, same in labels2 -> differs.
    # Pair (2,3): same in both -> agree.
    assert clustering_distance(labels1, labels2) == 3
    assert clustering_distance(labels1, labels2, percentage = True) == 3 / (4*3/2)

    assert clustering_distance(labels1, labels1) == 0
    assert clustering_distance([], []) is np.nan or np.isnan(clustering_distance([], []))
    assert clustering_distance([{0}], [{1}]) == 0.0

    with pytest.raises(ValueError):
        clustering_distance(labels1, [{0}])


####################################################################################################


def test_rule_pairwise_difference():
    # All points sharing one rule; if they all had the same true label there'd be 0
    # cross-cluster pairs. Here 3 points from cluster 0, 2 from cluster 1: mismatched
    # pairs = 3*2 = 6.
    labels = [{0}, {0}, {0}, {1}, {1}]
    assert rule_pairwise_difference(labels) == 6
    n_pairs = 5*4/2
    assert rule_pairwise_difference(labels, percentage = True) == 6 / n_pairs

    with pytest.raises(ValueError):
        rule_pairwise_difference([{0}, {0,1}])
