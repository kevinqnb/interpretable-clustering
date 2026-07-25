import pytest
import numpy as np
from intercluster.utils import *


####################################################################################################


def test_tiebreak():
    scores = np.array([1, 1, 1, 1, 1])
    proxy = np.array([1, 2, 3, 4, 5])
    assert np.array_equal(tiebreak(scores, proxy), [0, 1, 2, 3, 4])

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


####################################################################################################


def test_divide_with_zeros():
    x = np.array([0., 1., 3., 0., 4., 5.])
    y = np.array([1., 0., 3., 0., 2., 4.])
    assert np.array_equal(divide_with_zeros(x,y), np.array([0, np.inf, 1, 1, 2, 5/4]))

    X = np.array([[0., 1., 3.], [0., 4., 5.]])
    Y = np.array([[1., 0., 3.], [0., 2., 4.]])
    assert np.array_equal(divide_with_zeros(X,Y), np.array([[0, np.inf, 1], [1, 2, 5/4]]))


####################################################################################################


def test_update_centers():
    X = np.array([
        [1,2],
        [3,4],
        [5,6],
        [7,8],
        [9,10],
        [11,12]
    ])
    current_centers = np.zeros((3,2))

    assignment = np.array([
        [1,0,0],
        [1,0,0],
        [0,1,0],
        [0,1,0],
        [0,0,1],
        [0,0,1]
    ])

    centers = update_centers(X, current_centers, assignment)
    test_centers = np.array([
        [2,3],
        [6,7],
        [10,11]
    ])
    assert np.array_equal(centers, test_centers)


####################################################################################################


def test_label_formatting():
    labels = np.array([1,2,1,1,3,3,3])
    test_labels = [{1}, {2}, {1}, {1}, {3}, {3}, {3}]
    assert labels_format(labels) == test_labels
    assert np.array_equal(labels, flatten_labels(test_labels))

    test_labels2 = [{1,3}, {2}, {1}, set(), {3}, {1,3,4}, {3}]
    assert np.array_equal(flatten_labels(test_labels2), np.array([1,3,2,1,-1,3,1,3,4,3]))

    assert unique_labels(test_labels2) == {1,2,3,4}


####################################################################################################


def test_can_flatten():
    labels = [{1}, {2}, {1}, {1}, {3}, {3}, {3}]
    assert can_flatten(labels) == True

    labels = [{1}, {2}, {1}, {1,2}, {3}, {3}, {3}]
    assert can_flatten(labels) == False

    labels = [{1}, {2}, {1}, {1,2}, {3}, {3}, set()]
    assert can_flatten(labels) == False

    labels = [{1}, {2}, set(), {1,2}, {3}, {3}, {3}]
    assert can_flatten(labels) == False


####################################################################################################


def test_labels_to_assignment():
    labels = [{1,3}, {2}, {1}, set(), {3}, {1,3,4}, {3}]

    test_assignment = np.array([
        [0,1,0,1,0],
        [0,0,1,0,0],
        [0,1,0,0,0],
        [0,0,0,0,0],
        [0,0,0,1,0],
        [0,1,0,1,1],
        [0,0,0,1,0]
    ])

    assert np.array_equal(labels_to_assignment(labels, n_labels = 5), test_assignment)


####################################################################################################


def test_assignment_to_labels():
    labels = [{1,3}, {2}, {1}, set(), {3}, {1,3,4}, {3}]

    test_assignment = np.array([
        [0,1,0,1,0],
        [0,0,1,0,0],
        [0,1,0,0,0],
        [0,0,0,0,0],
        [0,0,0,1,0],
        [0,1,0,1,1],
        [0,0,0,1,0]
    ])

    assert assignment_to_labels(test_assignment) == labels


####################################################################################################


def test_assignment_to_dict():
    test_assignment = np.array([
        [0,1,0,1,0],
        [0,0,1,0,0],
        [0,1,0,0,0],
        [0,0,0,0,0],
        [0,0,0,1,0],
        [0,1,0,1,1],
        [0,0,0,1,0]
    ])

    assignment_dict = assignment_to_dict(test_assignment)
    for i in range(5):
        assert assignment_dict[i] == set(np.where(test_assignment[:,i])[0])
