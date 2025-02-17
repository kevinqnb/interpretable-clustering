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
    
    

def test_mode():
    arr = np.array([1,2,2,3,3,4,4,4,5,7])
    assert mode(arr) == 4
    
    samples = 1000
    arr = np.array([1,2,2,3,3,4,4,4,5,3])
    sampled_modes = np.zeros(samples)
    for i in range(samples):
        sampled_modes[i] = mode(arr)
        
    _,counts = np.unique(sampled_modes, return_counts = True)
    assert np.isclose(counts[0]/samples, 0.5, atol = 0.05, rtol = 0)
    assert np.isclose(counts[1]/samples, 0.5, atol = 0.05, rtol = 0)  
    


def test_entropy():
    assert np.isclose(entropy(np.array([1, 1, 1, 1, 1, 1])), 0, atol = 1e-5)
    assert np.isclose(entropy(np.array([1, 0, 1, 0, 1, 0])), 1, atol = 1e-5)
    
    uneven = np.array([2,2,2,3,3,3,2,2,2,2])
    assert np.isclose(entropy(uneven), 0.8816, atol = 0.05, rtol = 0)
    
    multiclass = np.array([1,1,2,2,3,3,4,4,5,5])
    assert np.isclose(entropy(multiclass),  -np.log2(1/5), atol = 0.05, rtol = 0)
    
    
    
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
    
    

def test_update_centers():
    X = np.array([
        [1,2],
        [3,4],
        [5,6],
        [7,8],
        [9,10],
        [11,12]
    ])
    
    assignment = np.array([
        [1,0,0],
        [1,0,0],
        [0,1,0],
        [0,1,0],
        [0,0,1],
        [0,0,1]
    ])
    
    centers = update_centers(X, assignment)
    test_centers = np.array([
        [2,3],
        [6,7],
        [10,11]
    ])
    assert np.array_equal(centers, test_centers)
    
    
    
def test_label_formatting():
    labels = np.array([1,2,1,1,3,3,3])
    test_labels = [{1}, {2}, {1}, {1}, {3}, {3}, {3}]
    assert labels_format(labels) == test_labels
    assert np.array_equal(labels, flatten_labels(test_labels))
    
    test_labels2 = [{1,3}, {2}, {1}, set(), {3}, {1,3,4}, {3}]
    assert np.array_equal(flatten_labels(test_labels2), np.array([1,3,2,1,3,1,3,4,3]))
    
    assert unique_labels(test_labels2) == {1,2,3,4}
        
        
def test_can_flatten():
    labels = [{1}, {2}, {1}, {1}, {3}, {3}, {3}]
    assert can_flatten(labels) == True
    
    labels = [{1}, {2}, {1}, {1,2}, {3}, {3}, {3}]
    assert can_flatten(labels) == False
    
    labels = [{1}, {2}, {1}, {1,2}, {3}, {3}, set()]
    assert can_flatten(labels) == False
    
    labels = [{1}, {2}, set(), {1,2}, {3}, {3}, {3}]
    assert can_flatten(labels) == False

    
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