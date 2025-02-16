import numpy as np
from intercluster.rules import LinearCondition


def test_axis_aligned():
    cond = LinearCondition(
        features = np.array([1]),
        weights = np.array([1]),
        threshold = 2,
        direction = -1
    )
    
    X = np.array([
        [0, 3],
        [np.inf, 0],
        [1,1],
        [3,4]
    ])
    
    assert np.array_equal(cond.evaluate(X), np.array([False, True, True, False]))
    
def test_linear(): 
    cond = LinearCondition(
        features = np.array([0,1]),
        weights = np.array([1/2, 3]),
        threshold = 5,
        direction = 1
    )
    
    X = np.array([
        [0, 3],
        [np.inf, 0],
        [1,1],
        [3,4]
    ])
    
    assert np.array_equal(cond.evaluate(X), np.array([True, True, False, True]))
    
    
def test_with_equality():
    cond = LinearCondition(
        features = np.array([1]),
        weights = np.array([1]),
        threshold = 2,
        direction = -1
    )
    
    X = np.array([
        [0, 3],
        [np.inf, 2],
        [1,2],
        [3,4]
    ])
    
    assert np.array_equal(cond.evaluate(X), np.array([False, True, True, False]))
    
    
def test_one_point():
    cond = LinearCondition(
        features = np.array([4]),
        weights = np.array([1]),
        threshold = 2,
        direction = -1
    )
    
    X = np.array([[0,3,1000, 5, 1, 35]])
    assert np.array_equal(cond.evaluate(X), np.array([True]))
    