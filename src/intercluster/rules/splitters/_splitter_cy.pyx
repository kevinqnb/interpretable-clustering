import numpy as np
cimport cython
cimport numpy as cnp
from itertools import combinations, permutations
from intercluster.utils import can_flatten, flatten_labels
from .._conditions import Condition, LinearCondition

# Typing
from numpy.typing import NDArray
cnp.import_array()
from typing import Tuple, List, Set

DTYPE_float = np.float64
ctypedef cnp.float64_t DTYPE_float_t
DTYPE_int = np.int64
ctypedef cnp.int64_t DTYPE_int_t
    

####################################################################################################


def split_cy(
    cnp.ndarray[DTYPE_float_t, ndim = 2] X,
    cnp.ndarray[DTYPE_int_t, ndim = 1] indices,
    int min_points_leaf,
    object get_split_indices_fn,
    object cost_fn, 
    object gain_fn,
) -> Tuple[float, Condition]:
    """
    Computes the best split of a leaf node.
    
    Args:
        indices (np.ndarray, optional): Indices for a subset of the original dataset.
    
    Returns:
        gain (float): The gain associated with the split.
        
        condition (Condition): Logical or functional condition for evaluating and 
            splitting the data points.
    """
    cdef cnp.ndarray[DTYPE_float_t, ndim = 2] X_ = X[indices, :]
    cdef int n = X_.shape[0]
    cdef int d = X_.shape[1]
    cdef float parent_cost = cost_fn(indices)
    
    cdef int feature
    cdef float threshold, gain_val, best_gain_val
    cdef object condition, best_condition
    cdef cnp.ndarray[DTYPE_int_t, ndim = 1] left_indices, right_indices
    cdef list best_conditions

    best_gain_val = -np.inf
    best_conditions = []
    for feature in range(d):
        for threshold in np.unique(X_[:,feature]):
            condition = LinearCondition(
                features = np.array([feature]),
                weights = np.array([1]),
                threshold = threshold,
                direction = -1
            )
            left_indices, right_indices = get_split_indices_fn(indices, condition)
            
            if (len(left_indices) < min_points_leaf or 
                len(right_indices) < min_points_leaf):
                gain_val = -np.inf
            else:
                gain_val = gain_fn(left_indices, right_indices, parent_cost)
            
            if gain_val > best_gain_val:
                best_gain_val = gain_val
                best_conditions = [condition]
            
            elif gain_val == best_gain_val:
                best_conditions.append(condition)
    
    # Randomly break ties if necessary:
    best_condition = best_conditions[np.random.randint(len(best_conditions))]
    return best_gain_val, best_condition
    

####################################################################################################


def oblique_split_cy(
    cnp.ndarray[DTYPE_float_t, ndim = 2] X,
    cnp.ndarray[DTYPE_int_t, ndim = 1] indices,
    int min_points_leaf,
    object get_split_indices_fn,
    object cost_fn, 
    object gain_fn,
) -> Tuple[float, Condition]:
    """
    Computes the best split of a leaf node.
    
    Args:
        indices (np.ndarray, optional): Indices for a subset of the original dataset.
    
    Returns:
        gain (float): The gain associated with the split.
        
        condition (Condition): Logical or functional condition for evaluating and 
            splitting the data points.
    """
    cdef cnp.ndarray[DTYPE_float_t, ndim = 2] X_ = X[indices, :]
    cdef int n = X_.shape[0]
    cdef int d = X_.shape[1]
    cdef float parent_cost = cost_fn(indices)
    
    cdef int feature
    cdef float threshold, gain_val, best_gain_val
    cdef object condition, best_condition
    cdef cnp.ndarray[DTYPE_int_t, ndim = 1] left_indices, right_indices
    cdef list best_conditions
    
    # Search only through pairs of features, and select slopes within that 
    # low dimensional space. 
    cdef cnp.ndarray[DTYPE_int_t, ndim = 1] pair
    cdef cnp.ndarray[DTYPE_int_t, ndim = 2] feature_pairs = np.array(
        list(combinations(range(d), 2))
    )
    cdef cnp.ndarray[DTYPE_float_t, ndim = 1] slopes = np.array(
        [0, 1/2, 1, 2, np.inf, -2, -1, -1/2], dtype = DTYPE_float
    )

    best_gain_val = -np.inf
    best_conditions = []
    for pair in feature_pairs:
        X_pair = X_[:, pair]
        for i in range(X_pair.shape[0]):
            data_point = X_pair[i,:]
            for slope in slopes:
                if slope == 0:
                    # axis aligned case (horizontal)
                    condition = LinearCondition(
                        features = np.array([pair[1]]),
                        weights = np.array([1]),
                        threshold = data_point[1],
                        direction = -1
                    )
                elif slope == np.inf:
                    # axis aligned case (vertical)
                    condition = LinearCondition(
                        features = np.array([pair[0]]),
                        weights = np.array([1]),
                        threshold = data_point[0],
                        direction = -1
                    )
                else:
                    # non-axis aligned slopes
                    threshold = -slope * data_point[0] + data_point[1]
                    condition = LinearCondition(
                        features = np.array(pair),
                        weights = np.array([-slope, 1]),
                        threshold = threshold,
                        direction = -1
                    )

                left_indices, right_indices = get_split_indices_fn(indices, condition)
                
                if (len(left_indices) < min_points_leaf or 
                    len(right_indices) < min_points_leaf):
                    gain_val = -np.inf
                else:
                    gain_val = gain_fn(left_indices, right_indices, parent_cost)
                
                if gain_val > best_gain_val:
                    best_gain_val = gain_val
                    best_conditions = [condition]
                
                elif gain_val == best_gain_val:
                    best_conditions.append(condition)
    
    # Randomly break ties if necessary:
    best_condition = best_conditions[np.random.randint(len(best_conditions))]
    return best_gain_val, best_condition


####################################################################################################