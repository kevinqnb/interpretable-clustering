import numpy as np
from itertools import combinations, permutations
from intercluster.measurements import entropy
cimport cython
cimport numpy as cnp

# Typing
from numpy.typing import NDArray
cnp.import_array()
from typing import Tuple, List, Set

DTYPE_float = np.float64
ctypedef cnp.float64_t DTYPE_float_t
DTYPE_int = np.int64
ctypedef cnp.int64_t DTYPE_int_t
    

####################################################################################################


cpdef float cost_cy(
    cnp.ndarray[DTYPE_int_t, ndim = 1] indices,
    cnp.ndarray[DTYPE_int_t, ndim = 1] labels,
):
    """
    Given a set of points X, computes the score as the entropy of the labels.
    
    Args:                
        indices (NDArray, optional): Indices of points to compute score with.
            
    Returns:
        (float): Score of the given data.
    """
    cdef cnp.ndarray[DTYPE_int_t, ndim = 1] sublabels
    cdef cnp.ndarray[DTYPE_int_t, ndim = 1] unique_labels, counts
    cdef float entropy_score
    if len(indices) == 0:
        return 0
    else:
        sublabels = labels[indices]
        return entropy(sublabels)

    
#####################################################################################################


cpdef float gain_cy(
    int n,
    cnp.ndarray[DTYPE_int_t, ndim = 1] y,
    cnp.ndarray[DTYPE_int_t, ndim = 1] left_indices,
    cnp.ndarray[DTYPE_int_t, ndim = 1] right_indices,
    float parent_cost
):
    """
    Computes the gain associated with a split.
    
    Args:
        left_indices (np.ndarray): Indices for the left child of the split.
        
        right_indices (np.ndarray): Indices for the right child of the split.
        
        parent_cost (float, optional): The cost of the parent node. Defaults to None.
    
    Returns:
        gain (float): The gain associated with the split.
    """
    cdef cnp.ndarray[DTYPE_int_t, ndim = 1] parent_indices
    cdef float left_cost, right_cost
    cdef float gain_val
    cdef int total_indices = len(left_indices) + len(right_indices)
    
    # This calculates the relative reduction in impurity, based on the 
    # definitions used by SKLearn's decision tree model.
    left_cost = len(left_indices)/total_indices * cost_cy(left_indices, y)
    right_cost = len(right_indices)/total_indices * cost_cy(right_indices, y)
    gain_val = total_indices/n * (parent_cost - (left_cost + right_cost))
    return gain_val


####################################################################################################


def split_cy(
    cnp.ndarray[DTYPE_float_t, ndim = 2] X,
    cnp.ndarray[DTYPE_int_t, ndim = 1] y,
    cnp.ndarray[DTYPE_int_t, ndim = 1] indices,
    int min_points_leaf,
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
    cdef float parent_cost = cost_cy(indices, y)
    
    cdef int i,j
    cdef cnp.ndarray[DTYPE_float_t, ndim = 1] unique_values
    cdef float threshold, gain_val, best_gain_val
    cdef cnp.ndarray[DTYPE_int_t, ndim = 1] evals, left_indices, right_indices
    cdef object condition, best_condition
    cdef list best_conditions

    best_gain_val = -np.inf
    best_conditions = []
    for i in range(d):
        unique_values = np.unique(X_[:, i])
        for j in range(unique_values.shape[0]):

            # Split condition:
            threshold = unique_values[j]
            evals = np.array(X_[:, i] <= threshold, dtype = DTYPE_int)
            left_indices = indices[np.where(evals)[0]]
            right_indices = indices[np.where(1 - evals)[0]]
            
            if (len(left_indices) < min_points_leaf or 
                len(right_indices) < min_points_leaf):
                gain_val = -np.inf
            else:
                gain_val = gain_cy(X.shape[0], y, left_indices, right_indices, parent_cost)
            
            if gain_val > best_gain_val:
                best_gain_val = gain_val
                best_conditions = [(i, threshold)]
            
            elif gain_val == best_gain_val:
                best_conditions.append((i, threshold))
    
    # Randomly break ties if necessary:
    best_condition = best_conditions[np.random.randint(len(best_conditions))]
    return best_gain_val, best_condition
    

####################################################################################################

# NOTE: I have no idea if this works!!

def oblique_split_cy(
    cnp.ndarray[DTYPE_float_t, ndim = 2] X,
    cnp.ndarray[DTYPE_int_t, ndim = 1] y,
    cnp.ndarray[DTYPE_int_t, ndim = 1] indices,
    int min_points_leaf,
    #object get_split_indices_fn,
    #object cost_fn, 
    #object gain_fn,
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
    cdef float parent_cost = cost_cy(indices, y)
    
    cdef int feature
    cdef float threshold, gain_val, best_gain_val
    #cdef object condition, best_condition
    cdef cnp.ndarray[DTYPE_int_t, ndim = 1] evals, left_indices, right_indices
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
                    '''
                    condition = LinearCondition(
                        features = np.array([pair[1]]),
                        weights = np.array([1]),
                        threshold = data_point[1],
                        direction = -1
                    )
                    '''
                    threshold = data_point[1]
                    evals = np.array(X_[:, pair[1]] <= threshold, dtype = DTYPE_int)


                elif slope == np.inf:
                    # axis aligned case (vertical)
                    '''
                    condition = LinearCondition(
                        features = np.array([pair[0]]),
                        weights = np.array([1]),
                        threshold = data_point[0],
                        direction = -1
                    )
                    '''
                    threshold = data_point[0]
                    evals = np.array(X_[:, pair[0]] <= threshold, dtype = DTYPE_int)
                else:
                    # non-axis aligned slopes
                    threshold = -slope * data_point[0] + data_point[1]
                    '''
                    condition = LinearCondition(
                        features = np.array(pair),
                        weights = np.array([-slope, 1]),
                        threshold = threshold,
                        direction = -1
                    )
                    '''
                    evals = np.array(
                        np.dot(X_[:, pair], np.array([-slope, 1])) <= threshold,
                        dtype = DTYPE_int
                    )

                #left_indices, right_indices = get_split_indices_fn(indices, condition)
                left_indices = indices[np.where(evals)[0]]
                right_indices = indices[np.where(1 - evals)[0]]

                if (len(left_indices) < min_points_leaf or 
                    len(right_indices) < min_points_leaf):
                    gain_val = -np.inf
                else:
                    gain_val = gain_cy(X.shape[0], y, left_indices, right_indices, parent_cost)
                
                if gain_val > best_gain_val:
                    best_gain_val = gain_val
                    best_conditions = [(pair, slope, threshold)]
                
                elif gain_val == best_gain_val:
                    best_conditions.append((pair, slope, threshold))
    
    # Randomly break ties if necessary:
    best_condition = best_conditions[np.random.randint(len(best_conditions))]
    return best_gain_val, best_condition


####################################################################################################