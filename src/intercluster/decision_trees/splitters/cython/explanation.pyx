cimport cython
import numpy as np
cimport numpy as cnp
from intercluster.utils import tiebreak

# Typing
from typing import Tuple, List, Set, Callable, Union
from numpy.typing import NDArray
cnp.import_array()

DTYPE_float = np.float64
ctypedef cnp.float64_t DTYPE_float_t
DTYPE_int = np.int64
ctypedef cnp.int64_t DTYPE_int_t


####################################################################################################


cpdef cnp.ndarray[DTYPE_int_t, ndim=1] get_split_outliers_cy(
        cnp.ndarray[DTYPE_int_t, ndim = 1] cluster_labels,
        cnp.ndarray[DTYPE_int_t, ndim = 1] left_indices,
        cnp.ndarray[DTYPE_int_t, ndim = 1] right_indices
):
    """
    Finds outliers to be removed from set of indices.

    Args:
        cluster_labels (np.ndarray): Integer labels describing cluster membership for each 
            data point.
        left_indices (np.ndarray): Indices for the left child of the split.
        
        right_indices (np.ndarray): Indices for the right child of the split.

    Returns:
        outliers (np.ndarray): Indices of the data points to be removed as outliers.
    """
    cdef int n = len(cluster_labels)
    cdef cnp.ndarray[DTYPE_int_t, ndim = 1] unique_clusters = np.unique(
        cluster_labels[np.concatenate((left_indices, right_indices))]
    )
    cdef int num_clusters = len(unique_clusters)
    if num_clusters < 2:
        raise ValueError("The clustering must contain at least 2 non-empty clusters.")
    cdef cluster_idx_dict = {clust:i for i,clust in enumerate(unique_clusters)}

    cdef cnp.ndarray[DTYPE_int_t, ndim = 1] outliers = np.zeros(
        n, dtype = DTYPE_int
    )
    cdef cnp.ndarray[DTYPE_int_t, ndim = 2] left_half_satisfies = np.zeros(
        (num_clusters, n), dtype = DTYPE_int
    )
    cdef cnp.ndarray[DTYPE_int_t, ndim = 2] right_half_satisfies = np.zeros(
        (num_clusters, n), dtype = DTYPE_int
    )
    cdef cnp.ndarray[DTYPE_int_t, ndim = 1] left_half_counts = np.zeros(
        num_clusters, dtype = DTYPE_int
    )
    cdef cnp.ndarray[DTYPE_int_t, ndim = 1] right_half_counts = np.zeros(
        num_clusters, dtype = DTYPE_int
    )
    cdef int i, idx, cluster, minimum_majority_cluster, left_assigned, right_assigned, left_assign
    cdef cnp.ndarray[DTYPE_int_t, ndim = 1] non_empty, left_positive, right_positive, tied
    cdef cnp.ndarray[DTYPE_int_t, ndim = 1] tied_left_assignment

    for i,idx in enumerate(left_indices):
        cluster = cluster_labels[idx]
        left_half_satisfies[cluster_idx_dict[cluster], idx] = 1
        left_half_counts[cluster_idx_dict[cluster]] += 1

    for i,idx in enumerate(right_indices):
        cluster = cluster_labels[idx]
        cluster = cluster_labels[idx]
        right_half_satisfies[cluster_idx_dict[cluster], idx] = 1
        right_half_counts[cluster_idx_dict[cluster]] += 1

    # Case 1. All clusters have majority in the left indices:
    if np.all(left_half_counts > right_half_counts):
        non_empty = np.where(right_half_counts > 0)[0]
        minimum_majority_cluster = non_empty[np.argmin(left_half_counts[non_empty])]
        outliers = np.sum(
            (
                outliers + 
                left_half_satisfies[minimum_majority_cluster,:] +
                right_half_satisfies[~np.isin(range(num_clusters), minimum_majority_cluster), :]
            ),
            axis = 0,
            dtype = DTYPE_int
        )

    # Case 2. All clusters have majority in the right indices:
    elif np.all(right_half_counts > left_half_counts):
        non_empty = np.where(left_half_counts > 0)[0]
        minimum_majority_cluster = non_empty[np.argmin(right_half_counts[non_empty])]
        outliers = np.sum(
            (
                outliers + 
                right_half_satisfies[minimum_majority_cluster,:] +
                left_half_satisfies[~np.isin(range(num_clusters), minimum_majority_cluster), :]
            ),
            axis = 0,
            dtype = DTYPE_int
        )

    # Case 3. Cluster membership is mixed:
    else:
        left_positive = np.where(left_half_counts > right_half_counts)[0]
        right_positive = np.where(right_half_counts > left_half_counts)[0]
        tied = np.where(left_half_counts == right_half_counts)[0]
        
        if len(left_positive) > 0:
            outliers = np.sum(
                outliers + right_half_satisfies[left_positive, :],
                axis = 0,
                dtype = DTYPE_int
            )

        if len(right_positive) > 0:
            outliers = np.sum(
                outliers + left_half_satisfies[right_positive, :],
                axis = 0,
                dtype = DTYPE_int
            )

        if len(tied) > 0:
            left_assigned = 0
            right_assigned = 0
            tied_left_assignment = np.zeros(len(tied), dtype = DTYPE_int)

            # Break ties until both halves contain at least one cluster.
            while left_assigned == 0 or right_assigned == 0:
                left_assigned = len(left_positive)
                right_assigned = len(right_positive)

                for i,idx in enumerate(tied):
                    # Flip a coin and assign
                    if np.random.uniform() <= 0.5:
                        left_assigned += 1
                        tied_left_assignment[i] = 1
                    else:
                        right_assigned += 1

            # Remove outliers
            for i,left_assign in enumerate(tied_left_assignment):
                if left_assign > 0:
                    outliers = outliers + right_half_satisfies[tied[i],:]
                else:
                    outliers = outliers + left_half_satisfies[tied[i],:]
        
    return np.where(outliers)[0]


####################################################################################################


cpdef double gain_cy(
    cnp.ndarray[DTYPE_int_t, ndim = 1] y,
    cnp.ndarray[DTYPE_int_t, ndim = 1] left_indices,
    cnp.ndarray[DTYPE_int_t, ndim = 1] right_indices
):
    """
    Computes the gain associated with a split.
    
    Args:
        y (np.ndarray): Integer labels describing cluster membership for each data point.

        left_indices (np.ndarray): Indices for the left child of the split.
        
        right_indices (np.ndarray): Indices for the right child of the split.
    Returns:
        gain (double): The gain associated with the split.
    """    
    # If only a single cluster is present, no gain to be had.
    cdef cnp.ndarray[DTYPE_int_t, ndim = 1] parent_indices = np.concatenate(
        (left_indices, right_indices)
    )
    cdef cnp.ndarray[DTYPE_int_t, ndim = 1] clusters_present = y[parent_indices]
    if np.all(clusters_present == clusters_present[0]):
        return -np.inf

    # Setting this to 0 simply minimizes the number of outliers removed without considering 
    # what happened in the parent node.
    cdef double parent_cost = 0
    cdef cnp.ndarray[DTYPE_int_t, ndim = 1] split_outliers = get_split_outliers_cy(
        y, left_indices, right_indices
    )
    cdef double split_cost = len(split_outliers)
    return parent_cost - (split_cost)


####################################################################################################


def split_cy(
    cnp.ndarray[DTYPE_float_t, ndim = 2] X,
    cnp.ndarray[DTYPE_int_t, ndim = 1] y,
    cnp.ndarray[DTYPE_int_t, ndim = 1] indices,
    int min_points_leaf,
) -> Tuple[double, Condition]:
    """
    Computes the best split of a leaf node.
    
    Args:
        indices (np.ndarray, optional): Indices for a subset of the original dataset.
    
    Returns:
        gain (double): The gain associated with the split.
        
        condition (Condition): Logical or functional condition for evaluating and 
            splitting the data points.
    """
    cdef cnp.ndarray[DTYPE_float_t, ndim = 2] X_ = X[indices, :]
    cdef int n = X_.shape[0]
    cdef int d = X_.shape[1]
    
    cdef int i,j
    cdef cnp.ndarray[DTYPE_float_t, ndim = 1] unique_values
    cdef double threshold, gain_val, best_gain_val
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
            print(threshold)
            evals = np.array(X_[:, i] <= threshold, dtype = DTYPE_int)
            left_indices = indices[np.where(evals)[0]]
            right_indices = indices[np.where(1 - evals)[0]]
            
            if (len(left_indices) < min_points_leaf or 
                len(right_indices) < min_points_leaf):
                gain_val = -np.inf
            else:
                gain_val = gain_cy(y, left_indices, right_indices)
            
            if gain_val > best_gain_val:
                best_gain_val = gain_val
                best_conditions = [(i, threshold)]
            
            elif gain_val == best_gain_val:
                best_conditions.append((i, threshold))
    
    # Randomly break ties if necessary:
    best_condition = best_conditions[np.random.randint(len(best_conditions))]
    return best_gain_val, best_condition