cimport cython
import numpy as np
cimport numpy as cnp
from intercluster.utils import tiebreak
from .._conditions import Condition, LinearCondition

# Typing
from typing import Tuple, List, Set, Callable, Union
from numpy.typing import NDArray
cnp.import_array()

DTYPE_float = np.float64
ctypedef cnp.float64_t DTYPE_float_t
DTYPE_int = np.int64
ctypedef cnp.int64_t DTYPE_int_t


####################################################################################################


def get_split_outliers_cy(
        cnp.ndarray[DTYPE_int_t, ndim = 1] cluster_labels,
        cnp.ndarray[DTYPE_int_t, ndim = 1] left_indices,
        cnp.ndarray[DTYPE_int_t, ndim = 1] right_indices
    ) -> NDArray:
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