cimport cython
import numpy as np
cimport numpy as cnp
cnp.import_array()

# Typing
from numpy.typing import NDArray
DTYPE = np.float64
ctypedef cnp.float64_t DTYPE_t
DTYPE_int = np.int64
ctypedef cnp.int64_t DTYPE_int_t


def oned_cluster_cy(
    cnp.ndarray[DTYPE_t, ndim = 1] x,
    float cluster_cost = 0.0,
    str method = "kmeans",
) -> cnp.ndarray[int]:
    """
    Clusters a one-dimensional array using the specified method.

    Args:
        x (np.ndarray): 1D array of data points to be clustered.
        cluster_cost (float): Cost associated with creating a new cluster.
            Must be a value between 0 and 1. 
        method (str): Clustering method to use. Supported options are "kmeans" or "kmedians".
    Returns:
        labels (np.ndarray): Array of cluster boundary indices.
    """
    if len(x) == 0:
        raise ValueError("Input data cannot be empty.")

    cdef int n = x.shape[0]
    cdef cnp.ndarray[DTYPE_int_t, ndim=1] sorting_indices = np.argsort(x)
    cdef cnp.ndarray[DTYPE_t, ndim=1] x_sorted = x[sorting_indices]

    if cluster_cost < 0 or cluster_cost > 1:
        raise ValueError("Cluster cost must be between 0 and 1.")

    if method not in ["kmeans", "kmedians"]:
        raise ValueError(f"Unsupported clustering method: {method}")

    if method == "kmeans":
        normalization = np.sum((x - np.mean(x))**2)
    else:
        normalization = np.sum(np.abs(x - np.median(x)))

    cdef cnp.ndarray[DTYPE_t, ndim=1] memo_table = np.full(n, np.nan, dtype=DTYPE)
    cdef int i, j
    cdef float error, objective, min_objective

    for i in range(n):
        min_objective = np.inf
        for j in range(i + 1):
            if method == "kmeans":
                error = np.sum((x_sorted[j:i+1] - np.mean(x_sorted[j:i+1]))**2) / normalization
            else:
                error = np.sum(np.abs(x_sorted[j:i+1] - np.median(x_sorted[j:i+1]))) / normalization

            if j == 0: # Base Case
                objective = error + cluster_cost
            else:
                objective = error + cluster_cost + memo_table[j - 1]

            if objective < min_objective:
                min_objective = objective

        memo_table[i] = min_objective


    # Backtrack:
    cdef list boundaries = [n]
    cdef DTYPE_t current_error
    cdef int current_idx = n - 1
    cdef float current_objective
    j = current_idx
    while j >= 0:
        current_objective = memo_table[current_idx]

        if method == "kmeans":
            error = np.sum(
                (x_sorted[j:current_idx + 1] - np.mean(x_sorted[j:current_idx + 1]))**2
            ) / normalization
        else:
            error = np.sum(
                np.abs(x_sorted[j:current_idx + 1] - np.median(x_sorted[j:current_idx + 1]))
            ) / normalization

        if j == 0:
            objective = error + cluster_cost
        else:
            objective = error + cluster_cost + memo_table[j - 1]

        if np.isclose(objective, current_objective):
            boundaries.append(j)
            current_idx = j - 1

        j -= 1

    # Translate boundaries to labels
    boundaries = boundaries[::-1]
    cdef cnp.ndarray[DTYPE_int_t, ndim=1] labels = np.zeros(n, dtype=DTYPE_int)
    cdef int cluster_label = 0
    for i in range(1, len(boundaries)):
        start = boundaries[i - 1]
        end = boundaries[i]
        for j in range(start, end):
            labels[sorting_indices[j]] = cluster_label
        cluster_label += 1

    return labels

