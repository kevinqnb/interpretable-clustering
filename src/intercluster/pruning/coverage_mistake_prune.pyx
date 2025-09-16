import numpy as np
cimport cython
cimport numpy as cnp
from libc.stdlib cimport malloc, free
from libc.string cimport memset

# Typing
cnp.import_array()
from numpy.typing import NDArray

DTYPE_bool = np.bool_
ctypedef cnp.uint8_t DTYPE_bool_t
DTYPE_int = np.int64
ctypedef cnp.int64_t DTYPE_int_t
DTYPE_float = np.float64
ctypedef cnp.float64_t DTYPE_float_t


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int count_union(
    cnp.ndarray[DTYPE_bool_t, ndim=1] arr1,
    cnp.ndarray[DTYPE_bool_t, ndim=1] arr2,
    int n
):
    """Count the number of True values in the union of two boolean arrays."""
    cdef int count = 0
    cdef int i
    for i in range(n):
        if arr1[i] or arr2[i]:
            count += 1
    return count


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int count_intersection(
    cnp.ndarray[DTYPE_bool_t, ndim=1] arr1,
    cnp.ndarray[DTYPE_bool_t, ndim=1] arr2,
    int n
):
    """Count the number of True values in the intersection of two boolean arrays."""
    cdef int count = 0
    cdef int i
    for i in range(n):
        if arr1[i] and arr2[i]:
            count += 1
    return count


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void update_coverage(
    cnp.ndarray[DTYPE_bool_t, ndim=1] covered,
    cnp.ndarray[DTYPE_bool_t, ndim=1] new_coverage,
    int n
):
    """Update the covered array with the union of covered and new_coverage."""
    cdef int i
    for i in range(n):
        if new_coverage[i]:
            covered[i] = 1


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cnp.ndarray[DTYPE_int_t, ndim=1] coverage_mistake_prune_cy(
    cnp.ndarray[DTYPE_bool_t, ndim=2] data_to_cluster_assignment,
    cnp.ndarray[DTYPE_bool_t, ndim=2] rule_to_cluster_assignment,
    cnp.ndarray[DTYPE_bool_t, ndim=2] data_to_rules_assignment,
    int n_rules,
    double lambda_val
):
    """
    Optimized Cython implementation of the Coverage Mistake Pruning algorithm.
    This makes use of a distorted greedy approach to select rules.
    
    For more information, see the following paper:
    "Submodular Maximization Beyond Non-Negativity: Guarantees, Fast Algorithms, and Applications"
    by Harshaw el al., ICML 2019.
    
    Args:
        data_to_cluster_assignment: Size (n x k) boolean array where entry (i,j) is 
            True if point i is assigned to cluster j.
        rule_to_cluster_assignment: Size (r x k) boolean array where entry (i,j) is 
            True if rule i is assigned to cluster j.
        data_to_rules_assignment: Size (n x r) boolean array where entry (i,j) is 
            True if data point i is assigned to rule j.
        n_rules: Maximum number of rules to select.
        lambda_val: Hyperparameter controlling tradeoff between coverage and overlap.
    
    Returns:
        Array of integers representing the indices of the selected rules.
    """
    cdef int n = data_to_cluster_assignment.shape[0]  # number of data points
    cdef int k = data_to_cluster_assignment.shape[1]  # number of clusters
    cdef int r = rule_to_cluster_assignment.shape[0]  # number of rules
    
    # Precompute rule labels (which cluster each rule belongs to)
    cdef cnp.ndarray[DTYPE_int_t, ndim=1] rule_labels = np.zeros(r, dtype=DTYPE_int)
    cdef int i, j, rule, cluster
    for i in range(r):
        for j in range(k):
            if rule_to_cluster_assignment[i, j]:
                rule_labels[i] = j
                break
    
    # Precompute which data points each cluster should cover
    cdef cnp.ndarray[DTYPE_bool_t, ndim=2] cluster_coverage = np.zeros((k, n), dtype=DTYPE_bool)
    for i in range(n):
        for j in range(k):
            if data_to_cluster_assignment[i, j]:
                cluster_coverage[j, i] = 1
    
    # Precompute which data points each rule covers
    cdef cnp.ndarray[DTYPE_bool_t, ndim=2] rule_coverage = np.zeros((r, n), dtype=DTYPE_bool)
    for i in range(n):
        for j in range(r):
            if data_to_rules_assignment[i, j]:
                rule_coverage[j, i] = 1
    
    # Precompute intersection of rule coverage with cluster coverage
    cdef cnp.ndarray[DTYPE_bool_t, ndim=2] rule_cluster_intersection = np.zeros((r, n), dtype=DTYPE_bool)
    for rule in range(r):
        cluster = rule_labels[rule]
        for i in range(n):
            if rule_coverage[rule, i] and cluster_coverage[cluster, i]:
                rule_cluster_intersection[rule, i] = 1
    
    # Initialize tracking arrays
    cdef cnp.ndarray[DTYPE_bool_t, ndim=1] selected_rules = np.zeros(r, dtype=DTYPE_bool)
    cdef cnp.ndarray[DTYPE_bool_t, ndim=2] covered_so_far = np.zeros((k, n), dtype=DTYPE_bool)
    cdef cnp.ndarray[DTYPE_int_t, ndim=1] selected_rule_indices = np.zeros(n_rules, dtype=DTYPE_int)
    cdef cnp.ndarray[DTYPE_bool_t, ndim=1] zero_array = np.zeros(n, dtype=DTYPE_bool)
    cdef int num_selected = 0
    
    # Main pruning loop
    cdef int best_rule
    cdef int best_rule_label
    cdef double best_score
    cdef double score
    cdef int g, c
    cdef double weight_factor
    cdef int iteration
    cdef int current_covered_count, new_covered_count
    
    for iteration in range(n_rules):
        best_rule = -1
        best_rule_label = -1
        best_score = -np.inf
        
        # Calculate weight factor for this iteration
        weight_factor = (1 - 1.0/n_rules) ** (n_rules - (iteration + 1))
        
        for rule in range(r):
            if not selected_rules[rule]:
                cluster = rule_labels[rule]
                
                # Calculate g: new coverage gained
                # This is the number of new points that would be covered by this rule
                # in the correct cluster that aren't already covered
                current_covered_count = count_union(covered_so_far[cluster], zero_array, n)
                new_covered_count = count_union(covered_so_far[cluster], rule_cluster_intersection[rule], n)
                g = new_covered_count - current_covered_count
                
                # Calculate c: coverage outside the cluster (mistakes)
                # This is the number of points covered by this rule that are NOT in the correct cluster
                c = count_union(rule_coverage[rule], zero_array, n) - \
                    count_union(rule_cluster_intersection[rule], zero_array, n)
                
                # Calculate score
                score = weight_factor * g - lambda_val * c
                
                if score > best_score:
                    best_rule = rule
                    best_rule_label = cluster
                    best_score = score
        
        # If we found a rule with positive score, select it
        if best_score > 0:
            selected_rules[best_rule] = 1
            selected_rule_indices[num_selected] = best_rule
            num_selected += 1
            
            # Update coverage for the cluster
            update_coverage(covered_so_far[best_rule_label], rule_cluster_intersection[best_rule], n)
        else:
            # No more rules with positive score, stop
            break
    
    # Return only the selected rules
    return selected_rule_indices[:num_selected]