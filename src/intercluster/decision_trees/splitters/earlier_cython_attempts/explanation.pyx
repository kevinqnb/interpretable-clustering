cimport cython
cimport numpy as cnp
import numpy as np
from itertools import combinations, permutations
from intercluster.utils import can_flatten, flatten_labels
from .._conditions import Condition, LinearCondition
from ._splitter import AxisAlignedSplitter

# Typing
from numpy.typing import NDArray
cnp.import_array()
from typing import Tuple, List, Set

DTYPE_float = np.float64
ctypedef cnp.float64_t DTYPE_float_t
DTYPE_int = np.int64
ctypedef cnp.int64_t DTYPE_int_t

####################################################################################################


class ExplanationSplitter(AxisAlignedSplitter):
    """
    Splits leaf nodes and removes outliers, in order to create an explainable clustering 
    for the remaining set of points. This follows the explainable clustering 
    algorithm outlied by Bandyapadhyay et al. in their paper titled 
    "How to Find a Good Explanation for Clustering?" (2022 AAAI).
    """
    #cdef int num_clusters
    #cdef object outliers

    def __init__(
        self,
        int num_clusters,
        int min_points_leaf = 1
    ):
        """
        Args:
            num_clusters (int): Number of clusters to split.

            min_points_leaf (int, optional): Minimum number of points in a leaf.

            cpu_count (int, optional): Number of processors to use. Defaults to 1.
            
        Attrs:
            X (np.ndarray): Dataset for splitting
            
            y (List[Set[int]]): Associated data labels.
                NOTE: Each data point must have exactly one label.
                
            y_array (np.ndarray): Flattened one-dim array of labels.

            outliers (Set[int]): List of data indices to be removed as outliers.
        """
        self.num_clusters = num_clusters
        super().__init__(min_points_leaf = min_points_leaf)
        self.outliers = set()


    def get_split_outliers(
        self,
        cnp.ndarray[DTYPE_int_t, ndim = 1] left_indices,
        cnp.ndarray[DTYPE_int_t, ndim = 1] right_indices
    ) -> NDArray:
        """
        Finds outliers to be removed from set of indices.

        Args:
            left_indices (np.ndarray): Indices for the left child of the split.
            
            right_indices (np.ndarray): Indices for the right child of the split.

        Returns:
            outliers (np.ndarray): Indices of the data points to be removed as outliers.
        """
        cdef int n = len(self.y_array)
        cdef cnp.ndarray[DTYPE_int_t, ndim = 1] unique_clusters = np.unique(
            self.y_array[np.concatenate((left_indices, right_indices))]
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
            cluster = self.y_array[idx]
            left_half_satisfies[cluster_idx_dict[cluster], idx] = 1
            left_half_counts[cluster_idx_dict[cluster]] += 1

        for i,idx in enumerate(right_indices):
            cluster = self.y_array[idx]
            cluster = self.y_array[idx]
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

    def update_outliers(self, new_outliers : NDArray):
        """
        Updates the outlier list.

        Args:
            new_outliers (Set[int]): Set of data points to include as new outliers. 
        """
        self.outliers = self.outliers.union(new_outliers)


    def cost(
        self,
        cnp.ndarray[DTYPE_int_t, ndim = 1] indices
    ) -> float:
        """
        Computes the cost associated with a leaf node. For this splitter, cost is a 
        dummy method, always returning 0.
        
        Args:            
            indices (np.ndarray, optional): Indices for a subset of the original dataset.
        
        Returns:
            cost (float): Cost associated with given subset.
        """
        return 0
    
    def gain(
        self,
        cnp.ndarray[DTYPE_int_t, ndim = 1] left_indices,
        cnp.ndarray[DTYPE_int_t, ndim = 1] right_indices,
        parent_cost : float = None
    ) -> float:
        """
        Computes the gain associated with a split.
        
        Args:
            left_indices (np.ndarray): Indices for the left child of the split.
            
            right_indices (np.ndarray): Indices for the right child of the split.
            
            parent_cost (float, optional): The cost of the parent node. Dummy variable, 
                Defaults to None.
        
        Returns:
            gain (float): The gain associated with the split.
        """
        cdef cnp.ndarray[DTYPE_int_t, ndim = 1] parent_indices
        parent_indices = np.unique(np.concatenate([left_indices, right_indices]))
        
        if len(parent_indices) < len(left_indices) + len(right_indices):
            raise ValueError("Indices are not disjoint.")
        
        # If only a single cluster is present, no gain to be had.
        cdef cnp.ndarray[DTYPE_int_t, ndim = 1] clusters_present = self.y_array[parent_indices]
        if np.all(clusters_present == clusters_present[0]):
            return -np.inf

        # Setting this to 0 simply minimizes the number of outliers removed without considering 
        # what happened in the parent node.
        parent_cost = 0
        cdef cnp.ndarray[DTYPE_int_t, ndim = 1] split_outliers = self.get_split_outliers(
            left_indices, right_indices
        )
        split_cost = len(split_outliers)
        return parent_cost - (split_cost)
    
        