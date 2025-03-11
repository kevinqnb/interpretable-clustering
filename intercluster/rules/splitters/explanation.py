import numpy as np
from numpy.typing import NDArray
from typing import List, Set, Tuple
from ._splitter import AxisAlignedSplitter
from .._conditions import Condition, LinearCondition

####################################################################################################


class ExplanationSplitter(AxisAlignedSplitter):
    """
    Splits leaf nodes and removes outliers, in order to create an explainable clustering 
    for the remaining set of points. This follows the explainable clustering 
    algorithm outlied by Bandyapadhyay et al. in their paper titled 
    "How to Find a Good Explanation for Clustering?" (2022 AAAI).
    """
    def __init__(
        self,
        num_clusters : int,
        min_points_leaf : int = 1
    ):
        """
        Args:
            num_clusters (int): Number of clusters to split.

            min_points_leaf (int, optional): Minimum number of points in a leaf.
            
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
            left_indices : NDArray,
            right_indices : NDArray
        ) -> Tuple[NDArray, NDArray]:
        """
        Finds outliers to be removed from set of indices. 
        """
        new_outliers = set()
        left_indices_rem = set(left_indices) - self.outliers
        right_indices_rem = set(right_indices) - self.outliers

        # Count number of clusters appearances in each half
        left_cluster_satisfies = {i:set() for i in range(self.num_clusters)}
        right_cluster_satisfies = {i:set() for i in range(self.num_clusters)}
        left_cluster_counts = np.zeros(self.num_clusters)
        right_cluster_counts = np.zeros(self.num_clusters)

        for i in left_indices_rem:
            i_cluster = self.y_array[i]
            left_cluster_satisfies[i_cluster].add(i)
            left_cluster_counts[i_cluster] += 1

        for i in right_indices_rem:
            i_cluster = self.y_array[i]
            right_cluster_satisfies[i_cluster].add(i)
            right_cluster_counts[i_cluster] += 1

        both_zeros = (left_cluster_counts == 0) & (right_cluster_counts == 0)

        # Case 1. All clusters have majority in the left indices:
        if np.sum((right_cluster_counts >= left_cluster_counts) & (~both_zeros)) == 0:
            left_positive = np.where(left_cluster_counts > right_cluster_counts)[0]
            positive_counts = left_cluster_counts[left_positive]
            minimum_majority_cluster = left_positive[np.argmin(positive_counts)]

            # Effectively assign minimum majority to right by removing its left satisfying points.
            new_outliers = new_outliers.union(left_cluster_satisfies[minimum_majority_cluster])

            # Effectively assign all others to left by rmeoving their right satisfying points.
            for i in range(self.num_clusters):
                if i != minimum_majority_cluster:
                    new_outliers = new_outliers.union(right_cluster_satisfies[i])


        # Case 2. All clusters have majority in the right indices:
        elif np.sum((left_cluster_counts >= right_cluster_counts) & (~both_zeros)) == 0:
            right_positive = np.where(right_cluster_counts > left_cluster_counts)[0]
            positive_counts = right_cluster_counts[right_positive]
            minimum_majority_cluster = right_positive[np.argmin(positive_counts)]

            # Effectively assign minimum majority to left by removing its right satisfying points.
            new_outliers = new_outliers.union(right_cluster_satisfies[minimum_majority_cluster])

            # Effectively assign all others to right by rmeoving their left satisfying points.
            for i in range(self.num_clusters):
                if i != minimum_majority_cluster:
                    new_outliers = new_outliers.union(left_cluster_satisfies[i])

        # Case 3. Cluster membership is mixed:
        else:
            left_positive = np.where(left_cluster_counts > right_cluster_counts)[0]
            right_positive = np.where(right_cluster_counts > left_cluster_counts)[0]
            tied = np.where(
                (left_cluster_counts == right_cluster_counts) & (~both_zeros)
            )

            # Distinct majorities are assigned to their respective halves.
            left_right_assignment = np.zeros(self.num_clusters)
            left_right_assignment[left_positive] = -1
            left_right_assignment[right_positive] = 1
            left_assigned = 0
            right_assigned = 0

            # Ties are broken, and then rebroken if necessary (until both halves have at least 
            # one cluster).
            left_assigned = 0
            right_assigned = 0

            iter_tracker = 0
            while left_assigned == 0 or right_assigned == 0:
                iter_tracker += 1
                if iter_tracker > 100:
                    print("hey slow down! " + str(iter_tracker))
                left_assigned = len(left_positive)
                right_assigned = len(right_positive)
                left_right_assignment[tied] = 0

                for i in tied:
                    # Flip a coin and assign
                    coin_val = np.random.uniform()
                    if coin_val <= 0.5:
                        left_assigned += 1
                        left_right_assignment[i] = -1
                    else:
                        right_assigned += 1
                        left_right_assignment[i] = 1

            # Track outliers based on cluster assignment
            for i,assign_val in enumerate(left_right_assignment):
                if assign_val == -1:
                    # remove right indices
                    new_outliers = new_outliers.union(right_cluster_satisfies[i])
                elif assign_val == 1:
                    # remove left indices
                    new_outliers = new_outliers.union(left_cluster_satisfies[i])
                else:
                    # must be a case where the cluster is not present. 
                    pass

        return new_outliers

    def update_outliers(self, new_outliers : Set[int]):
        """
        updates the outlier list.
        """
        self.outliers = self.outliers.union(new_outliers)

    def cost(
        self,
        indices : NDArray
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
        left_indices : NDArray,
        right_indices : NDArray,
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
        parent_indices = np.unique(np.concatenate([left_indices, right_indices]))
        
        if len(parent_indices) < len(left_indices) + len(right_indices):
            raise ValueError("Indices are not disjoint.")
        
        # If only a single cluster is present, no gain to be had.
        left_indices_rem = set(left_indices) - self.outliers
        right_indices_rem = set(right_indices) - self.outliers
        indices_rem = left_indices_rem.union(right_indices_rem)
        clusters_present = self.y_array[list(indices_rem)]
        if np.all(clusters_present == clusters_present[0]):
            return -np.inf

        # Setting this to 0 simply minimizes the number of outliers removed without considering 
        # what happened in the parent node.
        parent_cost = 0

        split_outliers = self.get_split_outliers(left_indices, right_indices)
        split_cost = len(split_outliers)
        return parent_cost - (split_cost)
    
        