import numpy as np
from joblib import Parallel, delayed, parallel_config
from intercluster.utils import tiebreak
from numpy.typing import NDArray
from typing import List, Set, Tuple
from ._splitter import AxisAlignedSplitter
from .._conditions import Condition, LinearCondition
from ._explanation import _get_split_outliers

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
        min_points_leaf : int = 1,
        cpu_count : int = 1
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
        self.cpu_count = cpu_count
        super().__init__(min_points_leaf = min_points_leaf)
        self.outliers = set()


    def get_split_outliers(
            self,
            left_indices : NDArray,
            right_indices : NDArray
        ) -> Tuple[NDArray, NDArray]:
        """
        Finds outliers to be removed from set of indices. Note that this a wrapper to an
        internal cython function.

        Args:
            left_indices (np.ndarray): Indices for the left child of the split.
            
            right_indices (np.ndarray): Indices for the right child of the split.

        Returns:
            outliers (np.ndarray): Indices of the data points to be removed as outliers.
        """
        outliers = _get_split_outliers(self.y_array, left_indices, right_indices)
        return outliers

    def update_outliers(self, new_outliers : Set[int]):
        """
        Updates the outlier list.

        Args:
            new_outliers (Set[int]): Set of data points to include as new outliers. 
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
        clusters_present = self.y_array[parent_indices]
        if np.all(clusters_present == clusters_present[0]):
            return -np.inf

        # Setting this to 0 simply minimizes the number of outliers removed without considering 
        # what happened in the parent node.
        parent_cost = 0
        split_outliers = self.get_split_outliers(left_indices, right_indices)
        split_cost = len(split_outliers)
        return parent_cost - (split_cost)
    

    def evaluate_condition(self, indices : NDArray, condition : Condition) -> float:
        """
        Evaluates the gain of a condition upon a set of remaning indices

        Args:
            indices (np.ndarray): Array of remaining index values.

            condition (Condition): Logical condition object for splitting indices. 

        Returns
            gain (float): Gain to be acheived via splitting indices upon the given condition.
        """
        left_indices, right_indices = self.get_split_indices(indices, condition)
        gain_val = None
        if (len(left_indices) < self.min_points_leaf or 
            len(right_indices) < self.min_points_leaf):
            gain_val = -np.inf
        else:
            gain_val = self.gain(left_indices, right_indices)

        return gain_val



    def split(
        self,
        indices : NDArray
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
        X_ = self.X[indices, :]
        n,d = X_.shape

        condition_list = []
        for feature in range(d):
            unique_vals = np.unique(X_[:,feature])
            for threshold in unique_vals:
                condition = LinearCondition(
                    features = np.array([feature]),
                    weights = np.array([1]),
                    threshold = threshold,
                    direction = -1
                )
                condition_list.append(condition)

        gain_vals = Parallel(n_jobs=self.cpu_count, backend = 'loky')(
                delayed(self.evaluate_condition)(indices, cond)
                for cond in condition_list
        )
        
        best_condition_idx = tiebreak(scores = -1 * np.array(gain_vals))[0]
        best_gain = gain_vals[best_condition_idx]
        best_condition = condition_list[best_condition_idx]
        return best_gain, best_condition
    
        