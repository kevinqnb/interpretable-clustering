import numpy as np
from numpy.typing import NDArray
from typing import Tuple, List, Set
from intercluster.measures import center_dists
from intercluster.utils import flatten_labels
from ._splitter import Splitter
from intercluster import Condition, LinearCondition
 
    
class ImmSplitter(Splitter):
    """
    Splits leaf nodes in order to 1) separate a set of reference cluster 
    centers and 2) minimize the number of points separated from their closest 
    cluster center. 
    
    This an implementation of the following work:
    "Explainable k-Means and k-Medians Clustering"
    Dasgupta, Frost, Moshkovitz, Rashtchian, 2020
    (https://arxiv.org/abs/2002.12538)
    """
    def __init__(
        self,
        centers : NDArray, 
        norm : int = 2,
        min_points_leaf : int = 1
    ):
        """
        Args:
            centers (NDArray): Array of centroid representatives.
            
            norm (int, optional): Norm to use for computing distances. 
                Takes values 1 or 2. Defaults to 2.
                
            min_points_leaf (int, optional): Minimum number of points in a leaf.
            
        Attrs:
            X (np.ndarray): Dataset for splitting
            
            y (List[Set[int]]): Associated data labels.
                NOTE: Each data point must have exactly one label.
                
            y_array (np.ndarray): Flattened one-dim array of labels.
        """
        self.centers = centers
        self.norm = norm
        self.min_points_leaf = min_points_leaf
        
    def fit(
        self,
        X : NDArray,
        y : List[Set[int]] = None
    ):
        """
        Fits the splitter to a dataset X. 
        
        Args:
            X (NDArray): Input dataset.
            
            y (List[Set[int]], optional): Dummy variable, defaults to None
                in which case the labels are manually set within the
                following method by assigning each data point to its closest center.
        """
        self.X = X
        self.y = y
        center_dist_array = center_dists(X, self.centers, self.norm, square = False)
        self.y_array = np.argmin(center_dist_array, axis = 1)
        
    def cost(
        self,
        indices : NDArray,
        centroid_indices : NDArray,
        parent_centroid_indices : NDArray
    ) -> float:
        """
        Given a set of points X, computes the cost as the number of points separated from 
        their assigned cluster center.
        
        Args:                
            indices (NDArray, optional): Indices of points to compute cost with.
            
            centroid_indices (NDArray): Indices of the leaf node's cluster centers.
            
            parent_centroid_indices (NDArray): Indices of the parent node's cluster centers.
                
        Returns:
            (float): cost of the given data.
        """
        if len(indices) == 0:
            return 0
        else:
            indices_labels = self.y_array[indices]
            mistakes = (
                ~np.isin(indices_labels, centroid_indices) &
                np.isin(indices_labels, parent_centroid_indices)
            ).sum()
            return mistakes
        
        
    def gain(
        self,
        left_indices : NDArray,
        right_indices : NDArray,
        left_centroid_indices : NDArray,
        right_centroid_indices : NDArray
    ) -> float:
        """
        Computes the gain associated with a split.
        
        Args:
            left_indices (np.ndarray): Indices for the left child of the split.
            
            right_indices (np.ndarray): Indices for the right child of the split.
            
            left_centroid_indices (np.ndarray): Indices of the left child's cluster centers.
            
            right_centroid_indices (np.ndarray): Indices of the right child's cluster centers.
            
        Returns:
            gain (float): The gain associated with the split.
        """
        parent_centroid_indices = np.unique(
            np.concatenate([left_centroid_indices, right_centroid_indices])
        )
        
        left_cost = self.cost(left_indices, left_centroid_indices, parent_centroid_indices)
        right_cost = self.cost(right_indices, right_centroid_indices, parent_centroid_indices)
        
        # Setting this to 0 simply minimizes the number of mistakes without considering 
        # what happened in the parent node.
        parent_cost = 0
        return parent_cost - (left_cost + right_cost)
    
    
    def get_split_indices(
        self,
        indices : NDArray,
        centroid_indices : NDArray,
        condition : Condition,
    ) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
        """
        Given features, weights, and threshold, returns the indices of data points 
        which fall to the left and right branches respectively.
        
        NOTE: This assumes an axis aligned split which is defined by a single feature
        with weight 1.
        
        Args:
            indices (np.ndarray): Indices for a subset of the original dataset.
            
            centroid_indices (np.ndarray): Indices of the node's cluster centers.
            
            condition (Condition): Logical or functional condition for evaluating and 
                splitting the data points.
                
        Returns:            
            left_indices (np.ndarray): Indices for the left child of the split.
            
            right_indices (np.ndarray): Indices for the right child of the split.
            
            left_centroid_indices (np.ndarray): Indices of the left child's cluster centers.
            
            right_centroid_indices (np.ndarray): Indices of the right child's cluster centers.
        """
        X_ = self.X[indices, :]
        split_left_mask = condition.evaluate(X_)
        split_right_mask = ~split_left_mask
        left_indices = indices[split_left_mask]
        right_indices = indices[split_right_mask]
        
        C_ = self.centers[centroid_indices, :]
        split_left_centroid_mask = condition.evaluate(C_)
        split_right_centroid_mask = ~split_left_centroid_mask
        left_centroid_indices = centroid_indices[split_left_centroid_mask]
        right_centroid_indices = centroid_indices[split_right_centroid_mask]
        
        return left_indices, right_indices, left_centroid_indices, right_centroid_indices
    
    
    def split(
        self,
        indices : NDArray,
        centroid_indices : NDArray
    ) -> Tuple[float, Condition]:
        """
        Computes the best split of a leaf node.
        
        Args:
            indices (np.ndarray, optional): Indices for a subset of the original dataset.
            
            centroid_indices (np.ndarray): Indices of the node's cluster centers.
        
        Returns:
            condition (Condition): Logical or functional condition for evaluating and 
                splitting the data points.
        """
        X_ = self.X[indices, :]
        C_ = self.centers[centroid_indices, :]
        n,d = X_.shape
        
        best_gain_val = -np.inf
        best_conditions = []
        for feature in range(d):
            unique_vals = np.unique(X_[:,feature])
            for threshold in unique_vals:
                condition = LinearCondition(
                    features = np.array([feature]),
                    weights = np.array([1]),
                    threshold = threshold,
                    direction = -1
                )
                (   left_indices,
                    right_indices,
                    left_centroid_indices,
                    right_centroid_indices
                ) = self.get_split_indices(indices, centroid_indices, condition)
                
                # Must separate at least one pair of cluster centers
                if (len(left_indices) < self.min_points_leaf or 
                    len(right_indices) < self.min_points_leaf):
                    gain_val = -np.inf
                    
                elif (len(left_centroid_indices) == 0 or
                    len(right_centroid_indices) == 0):
                    gain_val = -np.inf
                    
                else:
                    gain_val = self.gain(
                        left_indices,
                        right_indices,
                        left_centroid_indices,
                        right_centroid_indices
                    )
                
                if gain_val > best_gain_val:
                    best_gain_val = gain_val
                    best_conditions = [condition]
                
                elif gain_val == best_gain_val:
                    best_conditions.append(condition)
        
        # Randomly break ties if necessary:
        best_split = best_conditions[np.random.randint(len(best_conditions))]
        return best_gain_val, best_split
