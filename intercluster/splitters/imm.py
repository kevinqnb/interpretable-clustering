import numpy as np
import numpy.typing as npt
from typing import Tuple
from ._splitter import Splitter
from ..utils import center_dists
 
    
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
        centers : npt.NDArray, 
        norm : int = 2,
        min_points_leaf : int = 1
    ):
        """
        Args:
            centers (npt.NDArray): Array of centroid representatives.
            
            norm (int, optional): Norm to use for computing distances. 
                Takes values 1 or 2. Defaults to 2.
                
            min_points_leaf (int, optional): Minimum number of points in a leaf.
        """
        self.centers = centers
        self.norm = norm
        self.min_points_leaf = min_points_leaf
        
    def fit(
        self,
        X : npt.NDArray,
        y : npt.NDArray = None
    ):
        """
        Fits the splitter to a dataset X. 
        
        Args:
            X (npt.NDArray): Input dataset.
            
            y (npt.NDArray, optional): Dummy variable, defaults to None
                in which case the labels are manually set within the
                following method by assigning each data point to its closest center.
        """
        self.X = X
        self.center_dists = center_dists(X, self.centers, self.norm)
        self.y = np.argmin(self.center_dists, axis = 1)
        
    def cost(
        self,
        indices : npt.NDArray,
        centroid_indices : npt.NDArray,
        parent_centroid_indices : npt.NDArray
    ) -> float:
        """
        Given a set of points X, computes the cost as the sum of distances to 
        the closest center.
        
        Args:                
            indices (npt.NDArray, optional): Indices of points to compute cost with.
            
            centroid_indices (npt.NDArray): Indices of the leaf node's cluster centers.
            
            parent_centroid_indices (npt.NDArray): Indices of the parent node's cluster centers.
                
        Returns:
            (float): cost of the given data.
        """
        if len(indices) == 0:
            return 0
        else:
            indices_labels = self.y[indices]
            mistakes = (
                ~np.isin(indices_labels, centroid_indices) &
                np.isin(indices_labels, parent_centroid_indices)
            ).sum()
            return mistakes
        
        
    def gain(
        self,
        left_indices : npt.NDArray,
        right_indices : npt.NDArray,
        left_centroid_indices : npt.NDArray,
        right_centroid_indices : npt.NDArray
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
        indices : npt.NDArray,
        centroid_indices : npt.NDArray,
        split_info : Tuple[npt.NDArray, npt.NDArray, float]
    ) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
        """
        Given features, weights, and threshold, returns the indices of data points 
        which fall to the left and right branches respectively.
        
        Args:
            indices (np.ndarray): Indices for a subset of the original dataset.
            
            centroid_indices (np.ndarray): Indices of the node's cluster centers.
            
            split_info ((np.ndarray, np.ndarray, float)): Features, weights,
                and threshold of the split.

        Returns:            
            left_indices (np.ndarray): Indices for the left child of the split.
            
            right_indices (np.ndarray): Indices for the right child of the split.
            
            left_centroid_indices (np.ndarray): Indices of the left child's cluster centers.
            
            right_centroid_indices (np.ndarray): Indices of the right child's cluster centers.
        """
        X_ = self.X[indices, :]
        C_ = self.centers[centroid_indices, :]
        feature, weight, threshold = split_info
        feature = feature[0]
        left_mask = X_[:, feature] <= threshold
        right_mask = ~left_mask
        left_indices = indices[left_mask]
        right_indices = indices[right_mask]
        
        left_centroid_mask = C_[:, feature] <= threshold
        right_centroid_mask = ~left_centroid_mask
        left_centroid_indices = centroid_indices[left_centroid_mask]
        right_centroid_indices = centroid_indices[right_centroid_mask]
        
        return left_indices, right_indices, left_centroid_indices, right_centroid_indices
    
    
    def split(
        self,
        indices : npt.NDArray,
        centroid_indices : npt.NDArray
    ) -> Tuple[float, Tuple[npt.NDArray, npt.NDArray, float]]:
        """
        Computes the best split of a leaf node.
        
        Args:
            indices (np.ndarray, optional): Indices for a subset of the original dataset.
            
            centroid_indices (np.ndarray): Indices of the node's cluster centers.
        
        Returns:
            split_info ((np.ndarray, np.ndarray, float)): Features, weights,
                and threshold of the split.
        """
        X_ = self.X[indices, :]
        C_ = self.centers[centroid_indices, :]
        n,d = X_.shape
        
        best_gain_val = -np.inf
        best_splits = []
        for feature in range(d):
            unique_vals = np.unique(X_[:,feature])
            for threshold in unique_vals:
                split = ([feature], [1], threshold)
                (   left_indices,
                    right_indices,
                    left_centroid_indices,
                    right_centroid_indices
                ) = self.get_split_indices(indices, centroid_indices, split)
                
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
                    best_splits = [split]
                
                elif gain_val == best_gain_val:
                    best_splits.append(split)
        
        # Randomly break ties if necessary:
        best_split = best_splits[np.random.randint(len(best_splits))]
        return best_gain_val, best_split    