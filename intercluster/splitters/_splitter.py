import numpy as np
import numpy.typing as npt
from typing import List, Tuple


class Splitter:
    """
    Base class for a Splitting object designed to split leaf nodes 
    in a decision tree.
    """
    def __init__(self):
        pass
    
    
    def fit(
        self,
        X : npt.NDArray,
        y : npt.NDArray = None
    ):
        """
        Fits the splitter to a dataset X. 
        
        Args:
            X (npt.NDArray): Input dataset.
            
            y (npt.NDArray, optional): Target labels. Defaults to None.
        """
        self.X = X
        self.y = y
    
    def cost(
        self,
        indices : npt.NDArray
    ) -> float:
        """
        Computes the cost associated with a leaf node.
        
        Args:            
            indices (np.ndarray, optional): Indices for a subset of the original dataset.
        
        Returns:
            cost (float): Cost associated with given subset.
        """
        pass
    
    
    def gain(
        self,
        left_indices : npt.NDArray,
        right_indices : npt.NDArray,
        parent_cost : float = None
    ) -> float:
        """
        Computes the gain associated with a split.
        
        Args:
            left_indices (np.ndarray): Indices for the left child of the split.
            
            right_indices (np.ndarray): Indices for the right child of the split.
            
            parent_cost (float, optional): The cost of the parent node. Defaults to None.
        
        Returns:
            gain (float): The gain associated with the split.
        """
        pass
        
        
    def get_split_indices(
        self,
        indices : npt.NDArray,
        split_info : Tuple[npt.NDArray, npt.NDArray, float]
    ) -> Tuple[float, npt.NDArray, npt.NDArray]:
        """
        Given features, weights, and threshold, returns the indices of data points 
        which fall to the left and right branches respectively.
        
        Args:
            split_info ((np.ndarray, np.ndarray, float)): Features, weights,
                and threshold of the split.

        Returns:
            cost (float): Cost associated with the split.
            
            left_indices (np.ndarray): Indices for the left child of the split.
            
            right_indices (np.ndarray): Indices for the right child of the split.
        """
        pass
    
    
    def split(
        self,
        indices : npt.NDArray
    ) -> Tuple[float, Tuple[npt.NDArray, npt.NDArray, float]]:
        """
        Computes the best split of a leaf node.
        
        Args:
            indices (np.ndarray, optional): Indices for a subset of the original dataset.
        
        Returns:
            split_info ((np.ndarray, np.ndarray, float)): Features, weights,
                and threshold of the split.
        """
        pass
    
    

class AxisAlignedSplitter(Splitter):
    def __init__(self, min_points_leaf : int = 1):
        """
        Args:
            min_points_leaf (int, optional): Minimum number of points in a leaf node. 
                Defaults to 1.
        """
        self.min_points_leaf = min_points_leaf
        
        
    def gain(
        self,
        left_indices : npt.NDArray,
        right_indices : npt.NDArray,
        parent_cost : float = None
    ) -> float:
        """
        Computes the gain associated with a split.
        
        Args:
            left_indices (np.ndarray): Indices for the left child of the split.
            
            right_indices (np.ndarray): Indices for the right child of the split.
            
            parent_cost (float, optional): The cost of the parent node. Defaults to None.
        
        Returns:
            gain (float): The gain associated with the split.
        """
        parent_indices = np.unique(np.concatenate([left_indices, right_indices]))
        
        if len(parent_indices) < len(left_indices) + len(right_indices):
            raise ValueError("Indices are not disjoint.")
        
        if parent_cost is None:
            parent_cost = self.cost(parent_indices)
        
        left_cost = self.cost(left_indices)
        right_cost = self.cost(right_indices)
        return parent_cost - (left_cost + right_cost)
        
        
    def get_split_indices(
        self,
        indices : npt.NDArray,
        split_info : Tuple[npt.NDArray, npt.NDArray, float]
    ) -> Tuple[npt.NDArray, npt.NDArray]:
        """
        Given features, weights, and threshold, returns the indices of data points 
        which fall to the left and right branches respectively.
        
        Args:
            indices (np.ndarray): Indices for a subset of the original dataset.
            
            split_info ((np.ndarray, np.ndarray, float)): Features, weights,
                and threshold of the split.

        Returns:            
            left_indices (np.ndarray): Indices for the left child of the split.
            
            right_indices (np.ndarray): Indices for the right child of the split.
        """
        X_ = self.X[indices, :]
        feature, weight, threshold = split_info
        feature = feature[0]
        left_mask = X_[:, feature] <= threshold
        right_mask = ~left_mask
        left_indices = indices[left_mask]
        right_indices = indices[right_mask]
        
        return left_indices, right_indices
    
    
    def split(
        self,
        indices : npt.NDArray
    ) -> Tuple[float, Tuple[npt.NDArray, npt.NDArray, float]]:
        """
        Computes the best split of a leaf node.
        
        Args:
            indices (np.ndarray, optional): Indices for a subset of the original dataset.
        
        Returns:
            split_info ((np.ndarray, np.ndarray, float)): Features, weights,
                and threshold of the split.
        """
        X_ = self.X[indices, :]
        n,d = X_.shape
        parent_cost = self.cost(indices)
        
        best_gain_val = -np.inf
        best_splits = []
        for feature in range(d):
            unique_vals = np.unique(X_[:,feature])
            for threshold in unique_vals:
                split = ([feature], [1], threshold)
                left_indices, right_indices = self.get_split_indices(indices, split)
                
                if (len(left_indices) < self.min_points_leaf or 
                    len(right_indices) < self.min_points_leaf):
                    gain_val = -np.inf
                else:
                    gain_val = self.gain(left_indices, right_indices, parent_cost)
                
                if gain_val > best_gain_val:
                    best_gain_val = gain_val
                    best_splits = [split]
                
                elif gain_val == best_gain_val:
                    best_splits.append(split)
        
        # Randomly break ties if necessary:
        best_split = best_splits[np.random.randint(len(best_splits))]
        return best_gain_val, best_split
