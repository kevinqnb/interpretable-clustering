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
    
    def score(
        self,
        indices : npt.NDArray
    ) -> float:
        """
        Computes the cost associated with a leaf node.
        
        Args:            
            indices (np.ndarray, optional): Indices for a subset of the original dataset.
        
        Returns:
            score (float): Cost associated with given subset.
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
        
        best_split_val = np.inf
        best_splits = []
        for feature in range(d):
            unique_vals = np.unique(X_[:,feature])
            for threshold in unique_vals:
                split = ([feature], [1], threshold)

                left_mask = X_[:, feature] <= threshold
                right_mask = ~left_mask
                left_indices = indices[left_mask]
                right_indices = indices[right_mask]
                
                if (np.sum(left_mask) < self.min_points_leaf or 
                    np.sum(right_mask) < self.min_points_leaf):
                    split_val = np.inf
                else:
                    X_l_cost = self.score(left_indices)
                    X_r_cost = self.score(right_indices)
                    
                    split_val = X_l_cost + X_r_cost
                
                if split_val < best_split_val:
                    best_split_val = split_val
                    best_splits = [split]
                
                elif split_val == best_split_val:
                    best_splits.append(split)
        
        # Randomly break ties if necessary:
        best_split = best_splits[np.random.randint(len(best_splits))]
        return best_split_val, best_split


class SimpleSplitter(AxisAlignedSplitter):
    def __init__(self, min_points_leaf : int = 1):
        super().__init__(min_points_leaf = min_points_leaf)
    
    def score(self, indices : np.ndarray) -> float:
        return np.max(self.X[indices, :])
