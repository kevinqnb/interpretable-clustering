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
        pass
    
    def score(
        self,
        X : npt.NDArray,
        y : npt.NDArray = None,
        indices : npt.NDArray = None
    ) -> float:
        """
        Computes the cost associated with a leaf node.
        
        Args:
            X (np.ndarray): Input data subset.
            
            y (np.ndarray, optional): Data labels. Defaults to None.
            
            indices (np.ndarray, optional): Indices of data subset within the full dataset.
                Defaults to None. 
        
        Returns:
            score (float): Cost associated with the leaf node.
        """
        pass 
    
    
    def split(
        self,
        X : npt.NDArray,
        y : npt.NDArray = None,
        indices : npt.NDArray = None
    ) -> Tuple[float, Tuple[npt.NDArray, npt.NDArray, float]]:
        """
        Computes the best split of a leaf node.
        
        Args:
            X (np.ndarray): Input data subset.
            
            y (np.ndarray, optional): Data labels. Defaults to None.
            
            indices (np.ndarray, optional): Indices of data subset within the full dataset.
                Defaults to None. 
        
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
        X : npt.NDArray,
        y : npt.NDArray = None,
        indices : npt.NDArray = None
    ) -> Tuple[float, Tuple[npt.NDArray, npt.NDArray, float]]:
        """
        Computes the best split of a leaf node.
        
        Args:
            X (np.ndarray): Input data subset.
            
            y (np.ndarray, optional): Data labels. Defaults to None.
            
            indices (np.ndarray, optional): Indices of data subset within the full dataset.
                Defaults to None. 
        
        Returns:
            split_info ((np.ndarray, np.ndarray, float)): Features, weights,
                and threshold of the split.
        """
        n,d = X.shape
        
        best_split_val = np.inf
        best_split = None
        for feature in range(d):
            unique_vals = np.unique(X[:,feature])
            for threshold in unique_vals:
                split = ([feature], [1], threshold)

                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                left_indices = indices[left_mask]
                right_indices = indices[right_mask]
                
                if (np.sum(left_mask) < self.min_points_leaf or 
                    np.sum(right_mask) < self.min_points_leaf):
                    split_val = np.inf
                else:
                    X_l = X[left_mask, :]
                    y_l = y[left_mask] if y is not None else None
                    X1_cost = self.score(X_l, y_l, left_indices)
                    
                    X_r = X[right_mask, :]
                    y_r = y[right_mask] if y is not None else None
                    X2_cost = self.score(X_r, y_r, right_indices)
                    
                    split_val = X1_cost + X2_cost
                
                if split_val < best_split_val:
                    best_split_val = split_val
                    best_split = split
                    
        return best_split_val, best_split