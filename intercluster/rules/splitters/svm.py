import numpy as np
from sklearn.svm import LinearSVC
from numpy.typing import NDArray
from typing import Tuple
from intercluster.utils import entropy
from ._splitter import Splitter
from .._conditions import Condition, LinearCondition
 


class SVMSplitter(Splitter):
    def __init__(self, min_points_leaf : int = 1):
        """
        Args:
            min_points_leaf (int, optional): Minimum number of points in a leaf node. 
                Defaults to 1.
                
        Attrs:
            X (np.ndarray): Dataset for splitting
            
            y (List[Set[int]]): Associated data labels.
                NOTE: Each data point must have exactly one label.
                
            y_array (np.ndarray): Flattened one-dim array of labels.
        """
        self.min_points_leaf = min_points_leaf
        
    def cost(
        self,
        indices : NDArray
    ) -> float:
        """
        Given a set of points X, computes the score as the entropy of the labels.
        
        Args:                
            indices (NDArray, optional): Indices of points to compute score with.
                
        Returns:
            (float): Score of the given data.
        """
        if len(indices) == 0:
            return 0
        else:
            y_ = self.y_array[indices]
            entropy_score = entropy(y_) 
            return entropy_score
        
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
            
            parent_cost (float, optional): The cost of the parent node. Defaults to None.
        
        Returns:
            gain (float): The gain associated with the split.
        """
        parent_indices = np.unique(np.concatenate([left_indices, right_indices]))
        
        if len(parent_indices) < len(left_indices) + len(right_indices):
            raise ValueError("Indices are not disjoint.")
        
        if parent_cost is None:
            parent_cost = self.cost(parent_indices)
        
        # This calculates the relative reduction in impurity, based on the 
        # definitions used by SKLearn's decision tree model.
        left_cost = len(left_indices)/len(parent_indices) * self.cost(left_indices)
        right_cost = len(right_indices)/len(parent_indices) * self.cost(right_indices)
        return len(parent_indices)/len(self.X) * (parent_cost - (left_cost + right_cost))
    
    
    def split(
        self,
        indices : NDArray
    ) -> Tuple[float, Condition]:
        """
        Computes the best split of a leaf node.
        
        Args:
            indices (np.ndarray, optional): Indices for a subset of the original dataset.
        
        Returns:
            condition (Condition): Logical or functional condition for evaluating and 
                splitting the data points.
        """
        n,d = self.X.shape
        X_ = self.X[indices, :]
        y_ = self.y_array[indices]
        parent_cost = self.cost(indices)
        
        if len(np.unique(y_)) == 1:
            gain_val = -np.inf
            condition = None
        else:
            svm = LinearSVC(max_iter = 1000).fit(X_, y_)
            features = np.arange(d)
            weights = np.ndarray.flatten(svm.coef_)
            threshold = -svm.intercept_[0]
            condition = LinearCondition(
                    features = features,
                    weights = weights,
                    threshold = threshold,
                    direction = -1
                )
            
            left_indices, right_indices = self.get_split_indices(indices, condition)
                    
            if (len(left_indices) < self.min_points_leaf or 
                len(right_indices) < self.min_points_leaf):
                gain_val = -np.inf
            else:
                gain_val = self.gain(left_indices, right_indices, parent_cost)
        
        return gain_val, condition