import numpy as np
from itertools import combinations, permutations
from numpy.typing import NDArray
from typing import Tuple, List, Set
from intercluster.utils import can_flatten, flatten_labels
from intercluster import Condition, LinearCondition
from .cython.information_gain import split_cy, oblique_split_cy


####################################################################################################


class Splitter:
    """
    Base class for a Splitting object designed to split leaf nodes 
    in a decision tree.
    
    Attrs:
        X (np.ndarray): Dataset for splitting
        
        y (List[Set[int]]): Associated data labels.
            NOTE: Each data point must have exactly one label.
            
        y_array (np.ndarray): Flattened one-dim array of labels.
    """
    def __init__(self, min_points_leaf : int = 1):
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
            
            y (List[Set[int]], optional): Target labels. Defaults to None.
        """
        self.X = X
        self.y = y
        if not can_flatten(y):
            raise ValueError("Data points must each have a single label.")
        self.y_array = flatten_labels(y)
    
    def cost(
        self,
        indices : NDArray
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
        pass
        
        
    def get_split_indices(
        self,
        indices : NDArray,
        condition : Condition
    ) -> Tuple[float, NDArray, NDArray]:
        """
        Given an evaluation condtion returns the indices of data points 
        which fall to the left and right branches respectively.
        
        Args:
            indices (np.ndarray): Original array of data point indices to be split.
            
            condition (Condition): Logical or functional condition for evaluating and 
                splitting the data points.

        Returns:
            cost (float): Cost associated with the split.
            
            left_indices (np.ndarray): Indices for the left child of the split.
            
            right_indices (np.ndarray): Indices for the right child of the split.
        """
        X_ = self.X[indices, :]
        split_left_mask = condition.evaluate(X_)
        split_right_mask = ~split_left_mask
        left_indices = indices[split_left_mask]
        right_indices = indices[split_right_mask]
        return left_indices, right_indices
    
    
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
        pass
    

####################################################################################################


class AxisAlignedSplitter(Splitter):
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
        
        left_cost = self.cost(left_indices)
        right_cost = self.cost(right_indices)
        return parent_cost - (left_cost + right_cost)
    
    
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
        gain_val, condition_tuple = split_cy(
            X = self.X,
            y = self.y_array,
            indices = indices,
            min_points_leaf = self.min_points_leaf,
        )
        condition = LinearCondition(
            features = np.array([condition_tuple[0]]),
            weights = np.array([1]),
            threshold = condition_tuple[1],
            direction = -1
        )
        return gain_val, condition
    

####################################################################################################


class SimpleObliqueSplitter(Splitter):
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
        
        left_cost = self.cost(left_indices)
        right_cost = self.cost(right_indices)
        return parent_cost - (left_cost + right_cost)
    
    
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
        gain_val, condition = oblique_split_cy(
            X = self.X,
            indices = indices,
            min_points_leaf = self.min_points_leaf,
            get_split_indices_fn = self. get_split_indices,
            cost_fn = self.cost,
            gain_fn = self.gain
        )
        return gain_val, condition
    

####################################################################################################