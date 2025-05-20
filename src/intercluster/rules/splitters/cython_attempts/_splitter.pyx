cimport cython
cimport numpy as cnp
import numpy as np
from itertools import combinations, permutations
from intercluster.utils import can_flatten, flatten_labels
from .._conditions import Condition, LinearCondition

# Typing
from numpy.typing import NDArray
cnp.import_array()
from typing import Tuple, List, Set


DTYPE_float = np.float64
ctypedef cnp.float64_t DTYPE_float_t
DTYPE_int = np.int64
ctypedef cnp.int64_t DTYPE_int_t


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
    def __init__(self):
        pass
    
    
    def fit(
        self,
        cnp.ndarray[DTYPE_float_t, ndim = 2] X,
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
        cnp.ndarray[DTYPE_int_t, ndim = 1] indices
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
        cnp.ndarray[DTYPE_int_t, ndim = 1] left_indices,
        cnp.ndarray[DTYPE_int_t, ndim = 1] right_indices,
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
        cnp.ndarray[DTYPE_float_t, ndim = 2] X_,
        cnp.ndarray[DTYPE_int_t, ndim = 1] indices,
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
        #cdef cnp.ndarray[DTYPE_float_t, ndim = 2] X_ = self.X[indices, :]
        split_left_mask = condition.evaluate(X_)
        split_right_mask = ~split_left_mask
        cdef cnp.ndarray[DTYPE_int_t, ndim = 1] left_indices = indices[split_left_mask]
        cdef cnp.ndarray[DTYPE_int_t, ndim = 1] right_indices = indices[split_right_mask]
        return left_indices, right_indices


    def split(
        self,
        cnp.ndarray[DTYPE_int_t, ndim = 1] indices,
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
    def __init__(self, int min_points_leaf = 1):
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
        cnp.ndarray[DTYPE_int_t, ndim = 1] left_indices,
        cnp.ndarray[DTYPE_int_t, ndim = 1] right_indices,
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
        cdef cnp.ndarray[DTYPE_int_t, ndim = 1] parent_indices
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
        cnp.ndarray[DTYPE_int_t, ndim = 1] indices,
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
        cdef cnp.ndarray[DTYPE_float_t, ndim = 2] X_ = self.X[indices, :]
        cdef int n = X_.shape[0]
        cdef int d = X_.shape[1]
        cdef float parent_cost = self.cost(indices)
        
        cdef int feature
        cdef float threshold, gain_val, best_gain_val
        cdef object condition, best_condition
        cdef cnp.ndarray[DTYPE_int_t, ndim = 1] left_indices, right_indices
        cdef list best_conditions

        best_gain_val = -np.inf
        best_conditions = []
        for feature in range(d):
            for threshold in np.unique(X_[:,feature]):
                condition = LinearCondition(
                    features = np.array([feature]),
                    weights = np.array([1]),
                    threshold = threshold,
                    direction = -1
                )
                left_indices, right_indices = self.get_split_indices(X_, indices, condition)
                
                if (len(left_indices) < self.min_points_leaf or 
                    len(right_indices) < self.min_points_leaf):
                    gain_val = -np.inf
                else:
                    gain_val = self.gain(left_indices, right_indices, parent_cost)
                
                if gain_val > best_gain_val:
                    best_gain_val = gain_val
                    best_conditions = [condition]
                
                elif gain_val == best_gain_val:
                    best_conditions.append(condition)
        
        # Randomly break ties if necessary:
        best_condition = best_conditions[np.random.randint(len(best_conditions))]
        return best_gain_val, best_condition


####################################################################################################

'''
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
        X_ = self.X[indices, :]
        n,d = X_.shape
        parent_cost = self.cost(indices)
        
        # Search only through pairs of features, and select slopes within that 
        # low dimensional space. 
        feature_pairs = list(combinations(range(d), 2))
        slopes = np.array([0, 1/2, 1, 2, np.inf, -2, -1, -1/2])

        best_gain_val = -np.inf
        best_conditions = []
        for pair in feature_pairs:
            X_pair = X_[:, pair]
            for i in range(X_pair.shape[0]):
                data_point = X_pair[i,:]
                for slope in slopes:
                    if slope == 0:
                        # axis aligned case (horizontal)
                        condition = LinearCondition(
                            features = np.array([pair[1]]),
                            weights = np.array([1]),
                            threshold = data_point[1],
                            direction = -1
                        )
                    elif slope == np.inf:
                        # axis aligned case (vertical)
                        condition = LinearCondition(
                            features = np.array([pair[0]]),
                            weights = np.array([1]),
                            threshold = data_point[0],
                            direction = -1
                        )
                    else:
                        # non-axis aligned slopes
                        threshold = -slope * data_point[0] + data_point[1]
                        condition = LinearCondition(
                            features = np.array(pair),
                            weights = np.array([-slope, 1]),
                            threshold = threshold,
                            direction = -1
                        )

                    left_indices, right_indices = self.get_split_indices(indices, condition)
                    
                    if (len(left_indices) < self.min_points_leaf or 
                        len(right_indices) < self.min_points_leaf):
                        gain_val = -np.inf
                    else:
                        gain_val = self.gain(left_indices, right_indices, parent_cost)
                    
                    if gain_val > best_gain_val:
                        best_gain_val = gain_val
                        best_conditions = [condition]
                    
                    elif gain_val == best_gain_val:
                        best_conditions.append(condition)
        
        # Randomly break ties if necessary:
        best_condition = best_conditions[np.random.randint(len(best_conditions))]
        return best_gain_val, best_condition
    

####################################################################################################
'''