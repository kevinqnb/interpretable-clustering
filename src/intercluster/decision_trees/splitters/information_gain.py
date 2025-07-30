import numpy as np
from numpy.typing import NDArray
from intercluster.measures import entropy
from ._splitter import Splitter
from typing import Tuple
from intercluster import Condition, LinearCondition
from .cython.information_gain import cost_cy, gain_cy, split_cy


####################################################################################################


class InformationGainSplitter(Splitter):
    """
    Splits leaf nodes in order to maximize information gain.
    """
    def __init__(
        self,
        min_points_leaf : int = 1
    ):
        """
        Args:
            min_points_leaf (int, optional): Minimum number of points in a leaf.
            
        Attrs:
            X (np.ndarray): Dataset for splitting
            
            y (List[Set[int]]): Associated data labels.
                NOTE: Each data point must have exactly one label.
                
            y_array (np.ndarray): Flattened one-dim array of labels.
        """
        super().__init__(min_points_leaf = min_points_leaf)
        
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
        '''
        if len(indices) == 0:
            return 0
        else:
            y_ = self.y_array[indices]
            entropy_score = entropy(y_) 
            return entropy_score
        '''
        return cost_cy(self.y_array, indices)
        
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
        '''
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
        '''
        if parent_cost is None:
            parent_cost = self.cost(parent_indices)

        return gain_cy(
            n = len(self.X),
            y = self.y_array,
            left_indices = left_indices,
            right_indices = right_indices,
            parent_cost = parent_cost
        )
    

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


class ObliqueInformationGainSplitter(InformationGainSplitter):
    """
    Splits leaf nodes in order to maximize information gain.
    """
    def __init__(
        self,
        min_points_leaf : int = 1
    ):
        """
        Args:
            min_points_leaf (int, optional): Minimum number of points in a leaf.
            
        Attrs:
            X (np.ndarray): Dataset for splitting
            
            y (List[Set[int]]): Associated data labels.
                NOTE: Each data point must have exactly one label.
                
            y_array (np.ndarray): Flattened one-dim array of labels.
        """
        super().__init__(min_points_leaf = min_points_leaf)


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
        pair, slope, threshold = condition_tuple
        if slope == 0:
            condition = LinearCondition(
                features = np.array([pair[1]]),
                weights = np.array([1]),
                threshold = threshold,
                direction = -1
            )
        elif slope == np.inf:
            condition = LinearCondition(
                features = np.array([pair[0]]),
                weights = np.array([1]),
                threshold = threshold,
                direction = -1
            )
        else:
            condition = LinearCondition(
                features = np.array(pair),
                weights = np.array([-slope, 1]),
                threshold = threshold,
                direction = -1
            )

        return gain_val, condition
        

    

####################################################################################################
