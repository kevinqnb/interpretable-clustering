import numpy as np
from numpy.typing import NDArray
from intercluster.utils import entropy
from ._splitter import AxisAlignedSplitter


class InformationGainSplitter(AxisAlignedSplitter):
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
        if len(indices) == 0:
            return 0
        else:
            y_ = self.y[indices]
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