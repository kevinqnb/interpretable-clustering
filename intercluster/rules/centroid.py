import numpy as np
import numpy.typing as npt
from typing import List
from ._splitter import AxisAlignedSplitter
from ._node import Node
from ._tree import Tree


class CentroidSplitter(AxisAlignedSplitter):
    """
    Splits leaf nodes in order to minimize distances to a set of input centers.
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
        super().__init__(min_points_leaf = min_points_leaf)
        
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
        
        if self.norm == 2:
            diffs = X[np.newaxis, :, :] - self.centers[:, np.newaxis, :]
            distances = np.sum((diffs)**2, axis=-1)
            
        elif self.norm == 1:
            diffs = X[np.newaxis, :, :] - self.centers[:, np.newaxis, :]
            distances = np.sum(np.abs(diffs), axis=-1)
        
        self.center_dists = distances.T
        
    def score(
        self,
        indices : npt.NDArray
    ) -> float:
        """
        Given a set of points X, computes the score as the sum of distances to 
        the closest center.
        
        Args:                
            indices (npt.NDArray, optional): Indices of points to compute score with.
                
        Returns:
            (float): Score of the given data.
        """
        if len(indices) == 0:
            return np.inf
        else:
            dists_ = self.center_dists[indices,:]
            sum_array = np.sum(dists_, axis = 0)
            return np.min(sum_array)
    
    

class CentroidTree(Tree):
    """
    Inherits from the Tree class to implement a decision tree in which 
    axis aligned split criterion are chosen so that points in any leaf node are close
    in distance to their closest center or centroid, from a set of input centers.
    
    When input centers are those from the output of a k-means algorithm, this is 
    equivalent to the ExKMC algorithm, as designed by [Frost, Moshkovitz, Rashtchian '20]
    in their paper titled: 'ExKMC: Expanding Explainable k-Means Clustering.'
    
    Args:
        centers (np.ndarray, optional): Input list of reference centers to calculate cost with. 
            Defaults to None.
            
        norm (int, optional): Takes values 1 or 2. If norm = 1, compute distances using the 
            1 norm. If 2, compute distances with the squared two norm. Defaults to 2.
            
        base_tree (Node, optional): Root node of a baseline tree to start from. 
                Defaults to None, in which case the tree is grown from root.
            
        max_leaf_nodes (int, optional): Optional constraint for maximum number of leaf nodes. 
            Defaults to None.
            
        max_depth (int, optional): Optional constraint for maximum depth. 
            Defaults to None.
            
        min_points_leaf (int, optional): Optional constraint for the minimum number of points. 
            within a single leaf. Defaults to 1.
            
        feature_labels (List[str]): Iterable object with strings representing feature names. 
            
            
    Attributes:
        root (Node): Root node of the tree.
        
        heap (heapq list): Maintains the heap structure of the tree.
        
        leaf_count (int): Number of leaves in the tree.
        
        node_count (int): Number of nodes in the tree.
            
        depth (int): The maximum depth of the tree.
                
    """
    
    def __init__(
        self,
        centers : npt.NDArray = None,
        norm : int = 2,
        base_tree : Node = None,
        max_leaf_nodes : int = None,
        max_depth : int = None,
        min_points_leaf : int = 1,
        feature_labels : List[str] = None
    ):
        splitter = CentroidSplitter(
            centers = centers,
            norm = norm,
            min_points_leaf = min_points_leaf
        )  
        super().__init__(
            splitter = splitter,
            base_tree = base_tree, 
            max_leaf_nodes = max_leaf_nodes,
            max_depth = max_depth, 
            min_points_leaf = min_points_leaf,
            feature_labels = feature_labels
        )