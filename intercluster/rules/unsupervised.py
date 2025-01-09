import numpy as np
import numpy.typing as npt
from typing import List, Tuple
from ._splitter import AxisAlignedSplitter
from ._node import Node
from ._tree import Tree


class UnsupervisedSplitter(AxisAlignedSplitter):
    """
    Splits leaf nodes in order to minimize distances to a set of input centers.
    """
    def __init__(
        self,
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
        self.norm = norm
        super().__init__(min_points_leaf = min_points_leaf)
        
    def score(
        self,
        X : npt.NDArray,
        y : npt.NDArray = None,
        indices : npt.NDArray = None
    ) -> float:
        """
        Given a set of points X, computes the score as the sum of distances to 
        the closest center.
        
        Args:
            X (npt.NDArray): Array of points to compute score with.
            
            y (npt.NDArray, optional): Array of labels. Dummy variable, not used 
                for this class. Defaults to None.
                
            indices (npt.NDArray, optional): Indices of points to compute score with.
                
        Returns:
            (float): Score of the given data.
        """
        if len(X) == 0:
            return np.inf
        else:
            if self.norm == 2:
                mu = np.mean(X, axis = 0)
                cost = np.sum(np.linalg.norm(X - mu, axis = 1)**2) #/len(X_)
                
            elif self.norm == 1:
                eta = np.median(X, axis = 0)
                cost = np.sum(np.abs(X - eta))
                
            return cost
        
    def split(
        self,
        X : npt.NDArray,
        y : npt.NDArray = None,
        indices : npt.NDArray = None
    ) -> Tuple[float, Tuple[npt.NDArray, npt.NDArray, float]]:
        """
        Given a set of points X, computes the best split of the data.
        
        The following optimized version is attributed to 
        [Dasgupta, Frost, Moshkovitz, Rashtchian '20] in their paper
        'Explainable k-Means and k-Medians Clustering' 
        
        Args:
            X (npt.NDArray): Array of points to compute split with.
            
            y (npt.NDArray, optional): Array of labels. Dummy variable, not used 
                for this class. Defaults to None.
                
            indices (npt.NDArray, optional): Indices of points to compute split with.
                
        Returns:
            split_info ((np.ndarray, np.ndarray, float)): Features, weights,
                and threshold of the split.
        """
        if self.norm == 1:
            return super().split(X, y, indices)
        else:
            """
            The following optimized version is attributed to 
            [Dasgupta, Frost, Moshkovitz, Rashtchian '20] in their paper
            'Explainable k-Means and k-Medians Clustering' 
            """
            n, d = X.shape
            u = np.linalg.norm(X)**2
            
            best_split = None
            best_split_val = np.inf
            for i in range(X.shape[1]):
                s = np.zeros(d)
                r = np.sum(X, axis = 0)
                order = np.argsort(X[:, i])
                
                for j, idx in enumerate(order[:-1]):
                    threshold = X[idx, i]
                    s = s + X[idx, :]
                    r = r - X[idx, :]
                    split_val = u - np.sum(s**2)/(j + 1) - np.sum(r**2)/(n - j - 1)
                    
                    if split_val < best_split_val:
                        best_split_val = split_val
                        # Axis aligned split:
                        best_split = ([i], [1], threshold)
                        
            return best_split_val, best_split

class UnsupervisedTree(Tree):
    """
    Inherits from the Tree class to implement a decision tree in which 
    axis aligned split criterion are chosen so that points in any leaf node are close
    in distance to their mean or median.
    
    Args:            
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
        norm : int = 2,
        base_tree : Node = None,
        max_leaf_nodes : int = None,
        max_depth : int = None,
        min_points_leaf : int = 1,
        feature_labels : List[str] = None
    ):
        splitter = UnsupervisedSplitter(
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