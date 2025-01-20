import numpy.typing as npt
from typing import List
from intercluster.splitters import CentroidSplitter
from ._node import Node
from ._tree import Tree
    
    

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