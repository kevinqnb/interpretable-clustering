import numpy as np
from numpy.typing import NDArray
from typing import List, Set
from intercluster.utils import unique_labels
from .splitters import SVMSplitter
from .node import Node
from .tree import Tree
    

class SVMTree(Tree):
    """
    Inherits from the Tree class to implement a decision tree in which a single
    linear splitting condition is chosen with SVM to separate data. 
    
    NOTE: For now this is really just designed to split one class from another.
    Attempting to fit datasets with more than two classes will throw an error.
    Since this is is only desinged to produce
    a single split it defaults to max_leaf_nodes =2, max_depth = 1.
    
    Args:            
        base_tree (Node, optional): Root node of a baseline tree to start from. 
                Defaults to None, in which case the tree is grown from root.
            
        max_leaf_nodes (int, optional): Optional constraint for maximum number of leaf nodes. 
            Defaults to None.
            
        max_depth (int, optional): Optional constraint for maximum depth. 
            Defaults to None.
            
        min_points_leaf (int, optional): Optional constraint for the minimum number of points. 
            within a single leaf. Defaults to 1.
            
            
    Attributes:
        root (Node): Root node of the tree.
        
        heap (heapq list): Maintains the heap structure of the tree.
        
        leaf_count (int): Number of leaves in the tree.
        
        node_count (int): Number of nodes in the tree.
            
        depth (int): The maximum depth of the tree.
                
    """
    
    def __init__(
        self,
        base_tree : Node = None,
        min_points_leaf : int = 1
    ):  
        max_leaf_nodes = 2
        max_depth = 1
        
        splitter = SVMSplitter(
            min_points_leaf = min_points_leaf
        )
        super().__init__(
            splitter = splitter,
            base_tree = base_tree, 
            max_leaf_nodes = max_leaf_nodes,
            max_depth = max_depth, 
            min_points_leaf = min_points_leaf
        )
        
    def fit(
        self,
        X : NDArray,
        y : List[Set[int]]
    ):
        """
        Initiates and builds a decision tree around a given dataset. 
        
        NOTE: For now this is really just designed to split one class from another.
        Attempting to fit datasets with more than two classes will throw an error.
        
        Args:
            X (np.ndarray): Input dataset.
            
            y (List[Set[int]], optional): Target labels with two classes.
        """
        if len(unique_labels(y)) > 2:
            raise ValueError("SVMTree is designed to split two classes only.")
        
        # Reset if needed:
        self.heap = []
        self.leaf_count = 0
        self.node_count = 0
            
        # if stopping criteria weren't provided, set to the maximum possible
        if self.max_leaf_nodes is None:
            self.max_leaf_nodes = len(X)
        
        if self.max_depth is None:
            self.max_depth = len(X) - 1
            
        # Initialize dataset and fit Sklearn tree:
        self.X = X
        self.y = y

        super().fit(X, y)