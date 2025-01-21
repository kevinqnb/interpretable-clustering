import numpy as np
import numpy.typing as npt
from typing import List
from intercluster.splitters import SVMSplitter
from ._node import Node
from ._tree import Tree
    

class SVMTree(Tree):
    """
    Inherits from the Tree class to implement a decision tree in which 
    linear split criterion are chosen with SVM to separate data. 
    
    NOTE: For now this is really just designed to split one class from another.
    Attempting to fit datasets with more than two classes will throw an error.
    I also think it is best utilized with max_depth = 1.
    
    Args:            
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
        base_tree : Node = None,
        max_leaf_nodes : int = None,
        max_depth : int = None,
        min_points_leaf : int = 1,
        feature_labels : List[str] = None
    ):
        splitter = SVMSplitter(
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
        
    def fit(
        self,
        X : npt.NDArray,
        y : npt.NDArray
    ):
        """
        Initiates and builds a decision tree around a given dataset. 
        
        NOTE: For now this is really just designed to split one class from another.
        Attempting to fit datasets with more than two classes will throw an error.
        
        Args:
            X (np.ndarray): Input dataset.
            
            y (np.ndarray, optional): Target labels with two classes.
        """
        if len(np.unique(y)) > 2:
            raise ValueError("SVMTree is designed to split two classes only.")

        super().fit(X, y)