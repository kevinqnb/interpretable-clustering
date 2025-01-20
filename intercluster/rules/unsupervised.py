from typing import List
from intercluster.splitters import UnsupervisedSplitter
from ._node import Node
from ._tree import Tree
        

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
        feature_labels : List[str] = None
    ):
        splitter = UnsupervisedSplitter(
            norm = norm
        )  
        super().__init__(
            splitter = splitter,
            base_tree = base_tree, 
            max_leaf_nodes = max_leaf_nodes,
            max_depth = max_depth,
            feature_labels = feature_labels
        )