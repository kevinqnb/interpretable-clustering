from ._tree import Tree
from .._node import Node

class RandomTree(Tree):
    """
    Implements a tree that randomly selects splits from the dataset.
    
    Attributes:
        min_points_leaf (int): Minimum number of points in a leaf node.
    """
    def __init__(
        self,
        base_tree : Node = None,
        max_leaf_nodes : int = None,
        max_depth : int = None,
        min_points_leaf : int = 1
    ):
        super().__init__(min_points_leaf=min_points_leaf)
    
    def branch(self, node, condition):
        """
        Splits a leaf node into two new leaf nodes based on a random condition.
        
        Args:
            node (Node): The node to split.
            condition (Condition): The condition to apply for splitting.
        """
        # Randomly select a split condition
        pass  # Implement random split logic here
