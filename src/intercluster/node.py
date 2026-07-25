from numpy.typing import NDArray
from intercluster import Condition

class Node():
    """Node in a logical decision tree, with simple axis-aligned splitting conditions.

    Call `tree_node()` or `leaf_node()` after construction to initialize as an internal
    or leaf node.

    Attributes:
        type (str): 'node' or 'leaf'.
        leaf_num (int): (Leaf nodes only) Leaf identifying number.
        label (int): (Leaf nodes only) Prediction label associated with this node.
        left_child (Node): (Internal nodes only) Left branch.
        right_child (Node): (Internal nodes only) Right branch.
        condition (Condition): (Internal nodes only) Condition for splitting the data points.
        cost (float): Cost associated with points belonging to this node.
        indices (np.ndarray): Data indices from the training set belonging to this node.
        depth (int): Depth of this node in the tree.
        centroid_indices (np.ndarray): Indices of the cluster centers belonging to this node.
    """

    def __init__(self):
        self.type = None
        self.leaf_num = None
        self.label = None
        self.left_child = None
        self.right_child = None
        self.condition = None
        self.cost = None
        self.indices = None
        self.depth = None
        self.centroid_indices = None
    
    def tree_node(
        self,
        left_child,
        right_child,
        condition : Condition,
        cost : float, 
        indices : NDArray,
        depth : int,
        centroid_indices : NDArray = None
    ):
        """Initializes this as an internal node in the tree.

        Args:
            left_child (Node): Left child of this node.
            right_child (Node): Right child of this node.
            condition (Condition): Condition for splitting the data points.
            cost (float): Cost associated with points belonging to this node.
            indices (np.ndarray): Data indices from the training set belonging to this node.
            depth (int): Depth of this node in the tree.
            centroid_indices (np.ndarray): Indices of the cluster centers belonging to this node.
        """
        self.type = 'internal'
        self.leaf_num = None
        self.label = None
        self.left_child = left_child 
        self.right_child = right_child
        self.condition = condition
        self.cost = cost
        self.indices = indices
        self.depth = depth
        self.centroid_indices = centroid_indices
        
        
    def leaf_node(
        self,
        leaf_num : int,
        label : int,
        cost : float,
        indices : NDArray,
        depth : int,
        centroid_indices : NDArray = None
    ):
        """Initializes this as a leaf node in the tree.

        Args:
            leaf_num (int): Leaf identifying number.
            label (int): Prediction label associated with this node.
            cost (float): Cost associated with points belonging to this node.
            indices (np.ndarray): Data indices from the training set belonging to this node.
            depth (int): Depth of this node in the tree.
            centroid_indices (np.ndarray): Indices of the cluster centers belonging to this node.
        """
        self.type = 'leaf'
        self.leaf_num = leaf_num
        self.label = label
        self.cost = cost 
        self.indices = indices
        self.depth = depth
        self.centroid_indices = centroid_indices