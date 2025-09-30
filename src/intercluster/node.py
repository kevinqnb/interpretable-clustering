from numpy.typing import NDArray
from intercluster import Condition

class Node():
    """
    Node object to be used within a logical decision tree, with simple 
    axis aligned splitting conditions.
    
    Args:
        None
    Attributes:
        random_val (float): Uniform random sample used to split ties / and create
            a relative order of nodes. Note that in the creation of the tree, nodes are chosen
            based on their cost, and this value is simply used to break ties. 
            
        type (str): Internally set to 'node' or 'leaf' depending on if the node is a
            normal node or a leaf node.

        leaf_num (int): (For leaf nodes only) Leaf identifying number.
            
        label (int): (For leaf nodes only) Prediction label to be associated with this node.
        
        left_child (Node): (For non-leaf nodes only) Pointer to the left branch of the current node.
        
        right_child (Node): (For non-leaf nodes only) Pointer to the right branch of the 
            current node. 
        
        condition (Condition): Logical or functional condition for evaluating and 
                splitting the data points.
        
        cost (float): The cost associated with points belonging to this node.
            
        indices (np.ndarray): The subset of data indices from the training set 
            belonging to this node.
        
        depth (int): The depth of the current node in the tree.
        
        centroid_indices (np.ndarray): Indices of the cluster centers belonging to this node.
    """
    
    def __init__(self):
        #self.random_val = np.random.uniform()
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
        """
        Initializes this as a normal node in the tree.

        Args:
            left_child (Node): Pointer to the left child of the current node.
            
            right_child (Node): Pointer to the right child of the current node. 
            
            condition (Condition): Logical or functional condition for evaluating and 
                splitting the data points.
            
            cost (float): The cost associated with points belonging to this node.
            
            indices (np.ndarray): The subset of data indices from the training set 
                belonging to this node.
            
            depth (int): The depth of the current node in the tree.
            
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
        """
        Initializes this to be a leaf node in the tree.

        Args:
            leaf_num (int): Leaf identifying number.

            label (int): Prediction label to be associated with this node.
            
            cost (float): The cost associated with points belonging to this node. 
            
            indices (np.ndarray): The subset of data indices from the training set 
                belonging to this node.
            
            depth (int): The depth of the current node in the tree.
            
            centroid_indices (np.ndarray): Indices of the cluster centers belonging to this node.
        """
        self.type = 'leaf'
        self.leaf_num = leaf_num
        self.label = label
        self.cost = cost 
        self.indices = indices
        self.depth = depth
        self.centroid_indices = centroid_indices