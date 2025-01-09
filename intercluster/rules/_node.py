import numpy as np
import numpy.typing as npt
from typing import List

class Node():
    """
    Node object to be used within a logical decision tree, with simple 
    axis aligned splitting conditions.
    
    Args:
        None
    Attributes:
        random_val (float): Uniform random sample used to split ties / and create
            a relative order of nodes. Note that in the creation of the tree, nodes are chosen
            based on their score, and this value is simply used to break ties. 
            
        type (str): Internally set to 'node' or 'leaf' depending on if the node is a
            normal node or a leaf node.
            
        label (int): (For leaf nodes only) Prediction label to be associated with this node.
        
        left_child (Node): (For non-leaf nodes only) Pointer to the left branch of the current node.
        
        right_child (Node): (For non-leaf nodes only) Pointer to the right branch of the 
            current node. 
        
        features (np.ndarray): The axis aligned feature to split on. 
        
        weights (np.ndarray): Weights for each of the splitting features
        
        threshold (float): The threshold value to split on.
        
        score (float): The score associated with points belonging to this node.
            
        indices (np.ndarray): The subset of data indices from the training set 
            belonging to this node.
        
        depth (int): The depth of the current node in the tree.
        
        feature_labels (List[str]): Names for the given feature (for printing and display). 
    """
    
    def __init__(self):
        #self.random_val = np.random.uniform()
        self.type = None
        self.label = None
        self.left_child = None
        self.right_child = None
        self.features = None
        self.feature_labels = None
        self.weights = None
        self.threshold = None
        self.score = None
        self.indices = None
        self.depth = None
        self.feature_labels = None
    
    '''
    def __lt__(self, other):
        """
        Creates an ordering of nodes by defining a < comparison. 
        This simply order nodes randomly via randomly sampled values. 

        Args:
            other (Node): Another node to compare with.

        Returns:
            (bool): Evaluation of the comparison. 
        """
        return self.random_val < other.random_val
    '''
    
    def tree_node(
        self,
        left_child,
        right_child,
        features : npt.NDArray,
        weights : npt.NDArray,
        threshold : float,
        score : float, 
        indices : npt.NDArray,
        depth : int,
        feature_labels : List[str]
    ):
        """
        Initializes this as a normal node in the tree.

        Args:
            left_child (Node): Pointer to the left child of the current node.
            
            right_child (Node): Pointer to the right child of the current node. 
            
            features (np.ndarray): The features to use for splitting. 
        
            weights (np.ndarray): Weights for each of the splitting features.
        
            threshold (float): The threshold value to split on.
            
            score (float): The score associated with points belonging to this node.
            
            indices (np.ndarray): The subset of data indices from the training set 
                belonging to this node.
            
            depth (int): The depth of the current node in the tree.
            
            feature_labels (List[str]): Names for the given features (for printing and display). 
        """
        self.type = 'internal'
        self.label = None
        self.left_child = left_child 
        self.right_child = right_child
        self.features = features
        self.weights = weights
        self.threshold = threshold
        self.score = score
        self.indices = indices
        self.depth = depth
        self.feature_labels = feature_labels
        
        
    def leaf_node(
        self,
        label : int,
        score : float,
        indices : npt.NDArray,
        depth : int
    ):
        """
        Initializes this to be a leaf node in the tree.

        Args:
            label (int): Prediction label to be associated with this node.
            
            score (float): The score associated with points belonging to this node. 
            
            indices (np.ndarray): The subset of data indices from the training set 
                belonging to this node.
            
            depth (int): The depth of the current node in the tree.
        """
        self.type = 'leaf'
        self.label = label
        self.score = score 
        self.indices = indices
        self.depth = depth