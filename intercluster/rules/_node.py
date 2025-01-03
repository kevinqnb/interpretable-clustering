import numpy as np

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
            
        label (int): (For leaf nodes only) Prediction label to be associated with this node.
        
        features (List[int]): The axis aligned feature to split on. 
        
        feature_labels (List[str]): Names for the given feature (for printing and display). 
        
        features (List[int]): The axis aligned features to split on. 
        
        weights (List[float]): Weights for each of the splitting features
        
        threshold (float): The threshold value to split on.
        
        left_child (Node): (For non-leaf nodes only) Pointer to the left branch of the current node.
        
        right_child (Node): (For non-leaf nodes only) Pointer to the right branch of the 
            current node. 
            
        indices (np.ndarray): The indices of data points belonging to this node.
        
        size (int): The number of data points belonging to this node
        
        cost (float): The cost associated with points belonging to this node. 
        
    """
    def __init__(self):
        self.random_val = np.random.uniform()
        self.type = None
        self.label = None
        self.features = None
        self.feature_labels = None
        self.weights = None
        self.threshold = None
        self.left_child = None
        self.right_child = None
        self.indices = None
        self.size = None
        self.cost = None
    
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
    
    def tree_node(self, features, weights, threshold, left_child, right_child, 
                  indices, cost, feature_labels):
        """
        Initializes this as a normal node in the tree.

        Args:
            features (List[int]): The axis aligned features to split on. 
        
            weights (List[float]): Weights for each of the splitting features
        
            threshold (float): The threshold value to split on.
            
            left_child (Node): Pointer to the left branch of the current node.
            
            right_child (Node): Pointer to the right branch of the 
                current node. 
                
            indices (np.ndarray): The subset of data indices belonging to this node.
             
            cost (float): The cost associated with points belonging to this node. 
            
            feature_label (str): Name for the given feature (for printing and display). 
        """
        self.type = 'node'
        self.features = features
        self.weights = weights
        self.threshold = threshold
        self.left_child = left_child 
        self.right_child = right_child
        self.indices = indices
        self.cost = cost
        self.feature_labels = feature_labels
        self.size = len(indices)
        
        # reset label if this was previously a leaf node
        self.label = None
        
    def leaf_node(self, label, indices, cost):
        """
        Initializes this to be a leaf node in the tree.

        Args:
            label (int): Prediction label to be associated with this node.
            
            indices (np.ndarray): The subset of data indices belonging to this node.
            
            cost (float): The cost associated with points belonging to this node. 
        """
        self.type = 'leaf'
        self.label = label
        self.indices = indices
        self.cost = cost
        self.size = len(indices)