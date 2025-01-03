import numpy as np
from ._linear_tree import LinearTree

class UnsupervisedTree(LinearTree):
    """
    Inherits from the Tree class to implement a decision tree in which 
    split criterion are chosen so that points in any leaf node are close
    in distance to their means or medians. 
   
    Args:
        splits (str): May take values 'axis' or 'oblique' that decide on how to compute leaf splits.
        
        max_leaf_nodes (int, optional): Optional constraint for maximum number of leaf nodes. 
            Defaults to None.
            
        max_depth (int, optional): Optional constraint for maximum depth. 
            Defaults to None.
            
        min_points_leaf (int, optional): Optional constraint for the minimum number of points. 
            within a single leaf. Defaults to 1.
            
        norm (int, optional): Takes values 1 or 2. If norm = 1, compute distances using the 
            1 norm. If 2, compute distances with the squared two norm. Defaults to 2.
            
        random_seed (int, optional): Random seed. In the Tree object randomness is only ever 
            used for breaking ties between nodes, or if you are using a RandomTree!
            
        feature_labels (List[str]): Iterable object with strings representing feature names. 
            
            
        Attributes:
            X (np.ndarray): Size (n x m) dataset passed as input in the fitting process.
            
            root (Node): Root node of the tree.
        
            heap (heapq list): Maintains the heap structure of the tree.
            
            leaf_count (int): Number of leaves in the tree.
            
            node_count (int): Number of nodes in the tree.
                
        """
    def __init__(self, splits = 'axis', max_leaf_nodes=None, max_depth=None, min_points_leaf = 1, 
                 norm = 2, random_seed = None, feature_labels = None):
        
        super().__init__(splits = splits, max_leaf_nodes = max_leaf_nodes, max_depth = max_depth, 
                         min_points_leaf = min_points_leaf, norm = norm, random_seed = random_seed,
                         feature_labels = feature_labels)
        
        
    def _cost(self, indices):
        """
        Given a set of indices defining a data subset X_ --
        which may be thought of as a subset of points reaching a given node in the tree --
        compute a cost for X_.
        
        In an unsupervised tree this amounts to find the sum of distances to 
        means or medians (using squared 2 norm or 1 norm respectively).

        Args:
            indices (np.ndarray[int]): Indices of data points to compute cost with. 
            
        Returns:
            (float): Cost of the given data. 
        """
        
        if len(indices) == 0:
            return np.inf
        else:
            if self.norm == 2:
                X_ = self.X[indices,:]
                mu = np.mean(X_, axis = 0)
                cost = np.sum(np.linalg.norm(X_ - mu, axis = 1)**2) #/len(X_)
                
            elif self.norm == 1:
                X_ = self.X[indices, :]
                eta = np.median(X_, axis = 0)
                cost = np.sum(np.abs(X_ - eta))
                
            return cost