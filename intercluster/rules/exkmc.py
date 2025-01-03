import numpy as np
from ._node import Node
from ._tree import Tree

class ConvertExKMC(Tree):
    """
    Transforms an ExKMC tree to a Tree object as defined here (For visualization purposes). 
    The ExKMC tree is based around work 
    from work from [Frost, Moshkovitz, Rashtchian '20] in their 
    paper titled 'ExKMC: Expanding Explainable k-Means Clustering.'
    The following works by examining the tree created in their implementation, 
    which may be found at: https://github.com/navefr/ExKMC.
    
    NOTE: The fit() method of the parent class will not apply. Any fitting must be done 
    with the ExKMC code. This class simply builds a Tree object out of the already 
    fitted tree which output from their code. 
    """
    
    def __init__(self, ExKMC_root, X, feature_labels = None):
        """
        Args:
            ExKMC_root (ExKMC.Tree.Node): Root node of an ExKMC tree.
            
            feature_labels(List[str], optional): List of strings corresponding to feature names 
                in the data. Useful for explaining results in post. 
        """

        super().__init__()
        self.exkmc_root = ExKMC_root
        self.X = X
        
        # Set feature labels:
        if feature_labels is None:
            self.feature_labels = [None]*X.shape[1]
        else:
            if not len(feature_labels) == X.shape[1]:
                raise ValueError('Labels must match the shape of the data.')
            self.feature_labels = feature_labels
        
        
        self.root = Node()
        self._build_tree(self.exkmc_root, self.root, np.array(range(len(X))), 0)
        self.node_count += 1
        
        
    def _cost(self, indices):
        """
        Assigns cost that rewards data with points all close to their mean
        (i.e. small variance).

        Args:
            indices (np.ndarray[int]): Indices of data points to compute cost with. 
            
        Returns:
            (float): Cost of the given data. 
        """
        
        #if len(X_) == 0:
        #    return np.inf
        #else:
        #    mu = np.mean(X_, axis = 0)
        #    cost = np.sum(np.linalg.norm(X_ - mu, axis = 1)**2) #/len(X_)
        #    return cost
        return 0
        
    # The following methods become irrelevant for this class.
    def fit(self, X):
        self.X = X
        pass
    
    def fit_step(self):
        pass
    
    def split_leaf(self):
        pass
        
    def _build_tree(self, exkmc_node, node_obj: Node, indices, depth):
        """
        Builds the decision tree by traversing the ExKMC tree.
        
        Args:
            exkmc_node (ExKMC.Tree.Node): Node of an ExKMC tree.
            
            node_obj (Node): Corresponding Node object to copy conditions into. 
            
            indices (np.ndarray[int]): Subset of data indices to build node with. 
            
            depth (int): current depth of the tree
        """
        X_ = self.X[indices, :]
        if depth > self.depth:
            self.depth = depth
        
        if exkmc_node.is_leaf():
            class_label = self.leaf_count 
            self.leaf_count += 1
            node_obj.leaf_node(label = class_label, indices = indices, cost = self._cost(indices))
        else:
            feature, threshold = exkmc_node.feature, exkmc_node.value
            left_mask = X_[:, feature] <= threshold
            right_mask = ~left_mask
            
            left_node = Node()
            right_node = Node()
            
            node_obj.tree_node([feature], [1], threshold, left_node, right_node, 
                               indices, self._cost(indices),
                               feature_labels = [self.feature_labels[feature]])
            
            self._build_tree(exkmc_node.left, left_node, indices[left_mask], depth + 1)
            self._build_tree(exkmc_node.right, right_node, indices[right_mask], depth + 1)
            
            self.node_count += 2