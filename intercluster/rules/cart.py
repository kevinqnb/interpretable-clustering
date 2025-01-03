import numpy as np
from sklearn.tree import DecisionTreeClassifier
from ._node import Node
from ._tree import Tree


class SklearnTree(Tree):
    """
    Class designed to interface with an Sklearn Tree object. 
    For more information about Sklearn trees, please visit 
    (https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html
    #sphx-glr-auto-examples-tree-plot-unveil-tree-structure-py)
    
    
    """
    
    def __init__(
        self,
        data_labels = None,
        max_leaf_nodes=None,
        max_depth=None,
        min_points_leaf=1,
        random_seed=None,
        feature_labels=None 
    ):
        """
         Args:
            max_leaf_nodes (int, optional): Optional constraint for maximum number of leaf nodes. 
                Defaults to None.
                
            max_depth (int, optional): Optional constraint for maximum depth. 
                Defaults to None.
                
            min_points_leaf (int, optional): Optional constraint for the minimum number of points. 
                within a single leaf. Defaults to 1.
                
            random_seed (int, optional): Random seed. In the Tree object randomness is only ever 
                used for breaking ties between nodes, or if you are using a RandomTree!
                
            feature_labels (List[str]): Iterable object with strings representing feature names. 
            
            
        Attributes:
            X (np.ndarray): Size (n x m) dataset passed as input in the fitting process.
            
            root (Node): Root node of the tree.
        
            heap (heapq list): Maintains the heap structure of the tree.
            
            leaf_count (int): Number of leaves in the tree.
            
            node_count (int): Number of nodes in the tree.
            
            n_centers (int): The number of centers to use for computation of cost 
                or center updates.
                
            depth (int): The maximum depth of the tree.
        """

        super().__init__(
            max_leaf_nodes=max_leaf_nodes,
            max_depth=max_depth,
            min_points_leaf=min_points_leaf,
            random_seed=random_seed,
            feature_labels=feature_labels
        )
        
        self.data_labels = data_labels
        
        
        
    def _cost(self, indices):
        """
        Assigns cost that rewards data with points all close to their mean
        (i.e. small variance).

        Args:
            indices (np.ndarray[int]): Indices of data points to compute cost with. 
            
        Returns:
            (float): Cost of the given data. 
        """
        return 0
        
        
    def fit(self, X):
        self.X = X
        self.label = np.random.choice(self.data_labels)
        y = (self.data_labels == self.label).astype(int)
        
        # Set feature labels if not set already:
        if self.feature_labels is None:
            self.feature_labels = [None]*X.shape[1]
        else:
            if not len(self.feature_labels) == X.shape[1]:
                raise ValueError('Feature labels must match the shape of the data.')
        
        #not_clusterable = False
        #while not not_clusterable:
        self.SklearnT = DecisionTreeClassifier(
            criterion = 'entropy',
            max_depth = self.max_depth,
            min_samples_leaf = self.min_points_leaf,
            max_leaf_nodes = self.max_leaf_nodes
        )
        self.SklearnT.fit(X, y)
        self.sklearn_tree = self.SklearnT.tree_
        #    not_clusterable = True
        #else:
        #    print('Tree not clusterable, retrying...')
        
        self.root = Node()
        self._convert_tree(0, self.root, np.array(range(len(X))), 0)
        self.node_count += 1
        
    def fit_step(self):
        pass
    
    def split_leaf(self):
        pass
        
    def _convert_tree(self, node, node_obj, indices, depth):
        """
        Builds the decision tree by traversing the Sklearn tree.
        
        Args:
            node (int): Node of a new Sklearn tree.
            
            node_obj (Node): Corresponding Node object to copy conditions into. 
            
            indices (np.ndarray[int]): Subset of data indices to build node with. 
        """
        X_ = self.X[indices, :]
        y_ = self.data_labels[indices]
        
        if depth > self.depth:
            self.depth = depth
        
        if (self.sklearn_tree.children_left[node] < 0 and 
            self.sklearn_tree.children_right[node] < 0):
            class_label = self.leaf_count # This may be important...might need to change...
            self.leaf_count += 1
            node_obj.leaf_node(label = class_label, indices = indices, cost = self._cost(indices))
            
            if np.sum(y_ == self.label)/len(y_) < 0.5:
                node_obj.type = 'bad-leaf'
        else:
            feature, threshold = self.sklearn_tree.feature[node], self.sklearn_tree.threshold[node]
            left_mask = X_[:, feature] <= threshold
            right_mask = ~left_mask
            
            left_node = Node()
            right_node = Node()
            
            node_obj.tree_node([feature], [1], threshold, left_node, right_node, 
                               indices, self._cost(indices), 
                               feature_labels = [self.feature_labels[feature]])
            
            self._convert_tree(self.sklearn_tree.children_left[node], left_node, indices[left_mask], depth + 1)
            self._convert_tree(self.sklearn_tree.children_right[node], right_node, indices[right_mask], depth + 1)
            
            self.node_count += 2