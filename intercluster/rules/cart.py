import numpy as np
from sklearn.tree import DecisionTreeClassifier
from typing import List, Callable
import numpy.typing as npt
from ._node import Node
from ._splitter import Splitter
from ._tree import Tree
from .utils import *
from ..utils import mode


class SklearnTree(Tree):
    """
    Class designed to interface with an Sklearn Tree object. 
    For more information about Sklearn trees, please visit 
    (https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html
    #sphx-glr-auto-examples-tree-plot-unveil-tree-structure-py)
    """
    
    def __init__(
        self,
        criterion : str = 'entropy',
        max_leaf_nodes=None,
        max_depth=None,
        min_points_leaf=1,
        feature_labels=None 
    ):
        """
         Args:
            criterion (str): The function to measure the quality of a split. Supported criteria
                are 'gini' for the Gini impurity and 'entropy' for the information gain.
                
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
        self.criterion = criterion
        splitter = Splitter()
        super().__init__(
            splitter = splitter,
            max_leaf_nodes=max_leaf_nodes,
            max_depth=max_depth,
            min_points_leaf=min_points_leaf,
            feature_labels=feature_labels
        )
        
            
    def fit(
        self,
        X : npt.NDArray,
        y : npt.NDArray = None
    ):
        """
        Fits a Sklearn tree to a dataset X and labels y.
        
        Args:
            X (np.ndarray): Input dataset.
            
            y (np.ndarray, optional): Target labels. Defaults to None.
        """        
        # Reset if needed:
        self.heap = []
        self.leaf_count = 0
        self.node_count = 0
        
        # Set feature labels if not set already:
        if self.feature_labels is None:
            self.feature_labels = [None]*X.shape[1]
        else:
            if not len(self.feature_labels) == X.shape[1]:
                raise ValueError('Feature labels must match the shape of the data.')
        
        # if stopping criteria weren't provided, set to the maximum possible
        if self.max_leaf_nodes is None:
            self.max_leaf_nodes = len(X)
        
        if self.max_depth is None:
            self.max_depth = len(X) - 1
        
        
        self.sklearn_tree = DecisionTreeClassifier(
            criterion = self.criterion,
            max_depth = self.max_depth,
            min_samples_leaf = self.min_points_leaf,
            max_leaf_nodes = self.max_leaf_nodes
        )
        self.sklearn_tree.fit(X, y)
        self.tree_info = self.sklearn_tree.tree_
        self.root = Node()
        indices = np.arange(len(X))
        self.grow(X, y, indices, 0, self.root, 0)
        self.node_count += 1
        
        
    def grow(
        self,
        X : npt.NDArray,
        y : npt.NDArray,
        indices : npt.NDArray,
        sklearn_node : int,
        node_obj : Node, 
        depth : int
    ):
        """
        Builds the decision tree by traversing the Sklearn tree.
        
        Args:
            X (np.ndarray): Input dataset.
            
            y (np.ndarray): Data labels.
            
            indices (np.ndarray): Subset of data indices to build node with.
            
            sklearn_node (int): Integer index of node in sklearn tree.
            
            node_obj (Node): Corresponding Node object to copy conditions into.  
            
            depth (int): current depth of the tree
        """
        X_ = X[indices, :]
        y_ = y[indices]
        
        if depth > self.depth:
            self.depth = depth
        
        if (self.tree_info.children_left[sklearn_node] < 0 and 
            self.tree_info.children_right[sklearn_node] < 0):
            class_label = self.leaf_count # This may be important...might need to change...
            self.leaf_count += 1
            node_obj.leaf_node(
                label = class_label,
                score = -1,
                indices = indices,
                depth = depth
            )

        else:
            feature = self.tree_info.feature[sklearn_node]
            threshold = self.tree_info.threshold[sklearn_node]
            left_mask = X_[:, feature] <= threshold
            right_mask = ~left_mask
            
            left_child = Node()
            right_child = Node()
            
            node_obj.tree_node(
                left_child = left_child,
                right_child = right_child,
                features = [feature],
                weights = [1],
                threshold = threshold,
                score = -1,
                indices = indices,
                depth = depth,
                feature_labels = [self.feature_labels[feature]]
            )
            
            self.grow(
                X,
                y,
                indices[left_mask],
                self.tree_info.children_left[sklearn_node],
                left_child,
                depth + 1
            )
            
            self.grow(
                X,
                y, 
                indices[right_mask],
                self.tree_info.children_right[sklearn_node],
                right_child,
                depth + 1
            )
            
            self.node_count += 2
            
    def predict(
        self,
        X : npt.NDArray,
        leaf_labels : bool = True
    ) -> npt.NDArray:
        """
        Predicts the labels of a dataset X.
        
        Args:
            X (np.ndarray): Input dataset.
            
            leaf_labels (bool, optional): If true, gives labels based soley upon 
                leaf membership. That is, each leaf is given a unique label. 
                Otherwise, returns the orignal predictions from 
                the fitted tree. Defaults to True.  

        """
        if leaf_labels:
            labels = np.zeros(X.shape[0])
            decision_paths = get_decision_paths(self.root)
            for path in decision_paths:
                leaf = path[-1][0]
                satisfies = satisfies_path(X, path)
                labels[satisfies] = leaf.label
                
            return labels
        else:
            return self.sklearn_tree.predict(X)
        
    
    def get_leaves(self, y : npt.NDArray = None, label : int = None) -> List[Node]:
        """
        Returns the leaf nodes of the tree. If an array y of training data labels AND a 
        specific label are provided, only the leaf nodes with that specified label are returned.
        
        Args:
            y (np.ndarray, optional): Training Data labels. Defaults to None.
            
            label (int, optional): Label to filter by. Defaults to None.
        Returns:
            leaves (List[Node]): List of leaf nodes in the tree. 
        """
        
        leaves = []
        for path in traverse(self.root):
            last_node = path[-1][0]
            if last_node.type == 'leaf':
                if y is not None and label is not None:
                    if mode(y[last_node.indices[0]]) == label:
                        leaves.append(last_node)
                else:
                    leaves.append(last_node)
                
        return leaves