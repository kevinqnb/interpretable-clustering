import numpy as np
from ShallowTree.ShallowTree import ShallowTree as ShallowTree_
from typing import List, Set, Callable
from numpy.typing import NDArray
import warnings
from intercluster import Condition, LinearCondition
from intercluster import (
    Node,
    mode,
    labels_format,
    can_flatten,
    flatten_labels,
    traverse,
    get_decision_paths,
    satisfies_path
)
from .splitters import InformationGainSplitter, DummySplitter, ObliqueInformationGainSplitter
from .tree import Tree


####################################################################################################   


class ShallowTree(Tree):
    """
    Class designed to interface with the ShallowTree package to create shallow decision trees.
    Please see https://github.com/lmurtinho/ShallowTree/tree/main for more details. 
    """
    
    def __init__(
        self,
        n_clusters : int,
        depth_factor : float = None,
        kmeans_random_state : int = None
    ):
        """
         Args:
            n_clusters (int): Number of clusters to form as well as the number of centroids to 
                generate. This is the 'k' parameter in k-means.
            
            depth_factor (float, optional): Factor used to control the maximum depth of the tree.
                If None, defaults to a value of 0.03. Defaults to None.

            kmeans_random_state (int, optional): Random state used to initialize the k-means
                algorithm. If None, a new run of k-means will be done internally. Defaults to None.
            
        Attributes:
            root (Node): Root node of the tree.
        
            heap (heapq list): Maintains the heap structure of the tree.
            
            leaf_count (int): Number of leaves in the tree.
            
            node_count (int): Number of nodes in the tree.
                
            depth (int): The maximum depth of the tree.
        """
        if not isinstance(n_clusters, int) or n_clusters <= 1:
            raise ValueError("n_clusters must be an integer greater than 1.")
        self.n_clusters = n_clusters

        if depth_factor is not None and (not isinstance(depth_factor, float) or depth_factor < 0):
            raise ValueError("depth_factor must be a positive float or None.")
        self.depth_factor = depth_factor

        if kmeans_random_state is not None and not isinstance(kmeans_random_state, int):
            raise ValueError("kmeans_random_state must be an integer or None.")
        elif kmeans_random_state is None:
            warnings.warn(
                "kmeans_random_state is not set, in which case a new run of KMeans will be "
                "done internally.",
                UserWarning
            )
        self.kmeans_random_state = kmeans_random_state
        splitter = DummySplitter()

        super().__init__(
            splitter = splitter
        )
        
            
    def fit(
        self,
        X : NDArray,
        y : List[Set[int]] = None
    ):
        """
        Fits a Sklearn tree to a dataset X and labels y.
        
        Args:
            X (np.ndarray): Input dataset.
            
            y (List[Set[int]], optional): Target labels. Defaults to None.
        """ 
        # Reset if needed:
        self.heap = []
        self.leaf_count = 0
        self.node_count = 0
        
        # if stopping criteria weren't provided, set to the maximum possible
        if self.max_leaf_nodes is None:
            self.max_leaf_nodes = len(X)
        
        if self.max_depth is None:
            self.max_depth = len(X) - 1
            
        # Initialize dataset and fit Sklearn tree:
        self.X = X
        self.y = y
        
        if (y is not None) and can_flatten(y):
            self.y_array = flatten_labels(y)
        elif y is not None:
            raise ValueError("Each data point must have exactly one label.")
        else:
            self.y_array = None
        
        self.shallow_tree = ShallowTree_(
            k = self.n_clusters,
            depth_factor = self.depth_factor,
            random_state = self.kmeans_random_state,
        )
        self.shallow_tree.fit(x_data = X)

        self.root = Node()
        indices = np.arange(len(X))
        self.grow(X, indices, self.shallow_tree.tree, self.root, 0)
        self.node_count += 1
        
    
    def grow(
        self,
        X : NDArray,
        indices : NDArray,
        shtree_node : Callable,
        node_obj: Node,
        depth : int
    ):
        """
        Traverses through the ExKMC tree.
        
        Args:
            X (np.ndarray): Input dataset.
            
            indices (np.ndarray): Subset of data indices to build node with.
            
            shtree_node (ExKMC.Tree.Node): Node of a trained shallow tree.
            
            node_obj (Node): Corresponding Node object to copy conditions into.  
            
            depth (int): current depth of the tree
        """
        X_ = X[indices, :]
        if depth > self.depth:
            self.depth = depth
        
        if shtree_node.is_leaf():
            class_label = shtree_node.value
            node_obj.leaf_node(
                leaf_num = self.leaf_count,
                label = class_label,
                cost = -1,
                indices = indices,
                depth = depth
            )
            self.leaf_count += 1
        else:
            feature, threshold = shtree_node.feature, shtree_node.value
            left_mask = X_[:, feature] <= threshold
            right_mask = ~left_mask
            
            left_child = Node()
            right_child = Node()
            
            condition = LinearCondition(
                features = np.array([feature]),
                weights = np.array([1]),
                threshold = threshold,
                direction = -1
            )
            
            node_obj.tree_node(
                left_child = left_child,
                right_child = right_child,
                condition = condition,
                cost = -1,
                indices = indices,
                depth = depth
            )
            
            self.grow(
                X,
                indices[left_mask],
                shtree_node.left,
                left_child,
                depth + 1
            )
            self.grow(
                X,
                indices[right_mask],
                shtree_node.right,
                right_child,
                depth + 1
            )
            
            self.node_count += 2

    
    def predict(
        self,
        X : NDArray,
        leaf_labels : bool = False
    ) -> List[Set[int]]:
        """
        Predicts the labels of a dataset X.
        
        Args:
            X (np.ndarray): Input dataset.
            
            leaf_labels (bool, optional): If true, gives labels based soley upon 
                leaf membership. Otherwise, returns the orignal predictions from 
                the fitted tree. Defaults to True.  
                
        Returns:
            List[Set[int]]: List of label sets where the set at index i contains class 
                labels for data point i.

        """
        if leaf_labels:
            labels = [set() for _ in range(len(X))]
            decision_paths = get_decision_paths(self.root)
            for path in decision_paths:
                leaf = path[-1]
                satisfies = satisfies_path(X, path)
                for idx in satisfies:
                    labels[idx].add(int(leaf.leaf_num))
            return labels
        else:
            return labels_format(self.shallow_tree.predict(X).astype(int))
            
    
####################################################################################################